from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import ollama

from config import AppConfig, save_config
from core.executor import Executor, TaskExecution
from core.planner import Planner, TaskList
from core.react_loop import ReactLoop
from core.synthesizer import Synthesizer
from logger import get_logger
from memory.attachment_store import AttachmentStore
from memory.embedder import OllamaEmbedder
from memory.store import MemoryStore
from tools.registry import build_default_registry

_log = get_logger(__name__)


class Agent:
    def __init__(self, cfg: AppConfig):
        self._cfg = cfg
        self._registry = build_default_registry()
        self._embedder = OllamaEmbedder(cfg.embedding_model, cfg.ollama_base_url)
        self._memory = MemoryStore(cfg.db_path, self._embedder)
        self._attachment_store = AttachmentStore(self._embedder, dim=cfg.embedding_dim)
        self._planner = Planner(cfg.ollama_base_url, cfg.planner_model, self._registry)
        self._synthesizer = Synthesizer(cfg.ollama_base_url, cfg.synthesizer_model)
        react_model = cfg.react_model or cfg.planner_model
        from pathlib import Path as _Path
        self._react_loop = ReactLoop(
            base_url=cfg.ollama_base_url,
            model=react_model,
            registry=self._registry,
            cancel_event=threading.Event(),  # replaced in reset_cancel
            max_iterations=cfg.react_max_iterations,
            max_tool_output_chars=cfg.max_tool_output_chars,
            traces_dir=_Path(cfg.traces_dir) if cfg.traces_dir else None,
            max_conversation_history=cfg.max_conversation_history,
            prompt_injection_protection=cfg.prompt_injection_protection,
        )
        self._conversation_history: list[dict] = []
        self._session_image_b64: list[str] = []  # base64 images for vision models
        self._cancel_event = threading.Event()
        self._approval_event = threading.Event()
        self._approved = False
        if cfg.workspace:
            self.set_workspace(cfg.workspace)

    def set_workspace(self, path: str) -> None:
        self._cfg.workspace = path
        save_config(self._cfg)
        for tool_name in ("file_read", "file_write", "file_edit",
                           "shell_run", "dir_tree", "find_files"):
            try:
                self._registry.get(tool_name).workspace = path
            except KeyError:
                pass

    def cancel(self) -> None:
        self._cancel_event.set()
        self._approval_event.set()  # unblock any waiting approval

    def reset_cancel(self) -> None:
        self._cancel_event.clear()
        self._react_loop._cancel = self._cancel_event

    def approve_plan(self) -> None:
        self._approved = True
        self._approval_event.set()

    def reject_plan(self) -> None:
        self._approved = False
        self._cancel_event.set()
        self._approval_event.set()

    def run(
        self,
        user_query: str,
        attachments: list[str] | None = None,
        on_plan_ready: Callable[[TaskList], None] | None = None,
        on_task_update: Callable[[TaskExecution], None] | None = None,
        on_output_chunk: Callable[[str], None] | None = None,
        on_answer_chunk: Callable[[str], None] | None = None,
        require_approval: bool = True,
    ) -> str:
        self.reset_cancel()
        self._approval_event = threading.Event()
        self._approved = False

        _log.info("━" * 50)
        _log.info("USER QUERY: %s", user_query)
        if attachments:
            _log.info("ATTACHMENTS: %s", attachments)
        _log.info("━" * 50)

        # 0. Preprocess attachments → files_context (hybrid: direct inject or RAG)
        self._session_image_b64.clear()  # reset for this query; repopulated by _handle_image
        files_context = ""
        if attachments:
            files_context = self._preprocess_attachments(attachments)
            _log.debug("FILES CONTEXT (%d chars):\n%s", len(files_context), files_context[:500])

        # If session RAG store has indexed files, retrieve relevant chunks
        if self._attachment_store.has_data():
            rag_context = self._attachment_store.build_context(user_query)
            if rag_context:
                file_names = ", ".join(self._attachment_store.get_all_file_names())
                session_notice = (
                    f"### Session Attachment Context\n"
                    f"The following file(s) were attached earlier in this session: {file_names}\n"
                    f"Their content is provided below — do NOT use find_files or file_read to locate them.\n"
                )
                _log.debug("RAG CONTEXT (%d chars) for files: %s", len(rag_context), file_names)
                files_context = (files_context + "\n\n" + session_notice + "\n" + rag_context).strip()

        # 1. Memory search
        memory_results = self._memory.search(user_query, top_k=self._cfg.memory_top_k)
        _log.debug("Memory search returned %d result(s)", len(memory_results))
        for i, r in enumerate(memory_results, 1):
            _log.debug("  [mem %d] distance=%.3f  input=%r", i, r.distance, r.user_input[:80])

        # 1b. ReAct mode — bypass Planner/Executor/Synthesizer
        if self._cfg.use_react_loop:
            return self._run_react(
                user_query=user_query,
                memory_results=memory_results,
                files_context=files_context,
                image_data=list(self._session_image_b64),
                on_task_update=on_task_update,
                on_output_chunk=on_output_chunk,
                on_answer_chunk=on_answer_chunk,
            )

        # 2. Plan (classic mode)
        _log.info("Calling planner (model=%s)", self._cfg.planner_model)
        task_list = self._planner.plan(
            user_query,
            memory_context=memory_results,
            conversation_history=self._conversation_history,
            files_context=files_context,
        )
        _log.info("Plan reasoning: %s", task_list.reasoning)
        if task_list.clarifying_questions:
            _log.info("Planner returned %d clarifying question(s)", len(task_list.clarifying_questions))
            for q in task_list.clarifying_questions:
                _log.info("  ? %s", q)
        else:
            _log.info("Plan: %d task(s)", len(task_list.tasks))
            for t in task_list.tasks:
                _log.info("  [%s] tool=%s  deps=%s  desc=%s", t.id, t.tool, t.depends_on, t.description)
                _log.debug("       params=%s", json.dumps(t.params, ensure_ascii=False))

        # 2a. If clarification needed — stream questions, skip execution
        if task_list.clarifying_questions:
            questions_text = "\n".join(
                f"{i+1}. {q}" for i, q in enumerate(task_list.clarifying_questions)
            )
            answer = f"Devam edebilmem için birkaç şeyi netleştirmem gerekiyor:\n\n{questions_text}"
            if on_answer_chunk:
                on_answer_chunk(answer)
            self._conversation_history.append({"role": "user", "content": user_query})
            self._conversation_history.append({"role": "assistant", "content": answer})
            return answer

        if on_plan_ready:
            on_plan_ready(task_list)

        # Wait for user approval before executing
        if require_approval:
            _log.info("Waiting for user approval...")
            self._approval_event.wait()
            if not self._approved:
                _log.info("Plan rejected by user — aborting")
                return ""
            _log.info("Plan approved by user")

        # 3. Execute
        _log.info("Starting execution of %d task(s)", len(task_list.tasks))
        executor = Executor(
            registry=self._registry,
            max_retries=self._cfg.max_task_retries,
            max_tool_output_chars=self._cfg.max_tool_output_chars,
            cancel_event=self._cancel_event,
            on_task_update=on_task_update or (lambda _: None),
            on_output_chunk=on_output_chunk or (lambda _: None),
        )
        executions = executor.run(task_list)

        done = sum(1 for e in executions.values() if e.result and e.result.success)
        failed = sum(1 for e in executions.values() if e.result and not e.result.success)
        skipped = sum(1 for e in executions.values() if e.status.value == "skipped")
        _log.info("Execution complete — done=%d  failed=%d  skipped=%d", done, failed, skipped)

        # 4. Synthesize
        _log.info("Calling synthesizer (model=%s)", self._cfg.synthesizer_model)
        answer = self._synthesizer.synthesize(
            user_query,
            executions,
            plan_reasoning=task_list.reasoning,
            on_chunk=on_answer_chunk,
        )
        _log.info("ANSWER (%d chars): %s", len(answer), answer[:200])
        if len(answer) > 200:
            _log.debug("ANSWER (full): %s", answer)

        # 5. Store to memory
        task_summary = [
            {
                "id": e.task.id,
                "tool": e.task.tool,
                "description": e.task.description,
                "success": e.result.success if e.result else False,
            }
            for e in executions.values()
        ]
        self._memory.store(
            user_input=user_query,
            assistant_output=answer,
            task_summary=task_summary,
            metadata={"model": self._cfg.planner_model},
        )
        _log.debug("Stored interaction to memory")

        # 6. Update conversation history
        self._conversation_history.append({"role": "user", "content": user_query})
        self._conversation_history.append({"role": "assistant", "content": answer})

        return answer

    # ── Attachment preprocessing ───────────────────────────────────────

    _TEXT_EXTENSIONS = {
        ".txt", ".md", ".py", ".cs", ".js", ".ts", ".jsx", ".tsx",
        ".json", ".yaml", ".yml", ".xml", ".html", ".htm", ".css",
        ".csv", ".sql", ".sh", ".bat", ".ps1", ".go", ".rs", ".java",
        ".cpp", ".c", ".h", ".rb", ".php", ".swift", ".kt", ".r",
        ".toml", ".ini", ".cfg", ".conf", ".env",
    }
    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
    _MAX_TEXT_CHARS = 8000
    _MAX_PDF_CHARS = 12000
    _RAG_THRESHOLD_BYTES = 50 * 1024  # 50 KB — files larger than this go to RAG

    def _preprocess_attachments(self, attachments: list[str]) -> str:
        parts = ["### Attached Files"]
        for i, path in enumerate(attachments, 1):
            p = Path(path)
            if not p.exists():
                parts.append(f"[{i}] {path}\n(Dosya bulunamadı)")
                continue
            if p.is_dir():
                parts.append(self._format_dir(i, p))
            elif p.suffix.lower() == ".pdf":
                parts.append(self._handle_pdf(i, p))
            elif p.suffix.lower() in self._TEXT_EXTENSIONS:
                parts.append(self._handle_text_file(i, p))
            elif p.suffix.lower() in self._IMAGE_EXTENSIONS:
                parts.append(self._handle_image(i, p))
            else:
                size = p.stat().st_size
                parts.append(f"[{i}] {p.name} ({size:,} bytes — binary, içerik okunamadı)")
        return "\n\n".join(parts)

    def _handle_text_file(self, idx: int, p: Path) -> str:
        size = p.stat().st_size
        content = self._read_text(p)
        self._index_for_rag(p, content)  # Always index — makes content available for follow-up queries
        if size >= self._RAG_THRESHOLD_BYTES:
            # Large file — inject preview only
            preview = content[:self._MAX_TEXT_CHARS]
            suffix = (
                f"\n... [{len(content) - self._MAX_TEXT_CHARS} karakter kırpıldı — tam içerik RAG ile erişilebilir]"
                if len(content) > self._MAX_TEXT_CHARS else ""
            )
            return f"[{idx}] {p.name} ({p.suffix}, {size / 1024:.1f} KB)\n{'─'*40}\n{preview}{suffix}"
        # Small file — inject full text
        display = content
        if len(display) > self._MAX_TEXT_CHARS:
            display = display[:self._MAX_TEXT_CHARS] + f"\n... [{len(display) - self._MAX_TEXT_CHARS} karakter kırpıldı]"
        return f"[{idx}] {p.name} ({p.suffix}, {size:,} bytes)\n{'─'*40}\n{display}"

    def _handle_pdf(self, idx: int, p: Path) -> str:
        text = self._extract_pdf_text(p)
        self._index_for_rag(p, text)  # Always index — makes content available for follow-up queries
        if len(text.encode()) >= self._RAG_THRESHOLD_BYTES:
            # Large PDF — inject preview only
            preview = text[:self._MAX_PDF_CHARS]
            suffix = (
                f"\n... [{len(text) - self._MAX_PDF_CHARS} karakter kırpıldı — tam içerik RAG ile erişilebilir]"
                if len(text) > self._MAX_PDF_CHARS else ""
            )
            return f"[{idx}] {p.name} (PDF, {p.stat().st_size / 1024:.1f} KB)\n{'─'*40}\n{preview}{suffix}"
        # Small PDF — inject full text
        if len(text) > self._MAX_PDF_CHARS:
            text = text[:self._MAX_PDF_CHARS] + f"\n... [{len(text) - self._MAX_PDF_CHARS} karakter kırpıldı]"
        return f"[{idx}] {p.name} (PDF)\n{'─'*40}\n{text}"

    def _index_for_rag(self, p: Path, content: str) -> int:
        """Index a large file into the session AttachmentStore for supplemental RAG retrieval."""
        n_chunks = self._attachment_store.index(str(p), p.name, content)
        _log.info("RAG indexed %r → %d chunk(s)", p.name, n_chunks)
        return n_chunks

    def _read_text(self, p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"(Okuma hatası: {e})"

    def _handle_image(self, idx: int, p: Path) -> str:
        """Process image: encode as base64 for vision model, and try OCR as fallback."""
        import base64

        # Always encode for vision model (primary path)
        try:
            b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
            self._session_image_b64.append(b64)
        except Exception as e:
            return f"[{idx}] {p.name} (Görsel — okunamadı: {e})"

        # Try OCR as supplementary text extraction (fallback for non-vision models)
        try:
            import pytesseract
            from PIL import Image as _PILImage
            img = _PILImage.open(str(p))
            text = pytesseract.image_to_string(img, lang="tur+eng")
            if text.strip():
                if len(text) > self._MAX_TEXT_CHARS:
                    text = text[:self._MAX_TEXT_CHARS] + f"\n... [{len(text) - self._MAX_TEXT_CHARS} karakter kırpıldı]"
                self._index_for_rag(p, text)
                return f"[{idx}] {p.name} (Görsel — OCR metni aşağıda, görsel de vision modeline gönderildi)\n{'─'*40}\n{text}"
        except ImportError:
            pass  # No OCR — vision model will handle the image directly
        except Exception:
            pass

        size = p.stat().st_size
        return (
            f"[{idx}] {p.name} (Görsel, {size / 1024:.1f} KB — "
            f"görsel doğrudan vision modeline gönderildi, metin çıkarmak için OCR kurulmadı)"
        )

    def _extract_pdf_text(self, p: Path) -> str:
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(str(p)) as pdf:
                for page in pdf.pages:
                    try:
                        pages.append(page.extract_text() or "")
                    except Exception:
                        pages.append("")  # skip unreadable pages silently
            text = "\n\n".join(pages)
            if not text.strip():
                return f"(PDF metin içeriği okunamadı — taranmış görüntü veya özel font olabilir: {p.name})"
            return text
        except ImportError:
            return "(pdfplumber kurulu değil)"
        except Exception as e:
            return f"(PDF okuma hatası: {e})"

    def _format_text_file(self, idx: int, p: Path) -> str:
        content = self._read_text(p)
        if len(content) > self._MAX_TEXT_CHARS:
            content = content[:self._MAX_TEXT_CHARS] + f"\n... [{len(content) - self._MAX_TEXT_CHARS} karakter kırpıldı]"
        return f"[{idx}] {p.name} ({p.suffix}, {p.stat().st_size:,} bytes)\n{'─'*40}\n{content}"

    def _format_dir(self, idx: int, p: Path) -> str:
        lines = [f"[{idx}] {p.name}/ (Klasör)"]
        lines.append("─" * 40)
        try:
            tree = self._build_tree(p, prefix="", max_files=100)
            lines.extend(tree)
        except Exception as e:
            lines.append(f"(Dizin okuma hatası: {e})")
        return "\n".join(lines)

    def _build_tree(self, path: Path, prefix: str, max_files: int) -> list[str]:
        lines = []
        try:
            entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return [prefix + "(erişim reddedildi)"]
        for i, entry in enumerate(entries):
            if len(lines) >= max_files:
                lines.append(prefix + f"... ({len(entries) - i} daha)")
                break
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(prefix + connector + entry.name + ("/" if entry.is_dir() else ""))
            if entry.is_dir() and not entry.name.startswith("."):
                extension = "    " if i == len(entries) - 1 else "│   "
                lines.extend(self._build_tree(entry, prefix + extension, max_files - len(lines)))
        return lines

    # ── ReAct execution path ───────────────────────────────────────────

    def _run_react(
        self,
        user_query: str,
        memory_results,
        files_context: str,
        image_data: list[str] | None = None,
        on_task_update=None,
        on_output_chunk=None,
        on_answer_chunk=None,
    ) -> str:
        _log.info("Running in ReAct mode (model=%s)", self._react_loop._model)
        if image_data:
            _log.info("Passing %d image(s) to vision model", len(image_data))

        answer, executions = self._react_loop.run(
            user_query=user_query,
            conversation_history=self._conversation_history,
            memory_context=memory_results,
            files_context=files_context,
            image_data=image_data,
            on_task_update=on_task_update or (lambda _: None),
            on_output_chunk=on_output_chunk or (lambda _: None),
            on_answer_chunk=on_answer_chunk or (lambda _: None),
        )

        _log.info("ReAct answer (%d chars): %s", len(answer), answer[:200])

        # Store to memory
        task_summary = [
            {
                "id": e.task.id,
                "tool": e.task.tool,
                "description": e.task.description,
                "success": e.result.success if e.result else False,
            }
            for e in executions
        ]
        self._memory.store(
            user_input=user_query,
            assistant_output=answer,
            task_summary=task_summary,
            metadata={"model": self._react_loop._model, "mode": "react"},
        )

        # Update conversation history
        self._conversation_history.append({"role": "user", "content": user_query})
        self._conversation_history.append({"role": "assistant", "content": answer})

        return answer

    def clear_history(self) -> None:
        self._conversation_history.clear()
        self._attachment_store.clear()
        self._session_image_b64.clear()

    def get_available_models(self) -> list[str]:
        try:
            client = ollama.Client(host=self._cfg.ollama_base_url)
            resp = client.list()
            return [m.model for m in resp.models]
        except Exception:
            return []
