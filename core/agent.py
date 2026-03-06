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
        self._conversation_history: list[dict] = []
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
        files_context = ""
        if attachments:
            files_context = self._preprocess_attachments(attachments)
            _log.debug("FILES CONTEXT (%d chars):\n%s", len(files_context), files_context[:500])

        # If session RAG store has indexed large files, retrieve relevant chunks
        if self._attachment_store.has_data():
            rag_context = self._attachment_store.build_context(user_query)
            if rag_context:
                _log.debug("RAG CONTEXT (%d chars)", len(rag_context))
                files_context = (files_context + "\n\n" + rag_context).strip()

        # 1. Memory search
        memory_results = self._memory.search(user_query, top_k=self._cfg.memory_top_k)
        _log.debug("Memory search returned %d result(s)", len(memory_results))
        for i, r in enumerate(memory_results, 1):
            _log.debug("  [mem %d] distance=%.3f  input=%r", i, r.distance, r.user_input[:80])

        # 2. Plan
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
            else:
                size = p.stat().st_size
                parts.append(f"[{i}] {p.name} ({size:,} bytes — binary, içerik okunamadı)")
        return "\n\n".join(parts)

    def _handle_text_file(self, idx: int, p: Path) -> str:
        size = p.stat().st_size
        if size >= self._RAG_THRESHOLD_BYTES:
            return self._index_for_rag(idx, p, self._read_text(p))
        return self._format_text_file(idx, p)

    def _handle_pdf(self, idx: int, p: Path) -> str:
        text = self._extract_pdf_text(p)
        if len(text.encode()) >= self._RAG_THRESHOLD_BYTES:
            return self._index_for_rag(idx, p, text)
        # Small PDF — inject directly
        if len(text) > self._MAX_PDF_CHARS:
            text = text[:self._MAX_PDF_CHARS] + f"\n... [{len(text) - self._MAX_PDF_CHARS} karakter kırpıldı]"
        return f"[{idx}] {p.name} (PDF)\n{'─'*40}\n{text}"

    def _index_for_rag(self, idx: int, p: Path, content: str) -> str:
        """Index a large file into the session AttachmentStore; return a placeholder."""
        n_chunks = self._attachment_store.index(str(p), p.name, content)
        _log.info("RAG indexed %r → %d chunk(s)", p.name, n_chunks)
        size_kb = p.stat().st_size / 1024
        return (
            f"[{idx}] {p.name} ({size_kb:.1f} KB — büyük dosya, RAG ile indekslendi, "
            f"{n_chunks} parça. İlgili bölümler otomatik olarak alınacak.)"
        )

    def _read_text(self, p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"(Okuma hatası: {e})"

    def _extract_pdf_text(self, p: Path) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(str(p)) as pdf:
                return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
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

    def clear_history(self) -> None:
        self._conversation_history.clear()
        self._attachment_store.clear()

    def get_available_models(self) -> list[str]:
        try:
            client = ollama.Client(host=self._cfg.ollama_base_url)
            resp = client.list()
            return [m.model for m in resp.models]
        except Exception:
            return []
