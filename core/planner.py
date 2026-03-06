from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import ollama
from pydantic import BaseModel, model_validator

from logger import get_logger

if TYPE_CHECKING:
    from memory.store import MemoryResult
    from tools.registry import ToolRegistry

_log = get_logger(__name__)

# --- Configuration & Regex ---
_ACTION_RE = re.compile(
    r"\b(oluştur|yarat|ekle|sil|taşı|düzenle|güncelle|yaz|değiştir|kur|çalıştır|indir|"
    r"create|make|add|delete|remove|move|edit|write|update|generate|build|install|run|"
    r"setup|init|deploy|configure|fix|refactor|implement|rename)\b",
    re.IGNORECASE,
)

_EXISTENCE_CHECK_RE = re.compile(
    r"^\s*("
    r"dir\b"           
    r"|if\s+exist\b"   
    r"|test\s+-[dfe]\b"
    r")\s*",
    re.IGNORECASE,
)

_SERVER_CMD_RE = re.compile(
    r"^\s*(dotnet\s+run|npm\s+(run\s+)?start|yarn\s+start|"
    r"python\s+.*manage\.py\s+runserver|flask\s+run|"
    r"uvicorn|gunicorn|node\s+\S+\.js)\b",
    re.IGNORECASE,
)

_DESTRUCTIVE_CMD_RE = re.compile(
    r"^\s*(rmdir\s+/s|rd\s+/s|del\s+/s|format\s+|diskpart\s+|takeown\s+|icacls\s+)",
    re.IGNORECASE,
)

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([^.}]+)\.[^}]+\s*\}\}")
_READ_FILE_INTENT_RE = re.compile(
    r"\b("
    r"read|open|inspect|analy[sz]e|review|explain|show|summari[sz]e|"
    r"oku|ac|incele|incelemesini|analiz|yorumla|goster|ozetle"
    r")\b",
    re.IGNORECASE,
)


# --- Exceptions ---
class PlannerParseError(Exception):
    """Raised when JSON parsing or structural schema validation completely exhausts internal auto-healing retries."""
    pass


# --- Pydantic Data Topology Validation ---
class Task(BaseModel):
    id: str
    tool: str
    params: dict[str, Any]
    depends_on: list[str] = []
    description: str

    @model_validator(mode="after")
    def validate_tool_params(self) -> "Task":
        tool = self.tool
        p = self.params
        
        if tool == "shell_run" and "command" not in p:
            raise ValueError(f"Task '{self.id}': Tool 'shell_run' strictly dictates 'command' payload mapping.")
        elif tool == "file_write" and ("path" not in p or "content" not in p):
            raise ValueError(f"Task '{self.id}': 'file_write' requires explicit 'path' and nested 'content'.")
        elif tool == "file_read" and "path" not in p:
            raise ValueError(f"Task '{self.id}': 'file_read' requires parameter 'path'.")
        elif tool == "file_edit" and not all(k in p for k in ("path", "old_string", "new_string")):
            raise ValueError(f"Task '{self.id}': Logically missing 'path', 'old_string', or 'new_string' mapping on edit.")
        return self


class TaskList(BaseModel):
    reasoning: str
    clarifying_questions: list[str] = []
    tasks: list[Task]

    @model_validator(mode="after")
    def validate_graph(self) -> "TaskList":
        if not self.tasks:
            return self

        seen_ids = set()
        adj: dict[str, list[str]] = {}

        # 1. Deduplication Guarding
        for t in self.tasks:
            if t.id in seen_ids:
                raise ValueError(f"Duplicate DAG reference tracking anomaly - Task ID duplicated: {t.id}")
            seen_ids.add(t.id)
            adj[t.id] = t.depends_on

        # 2. Hard existence of mapping branches
        for t in self.tasks:
            for dep in t.depends_on:
                if dep not in seen_ids:
                    raise ValueError(f"Edge structure corrupted: '{t.id}' mapped identically down non-existent dependency '{dep}'.")

        # 3. Detection of Circular Loop Paths via DFS algorithm
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> None:
            if node in rec_stack:
                raise ValueError(f"Fatal cyclic architectural breakdown identified dynamically circling through node '{node}'")
            if node in visited:
                return
            visited.add(node)
            rec_stack.add(node)
            for structural_neighbor in adj.get(node, []):
                dfs(structural_neighbor)
            rec_stack.remove(node)

        for t in self.tasks:
            if t.id not in visited:
                dfs(t.id)

        # 4. Enforce template variable dependency compliance structurally mapping down {{target.output}} bindings.
        def _check_placeholders(obj: Any, task_id: str) -> None:
            if isinstance(obj, dict):
                for v in obj.values():
                    _check_placeholders(v, task_id)
            elif isinstance(obj, list):
                for v in obj:
                    _check_placeholders(v, task_id)
            elif isinstance(obj, str):
                matches = _PLACEHOLDER_RE.findall(obj)
                for structural_ref_id in matches:
                    if structural_ref_id not in seen_ids:
                        raise ValueError(f"Task target '{task_id}' maps missing structural placeholder to non-existent architecture '{structural_ref_id}'.")
                    if structural_ref_id not in adj.get(task_id, []) and structural_ref_id != task_id:
                        raise ValueError(f"Explicit reference placeholder bindings must be codified securely within 'depends_on' -> Task '{task_id}' uses '{structural_ref_id}'")

        for t in self.tasks:
            _check_placeholders(t.params, t.id)

        return self


# --- Direct System Templates ---
_SYSTEM_TEMPLATE = """\
You are a task planning agent. Your job is to decompose the user's request into an ordered list of tool calls.
You MUST output ONLY valid JSON conforming to the provided schema — no commentary, no markdown fences.

## CRITICAL: web_search is FORBIDDEN for programming tasks
NEVER use web_search or web_fetch to answer programming questions or create projects.
You are a coding expert. You do NOT need the internet to create projects, write API components, or explain development patterns.
Use `shell_run` and `file_write` respectively to act.

web_search / web_fetch is ONLY allowed for:
- Fetching a specific URL the user explicitly provides
- Real-time data: live prices, current news, weather

## Attached Files
If the user message contains a `### Attached Files` section, the file contents are already embedded there.
Do NOT use `file_read` to re-read those same files. Use the provided content directly to answer or act.

## Available Tools
{tool_manifest}

## Planning Rules
1. Use the fewest tasks that correctly satisfy the request.
2. Set `depends_on` to express data flow: if t2 needs t1's output, set t2.depends_on = ["t1"].
3. To pass a prior task's output as a parameter, use: {{{{t1.output}}}} or {{{{t1.data.key}}}}.
4. For file edits: always read first (file_read), then edit (file_edit), with the edit depending on the read. 
   The `old_string` parameter MUST be copied character-for-character from the file. Include enough surrounding context.
5. Never use tool names that are not in the Available Tools list.
6. If no tool is needed, use tool "noop" with params: {{"message": "explanation"}}.
7. **CRITICAL: NEVER GUESS OR ASSUME FILE/FOLDER NAMES (e.g. "MyProject")**. 
   If the exact name isn't provided, FIRST execute `dir_tree` (or `find_files`) to observe directories. You CAN AND MUST chain a subsequent `file_read` in the EXACT SAME PLAN if the ultimate goal is to read a file you are searching for. A search-only plan is invalid when the user asked to inspect/analyze/open/read the file contents.
   Example:
   `[t1] find_files(pattern="*GetCalculatedProduct*")`
   `[t2] file_read(path="{{t1.data.first_path}}", start_line=1, end_line=200)` -> `depends_on: ["t1"]`
   `find_files` uses the parameter `pattern` and exposes the first hit as `data.first_path`.
8. **FORBIDDEN tasks**:
   - `dir`, `if exist ...`, or checking existence statically before acting. Just force creation/editing.
   - Long-running server/daemon commands (`dotnet run`, `npm start`, etc.).
   - Destructive OS wipes (`rmdir /s`, `del /s`, `format`).
9. The shell runs in the workspace directory. Use relative paths.
10. ALWAYS leave the `path` parameter inside directory crawlers totally empty to hit the active workspace root unless asked distinctly for a nested subfolder block.

## CRITICAL: Chunked Reads Required for Large Files
For any potentially large file, log, SQL script, or multi-result search output, you MUST NOT read the entire content in a single tool call.

Instead:
- first read a small chunk
- inspect relevance
- read additional chunks only when required
- use depends_on between chunk tasks

For `file_read`, always prefer chunked parameters when available.
Examples:
- `file_read(path="x.sql", start_line=1, end_line=200)`
- `file_read(path="x.sql", start_line=201, end_line=400)`

If the user asks for full analysis of a large file, decompose it into multiple chunk-reading tasks followed by a final summarize/analyze task.

Never perform `file_edit` using incomplete or truncated content.

## Shell Environment
The shell is **Windows Command Prompt (cmd.exe)**. 
- Use `mkdir FolderName` (NOT `mkdir -p`)
- Use `dir` instead of `ls`, `type` instead of `cat`
- To create files structurally, ALWAYS prefer `file_write` native system execution over blind shell terminals.
- Chain commands sequentially mapping bounds iteratively via `&&` architecture.
"""

_USER_TEMPLATE = """\
{memory_context}

{conv_history}

{files_context}

User request: {user_query}
"""

_MEMORY_HEADER = """\
### Relevant Past Interactions
> WARNING: This shows PAST ACTIONS only — NOT the current state of the filesystem.
> Assume files created previously have been entirely wiped. ALWAYS act dynamically to recreate logic parameters securely against historical dependencies. 
"""

# --- Static Execution Coordinator ---
class Planner:
    def __init__(self, base_url: str, model: str, registry: "ToolRegistry"):
        self._client = ollama.Client(host=base_url)
        self._model = model
        self._registry = registry

    def plan(
        self,
        user_query: str,
        memory_context: list["MemoryResult"] | None = None,
        conversation_history: list[dict] | None = None,
        files_context: str = "",
    ) -> TaskList:
        result = self._execute_self_healing_llm_inference(user_query, memory_context, conversation_history, files_context)

        # Memory contamination logic tracking "Already Done" behavioral anomalies defensively targeting NOOP cycles.
        if (
            result.tasks
            and all(t.tool == "noop" for t in result.tasks)
            and _ACTION_RE.search(user_query)
        ):
            _log.warning("Systemic NOOP failure triggered - Memory context driving systemic contamination blocking. Retrying without historical map.")
            result = self._execute_self_healing_llm_inference(user_query, None, conversation_history, files_context)

        return result

    def _execute_self_healing_llm_inference(
        self,
        user_query: str,
        memory_context: list["MemoryResult"] | None,
        conversation_history: list[dict] | None,
        files_context: str,
        max_retries: int = 3,
    ) -> TaskList:
        """Centralized Execution wrapper feeding schema exception bounds tracebacks dynamically to self-correct."""
        
        system_prompt = _SYSTEM_TEMPLATE.format(tool_manifest=self._registry.to_text_manifest())
        
        user_message = _USER_TEMPLATE.format(
            memory_context=self._format_memory(memory_context or []),
            conv_history=self._format_history(conversation_history or []),
            files_context=files_context,
            user_query=user_query,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        for internal_attempt in range(1, max_retries + 1):
            try:
                response = self._client.chat(
                    model=self._model,
                    messages=messages,
                    format=TaskList.model_json_schema(),
                    options={"temperature": 0.05 if internal_attempt == 1 else 0.3},
                )
                raw = response.message.content
                _log.debug("PLANNER LLM INTERNAL RESOLVE (Attempt %d):\n%s", internal_attempt, raw)
                
                parsed_list = self._parse_and_validate(raw)
                _log.info("Planner parsing structural deployment accepted (Attempt %d). Execution Map Built: %d", internal_attempt, len(parsed_list.tasks))
                filtered = self._filter_forbidden_tasks(parsed_list)
                return self._ensure_file_reads(user_query, filtered)

            except Exception as e:
                _log.warning("Planner structural trace failed integrity validation dynamically parameters (Attempt %d): %s", internal_attempt, e)
                
                if internal_attempt == max_retries:
                    _log.error("Structural Retry Mechanism completely exhausted attempting to evaluate syntax mapping.")
                    raise PlannerParseError(f"Complete system architectural collapse parsing executing logic boundary bounds: {e}") from e
                
                # Feedback loop insertion mapping standard assistant blocks immediately preceding user syntax bounds logic maps. 
                messages.append({"role": "assistant", "content": raw if 'raw' in locals() else "{}"})
                messages.append({
                    "role": "user", 
                    "content": f"The architectural JSON bounds produced a fatal schema pipeline validation violation: {e}\nRectify JSON hierarchy."
                })
        
        return TaskList(reasoning="Resolution safe pipeline execution halt trigger.", tasks=[])

    def _parse_and_validate(self, raw: str) -> TaskList:
        """Decouples markdown boundaries, cleans trailing schema syntactics, triggers pydantic topology DAG bounds."""
        cleaned = raw.strip()
        
        # Guard mechanism dynamically handling LLM native codeblock hallucinations outputting standard markdown boundaries mapping strings. 
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1:
            cleaned = cleaned[first_brace : last_brace + 1]

        # Automatic JSON sanitation logic isolating nested list trailing structure parameters 
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as internal_error:
            raise ValueError(f"System logical parse degradation JSON structure failure: {internal_error}") from internal_error

        # Triggers standard internal Pydantic @model_validator directives directly evaluating entire architectural bounds immediately upon initialization trigger map sequence.
        return TaskList.model_validate(data)

    def _filter_forbidden_tasks(self, task_list: TaskList) -> TaskList:
        """Strict constraint filter structurally sweeping destructive parameters natively mapping cascade teardowns blocking execution faults propagating."""
        kept: list[Task] = []
        removed_ids: set[str] = set()

        for task in task_list.tasks:
            
            # Cascade Architecture Verification evaluating completely against missing parameter payload integrations downstream targeting historical faults.
            is_orphaned_completely = False
            for mapped_dep in task.depends_on:
                if mapped_dep in removed_ids:
                    if self._uses_placeholder_recursively(task.params, mapped_dep):
                        is_orphaned_completely = True
                        break

            if is_orphaned_completely:
                _log.warning("CASCADE TEARDOWN STRUCTURAL BLOCK - Erasing dependent task execution mapping parameter target [%s]", task.id)
                removed_ids.add(task.id)
                continue

            reason = self._forbidden_reason(task)
            
            if reason:
                _log.warning("STRUCTURAL FILTER REJECTION OMIT - Targeted removal execution deployment constraint map: [%s] -> %s", task.id, reason)
                removed_ids.add(task.id)
            else:
                clean_deps = [d for d in task.depends_on if d not in removed_ids]
                if clean_deps != task.depends_on:
                    task = task.model_copy(update={"depends_on": clean_deps})
                kept.append(task)

        if len(kept) != len(task_list.tasks):
            _log.info("FILTER PIPELINE TOTAL EVAL: Removed %d, Processed %d architectural bound instances securely resolving logic limits target bounds.", len(task_list.tasks) - len(kept), len(kept))

        return task_list.model_copy(update={"tasks": kept})

    def _uses_placeholder_recursively(self, obj: Any, target_id: str) -> bool:
        """Dynamically iterates parameter structure dictionary types tracing explicit dependencies via regular expressions."""
        if isinstance(obj, dict):
            return any(self._uses_placeholder_recursively(v, target_id) for v in obj.values())
        if isinstance(obj, list):
            return any(self._uses_placeholder_recursively(v, target_id) for v in obj)
        if isinstance(obj, str):
            regex_map = re.compile(r"\{\{\s*" + re.escape(target_id) + r"\.[^}]+\s*\}\}")
            return bool(regex_map.search(obj))
        return False

    def _forbidden_reason(self, task: Task) -> str | None:
        """Determines logic matching parameter target architectures matching illegal state bound strings dynamically mapped parameters constraint sets"""
        if task.tool != "shell_run":
            return None
            
        cmd = str(task.params.get("command", "")).strip()
        
        if _EXISTENCE_CHECK_RE.match(cmd):
            return f"Illegal existence structural check detected strictly preventing sequence execution: {cmd!r}"
        if _SERVER_CMD_RE.match(cmd):
            return f"Blocked infinite execution execution server logic fault matching sequence logic: {cmd!r}"
        if _DESTRUCTIVE_CMD_RE.match(cmd):
            return f"CRITICAL: Caught destructive structural target executing matching system removal fault constraint logic sequences targeting OS variables: {cmd!r}"
            
        desc_lower = task.description.lower()
        if any(p in desc_lower for p in (
            "check if", "verify if", "determine if",
            "check the existence", "see if", "test if"
        )):
            return f"Identified constraint existence verification descriptions target sequence explicitly: {task.description!r}"
            
        return None

    def _ensure_file_reads(self, user_query: str, task_list: TaskList) -> TaskList:
        if not _READ_FILE_INTENT_RE.search(user_query):
            return task_list

        find_tasks = [t for t in task_list.tasks if t.tool == "find_files"]
        if not find_tasks:
            return task_list

        covered_find_ids = {
            dep
            for task in task_list.tasks
            if task.tool == "file_read"
            for dep in task.depends_on
        }
        missing_reads = [task for task in find_tasks if task.id not in covered_find_ids]
        if not missing_reads:
            return task_list

        next_index = self._next_task_index(task_list.tasks)
        augmented = list(task_list.tasks)
        for find_task in missing_reads:
            augmented.append(
                Task(
                    id=f"t{next_index}",
                    tool="file_read",
                    params={
                        "path": f"{{{{{find_task.id}.data.first_path}}}}",
                        "start_line": 1,
                        "end_line": 200,
                    },
                    depends_on=[find_task.id],
                    description="Read the beginning of the first matched file for content analysis.",
                )
            )
            next_index += 1

        return task_list.model_copy(update={"tasks": augmented})

    def _next_task_index(self, tasks: list[Task]) -> int:
        max_index = 0
        for task in tasks:
            match = re.fullmatch(r"t(\d+)", task.id)
            if match:
                max_index = max(max_index, int(match.group(1)))
        return max_index + 1

    def _format_memory(self, results: list["MemoryResult"]) -> str:
        if not results:
            return ""
        lines = [_MEMORY_HEADER.strip()]
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] (dist: {r.distance:.3f}) Sequence Trigger Input Context mapping variable strings targets: {r.user_input[:100]!r}\n"
                f"   Summary Execute: {r.task_summary[:200]}"
            )
        return "\n".join(lines)

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return ""
        lines = ["### Context Conversation Bound Variable History String Execute Sequence Maps"]
        for msg in history[-8:]:
            role = msg.get("role", "?").capitalize()
            content = str(msg.get("content", ""))[:8000]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
