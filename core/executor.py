from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TYPE_CHECKING

from logger import get_logger
from tools.base import ToolResult

if TYPE_CHECKING:
    from core.planner import Task, TaskList
    from tools.registry import ToolRegistry

_log = get_logger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskExecution:
    task: "Task"
    status: TaskStatus = TaskStatus.PENDING
    result: ToolResult | None = None
    started_at: float | None = None
    finished_at: float | None = None
    attempt: int = 0

    @property
    def elapsed(self) -> float | None:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None


_TEMPLATE_RE = re.compile(r"\{\{(\w+)\.(\w+)(?:\.(\w+))?\}\}")


def _resolve_templates(value, prior: dict[str, TaskExecution]):
    if isinstance(value, str):
        def replacer(m):
            task_id, attr, subkey = m.group(1), m.group(2), m.group(3)
            exec_ = prior.get(task_id)
            if exec_ is None or exec_.result is None:
                return m.group(0)
            if attr == "output":
                return exec_.result.output
            if attr == "data" and subkey and isinstance(exec_.result.data, dict):
                return str(exec_.result.data.get(subkey, m.group(0)))
            return m.group(0)
        return _TEMPLATE_RE.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _resolve_templates(v, prior) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_templates(v, prior) for v in value]
    return value


_TRUNCATION_NOTICE = "\n\n... [Çıktı çok uzun, kırpıldı. Toplam: {total} karakter, gösterilen: {limit}]"


def _truncate(result: ToolResult, max_chars: int) -> ToolResult:
    if not result.output or len(result.output) <= max_chars:
        return result
    notice = _TRUNCATION_NOTICE.format(total=len(result.output), limit=max_chars)
    return ToolResult(
        success=result.success,
        output=result.output[:max_chars] + notice,
        data=result.data,
        error=result.error,
    )


class Executor:
    def __init__(
        self,
        registry: "ToolRegistry",
        max_retries: int = 2,
        max_tool_output_chars: int = 8000,
        cancel_event: threading.Event | None = None,
        on_task_update: Callable[[TaskExecution], None] | None = None,
        on_output_chunk: Callable[[str], None] | None = None,
    ):
        self._registry = registry
        self._max_retries = max_retries
        self._max_output_chars = max_tool_output_chars
        self._cancel = cancel_event or threading.Event()
        self._on_task_update = on_task_update or (lambda _: None)
        self._on_output_chunk = on_output_chunk or (lambda _: None)

        # Wire cancel event to shell tool if present
        try:
            shell = registry.get("shell_run")
            shell.set_cancel_event(self._cancel)
        except KeyError:
            pass

    def run(self, task_list: "TaskList") -> dict[str, TaskExecution]:
        executions: dict[str, TaskExecution] = {
            t.id: TaskExecution(task=t) for t in task_list.tasks
        }
        waves = self._resolve_waves(task_list.tasks)

        for wave in waves:
            if self._cancel.is_set():
                for task in wave:
                    executions[task.id].status = TaskStatus.SKIPPED
                    self._on_task_update(executions[task.id])
                continue

            # Check if any dependency failed → skip
            runnable, skipped = [], []
            for task in wave:
                if any(
                    executions[dep].status in (TaskStatus.FAILED, TaskStatus.SKIPPED)
                    for dep in task.depends_on
                    if dep in executions
                ):
                    skipped.append(task)
                else:
                    runnable.append(task)

            for task in skipped:
                executions[task.id].status = TaskStatus.SKIPPED
                self._on_task_update(executions[task.id])

            if len(runnable) == 1:
                self._execute_single(runnable[0], executions)
            else:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {
                        pool.submit(self._execute_single, t, executions): t
                        for t in runnable
                    }
                    for _ in as_completed(futures):
                        pass

        return executions

    def _resolve_waves(self, tasks: list["Task"]) -> list[list["Task"]]:
        task_map = {t.id: t for t in tasks}
        remaining = list(tasks)
        waves = []
        resolved: set[str] = set()

        while remaining:
            wave = [t for t in remaining if all(d in resolved for d in t.depends_on)]
            if not wave:
                # Circular or unresolvable — add all remaining as one wave
                wave = remaining
            waves.append(wave)
            resolved.update(t.id for t in wave)
            remaining = [t for t in remaining if t.id not in resolved]

        return waves

    def _execute_single(
        self, task: "Task", executions: dict[str, TaskExecution]
    ) -> None:
        exec_ = executions[task.id]

        for attempt in range(self._max_retries + 1):
            if self._cancel.is_set():
                _log.info("[%s] Cancelled before attempt %d", task.id, attempt + 1)
                exec_.status = TaskStatus.SKIPPED
                self._on_task_update(exec_)
                return

            exec_.status = TaskStatus.RUNNING
            exec_.started_at = time.monotonic()
            exec_.attempt = attempt + 1
            self._on_task_update(exec_)

            if task.tool == "noop":
                msg = task.params.get("message", "(no action)")
                exec_.result = ToolResult(success=True, output=msg)
                exec_.status = TaskStatus.DONE
                exec_.finished_at = time.monotonic()
                self._on_output_chunk(f"[{task.id}] noop: {msg}\n")
                self._on_task_update(exec_)
                _log.info("[%s] noop — %s", task.id, msg[:120])
                return

            try:
                tool = self._registry.get(task.tool)
            except KeyError:
                exec_.result = ToolResult(
                    success=False, output="",
                    error=f"Unknown tool: {task.tool!r}"
                )
                exec_.status = TaskStatus.FAILED
                exec_.finished_at = time.monotonic()
                self._on_task_update(exec_)
                _log.error("[%s] Unknown tool: %r", task.id, task.tool)
                return

            resolved_params = _resolve_templates(task.params, executions)
            _log.info("[%s] RUNNING tool=%s  attempt=%d  desc=%s",
                      task.id, task.tool, attempt + 1, task.description)
            _log.debug("[%s] params=%s", task.id, resolved_params)
            self._on_output_chunk(f"[{task.id}] {task.tool}: {task.description}\n")

            result = _truncate(tool.execute(**resolved_params), self._max_output_chars)
            exec_.result = result
            exec_.finished_at = time.monotonic()
            elapsed = exec_.elapsed or 0.0

            if result.success:
                exec_.status = TaskStatus.DONE
                self._on_output_chunk(result.output + "\n")
                self._on_task_update(exec_)
                _log.info("[%s] SUCCESS (%.2fs)  output_len=%d",
                          task.id, elapsed, len(result.output or ""))
                _log.debug("[%s] OUTPUT:\n%s", task.id, result.output)
                return
            else:
                self._on_output_chunk(f"ERROR: {result.error}\n")
                _log.warning("[%s] FAILED (%.2fs)  error=%s", task.id, elapsed, result.error)
                if attempt < self._max_retries:
                    _log.info("[%s] Retrying in 1s... (attempt %d/%d)",
                              task.id, attempt + 1, self._max_retries + 1)
                    time.sleep(1)
                else:
                    exec_.status = TaskStatus.FAILED
                    self._on_task_update(exec_)
                    _log.error("[%s] All retries exhausted — FAILED", task.id)
