"""
ReactTracer — session-per-file human-readable trace log for the ReAct loop.

Each run writes a separate file:
    ~/.desktop_agent/traces/YYYY-MM-DD_HH-MM-SS_react.log

File format example:
════════════════════════════════════════════════════════════════════════
SESSION   2026-03-07 14:23:01
MODEL     qwen2.5:7b
QUERY     python projemi çalıştır ve hataları düzelt
════════════════════════════════════════════════════════════════════════

[iter 1] LLM → tool call(s)

  ┌─ [r1] shell_run
  │  command = 'python main.py'
  └─ STATUS: OK  (1.23s)
     OUTPUT (247 chars):
     Traceback (most recent call last):
       File "main.py", line 3, in <module>
     ...

────────────────────────────────────────────────────────────────────────

[iter 2] LLM → tool call(s)

  ┌─ [r2] file_read
  │  path    = 'main.py'
  │  start_line = 1
  │  end_line   = 50
  └─ STATUS: OK  (0.45s)
     OUTPUT (1024 chars):
     import os
     ...

────────────────────────────────────────────────────────────────────────

[iter 3] LLM → final answer

ANSWER ─────────────────────────────────────────────────────────────────
3. satırda import hatası var. Şu değişikliği yap: ...

SUMMARY ────────────────────────────────────────────────────────────────
  Tool calls : 2
  Iterations : 3
  Elapsed    : 8.2s
  Trace file : C:/Users/..desktop_agent/traces/2026-03-07_14-23-01_react.log
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.executor import TaskExecution

_W = 72  # line width


class ReactTracer:
    """Writes a structured, human-readable trace file for one ReAct session."""

    def __init__(self, traces_dir: Path, model: str):
        traces_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._path = traces_dir / f"{ts}_react.log"
        self._model = model
        self._start = time.monotonic()
        self._tool_calls = 0
        self._iteration = 0
        self._f = open(self._path, "w", encoding="utf-8")

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    def write_header(self, user_query: str) -> None:
        self._line("═" * _W)
        self._line(f"SESSION   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._line(f"MODEL     {self._model}")
        self._line(f"QUERY     {user_query}")
        self._line("═" * _W)

    def write_iteration_start(self, iteration: int) -> None:
        self._iteration = iteration
        self._line("")
        self._line(f"[iter {iteration}] LLM → deciding next action...")

    def write_tool_call_start(self, task_id: str, tool_name: str, args: dict) -> None:
        self._tool_calls += 1
        self._line("")
        self._line(f"  ┌─ [{task_id}] {tool_name}")
        for k, v in args.items():
            val = repr(v)
            if len(val) > 100:
                val = val[:100] + "..."
            self._line(f"  │  {k:<12} = {val}")

    def write_tool_result(self, task_id: str, execution: "TaskExecution") -> None:
        result = execution.result
        elapsed = execution.elapsed or 0.0
        status = "OK" if (result and result.success) else "FAIL"
        self._line(f"  └─ STATUS: {status}  ({elapsed:.2f}s)")

        if result:
            if result.output:
                output = result.output
                preview = output[:600]
                # indent every line
                indented = "\n     ".join(preview.splitlines())
                self._line(f"     OUTPUT ({len(output)} chars):")
                self._line(f"     {indented}")
                if len(output) > 600:
                    self._line(f"     ... [{len(output) - 600} chars truncated]")
            if result.error:
                self._line(f"     ERROR: {result.error}")

        if result and result.data:
            self._line(f"     DATA:  {repr(result.data)[:200]}")

    def write_llm_reasoning(self, text: str) -> None:
        """If the LLM outputs text alongside tool calls (chain-of-thought)."""
        if not text or not text.strip():
            return
        self._line("")
        self._line("  [LLM reasoning]")
        for line in text.strip().splitlines():
            self._line(f"    {line}")

    def write_iteration_separator(self) -> None:
        self._line("")
        self._line("─" * _W)

    def write_answer(self, answer: str, iteration_count: int) -> None:
        elapsed = time.monotonic() - self._start
        self._line("")
        self._line(f"[iter {iteration_count}] LLM → final answer")
        self._line("")
        self._line(f"ANSWER {'─' * (_W - 7)}")
        for line in answer.splitlines():
            self._line(line)
        self._line("")
        self._line(f"SUMMARY {'─' * (_W - 8)}")
        self._line(f"  Tool calls : {self._tool_calls}")
        self._line(f"  Iterations : {iteration_count}")
        self._line(f"  Elapsed    : {elapsed:.1f}s")
        self._line(f"  Trace file : {self._path}")
        self._line("═" * _W)
        self._line("")

    def write_cancelled(self) -> None:
        elapsed = time.monotonic() - self._start
        self._line("")
        self._line("*** CANCELLED BY USER ***")
        self._line(f"  Tool calls so far : {self._tool_calls}")
        self._line(f"  Elapsed           : {elapsed:.1f}s")
        self._line("═" * _W)

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    # ── Internal ───────────────────────────────────────────────────────

    def _line(self, text: str) -> None:
        self._f.write(text + "\n")
        self._f.flush()
