from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import ClassVar

from tools.base import BaseTool, ToolParam, ToolResult

# Module-level registry — shared across all tool instances in a session
_running: dict[str, subprocess.Popen] = {}

# Persistent PID store — survives agent restarts
_PIDS_FILE = Path.home() / ".desktop_agent" / "running_pids.json"


def _save_pids() -> None:
    _PIDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {name: proc.pid for name, proc in _running.items() if proc.poll() is None}
    try:
        _PIDS_FILE.write_text(json.dumps(data))
    except Exception:
        pass


def _load_pids() -> dict[str, int]:
    """Return {name: pid} from previous session (if any)."""
    try:
        if _PIDS_FILE.exists():
            return json.loads(_PIDS_FILE.read_text())
    except Exception:
        pass
    return {}


def _kill_pid(pid: int) -> None:
    """Kill a process tree by PID (works even without a Popen handle)."""
    try:
        if os.name == "nt":
            subprocess.run(f"taskkill /F /T /PID {pid}", shell=True, capture_output=True)
        else:
            subprocess.run(["kill", "-9", str(pid)], capture_output=True)
    except Exception:
        pass


def _kill(name: str) -> None:
    proc = _running.pop(name, None)
    if proc is None:
        return
    if proc.poll() is not None:
        return  # already exited
    _kill_pid(proc.pid)
    _save_pids()


def _kill_port(port: int) -> bool:
    """Kill whatever process is listening on `port`. Returns True if killed."""
    try:
        result = subprocess.run(
            f"netstat -ano | findstr :{port}",
            shell=True, capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if parts and parts[-1].isdigit():
                pid = int(parts[-1])
                if pid > 0:
                    _kill_pid(pid)
                    time.sleep(0.5)
                    return True
    except Exception:
        pass
    return False


class ProcessStart(BaseTool):
    name = "process_start"
    description = (
        "Start a long-running process in the background (e.g. dotnet run, npm start). "
        "Port conflicts are resolved automatically. Returns startup log preview."
    )
    params = [
        ToolParam("name", "string", "Unique label for this process (e.g. 'gateway', 'webapp')", required=True),
        ToolParam("command", "string", "Full command to execute", required=True),
        ToolParam("cwd", "string", "Absolute working directory for the process", required=True),
    ]

    def execute(self, name: str, command: str, cwd: str) -> ToolResult:  # type: ignore[override]
        if name in _running and _running[name].poll() is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Process '{name}' is already running (PID {_running[name].pid}). Stop it first.",
            )

        log_dir = Path(cwd) / ".agent_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{name}.log"

        for attempt in range(2):  # retry once on port conflict
            try:
                log_file = open(log_path, "w", encoding="utf-8")  # noqa: WPS515
                flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=cwd,
                    stdout=log_file,
                    stderr=log_file,
                    creationflags=flags,
                )
                _running[name] = proc
                _save_pids()
            except Exception as exc:
                return ToolResult(success=False, output="", error=str(exc))

            # Poll up to 30s for startup success / failure
            _SUCCESS_MARKERS = ("now listening on", "application started")
            _FAILURE_MARKERS = ("address already in use", "failed to start",
                                "unhandled exception", "application failed to start")
            deadline = time.monotonic() + 30
            log_preview = ""
            outcome = "timeout"  # "success" | "port_conflict" | "failure" | "timeout"

            while time.monotonic() < deadline:
                time.sleep(1)
                log_file.flush()
                rc = proc.poll()

                try:
                    with open(log_path, encoding="utf-8", errors="replace") as f:
                        log_preview = f.read(6000)
                except Exception:
                    pass

                lower = log_preview.lower()

                if any(m in lower for m in _SUCCESS_MARKERS):
                    outcome = "success"
                    break
                if "address already in use" in lower:
                    outcome = "port_conflict"
                    break
                if rc is not None and any(m in lower for m in _FAILURE_MARKERS):
                    outcome = "failure"
                    break
                if rc is not None:
                    outcome = "failure"
                    break

            # Timeout reached — if process still alive, treat as success
            if outcome == "timeout" and proc.poll() is None:
                outcome = "success"

            if outcome in ("success", "timeout"):
                # Still running (or cleanly exited with 0)
                status = (
                    f"Started '{name}' (PID {proc.pid}) — running. Log: {log_path}"
                )
                if log_preview:
                    status += f"\n--- Log preview (last read) ---\n{log_preview[:800]}"
                return ToolResult(
                    success=True,
                    output=status,
                    data={"name": name, "pid": proc.pid, "log": str(log_path)},
                )

            # Process failed — clean up
            _running.pop(name, None)
            _save_pids()

            if outcome == "port_conflict" and attempt == 0:
                import re as _re
                ports = _re.findall(r":(\d{4,5})", log_preview)
                killed_any = any(_kill_port(int(p)) for p in ports)
                if killed_any:
                    time.sleep(1)
                    continue  # retry

            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Process '{name}' failed to start (outcome={outcome}, attempt={attempt+1}).\n"
                    f"Log:\n{log_preview[:1500] or '(empty)'}"
                ),
            )

        return ToolResult(success=False, output="", error=f"Failed to start '{name}' after 2 attempts.")


class ProcessStop(BaseTool):
    name = "process_stop"
    description = (
        "Stop a background process started with process_start. "
        "Use name='ALL' to stop every managed process — including ones from previous sessions."
    )
    params = [
        ToolParam(
            "name",
            "string",
            "Label of the process to stop, or 'ALL' to stop everything",
            required=True,
        ),
    ]

    def execute(self, name: str) -> ToolResult:  # type: ignore[override]
        if name == "ALL":
            # Also kill processes from previous sessions stored on disk
            prev = _load_pids()
            for n, pid in prev.items():
                _kill_pid(pid)

            names = list(_running.keys())
            for n in names:
                _kill(n)

            # Clear persistent store
            try:
                _PIDS_FILE.write_text(json.dumps({}))
            except Exception:
                pass

            total = len(set(list(prev.keys()) + names))
            if total == 0:
                return ToolResult(success=True, output="No managed processes were running.")
            all_names = sorted(set(list(prev.keys()) + names))
            return ToolResult(success=True, output=f"Stopped {total} process(es): {', '.join(all_names)}")

        if name not in _running:
            return ToolResult(
                success=False,
                output="",
                error=f"No managed process named '{name}'. Running: {list(_running.keys())}",
            )
        _kill(name)
        return ToolResult(success=True, output=f"Stopped '{name}'.")


class ProcessList(BaseTool):
    name = "process_list"
    description = "List all background processes (current session + previous sessions from disk)."
    params: ClassVar[list] = []

    def execute(self) -> ToolResult:  # type: ignore[override]
        lines = []
        for n, proc in _running.items():
            rc = proc.poll()
            status = "running" if rc is None else f"exited (code {rc})"
            lines.append(f"  {n}: PID {proc.pid} — {status} [this session]")

        prev = _load_pids()
        for n, pid in prev.items():
            if n not in _running:
                # Check if actually still alive
                alive = subprocess.run(
                    f"tasklist /FI \"PID eq {pid}\" /NH",
                    shell=True, capture_output=True, text=True,
                ).stdout.strip()
                status = "running" if str(pid) in alive else "exited"
                lines.append(f"  {n}: PID {pid} — {status} [previous session]")

        if not lines:
            return ToolResult(success=True, output="No managed processes.")
        return ToolResult(success=True, output="\n".join(lines))
