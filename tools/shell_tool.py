from __future__ import annotations

import locale
import os
import re
import subprocess
import sys
import threading
from typing import IO

from tools.base import BaseTool, ToolParam, ToolResult

_BLOCKED_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"format\s+[a-zA-Z]:", re.IGNORECASE),
    re.compile(r"dd\s+if=", re.IGNORECASE),
    re.compile(r"mkfs", re.IGNORECASE),
    re.compile(r"shutdown", re.IGNORECASE),
    re.compile(r"reboot", re.IGNORECASE),
    # Recursive disk-wide scans — extremely slow and expose unrelated files
    re.compile(r"\bdir\b[^|&;]*\/s[^|&;]*[a-zA-Z]:\\", re.IGNORECASE),   # dir /s ... D:\...
    re.compile(r"\bdir\b[^|&;]*[a-zA-Z]:\\[^|&;]*\/s", re.IGNORECASE),   # dir D:\... /s
    re.compile(r"\bfind\b\s+[a-zA-Z]:\\\s", re.IGNORECASE),               # find D:\ ...
]


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode(locale.getpreferredencoding(False), errors="replace")


def _kill_tree(proc: subprocess.Popen) -> None:
    """Process ve tüm alt process'lerini öldür (Windows + Unix)."""
    try:
        if sys.platform == "win32":
            # /T: child tree dahil, /F: force
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    finally:
        try:
            proc.kill()
        except Exception:
            pass


class ShellRun(BaseTool):
    name = "shell_run"
    description = "Execute a shell command and return stdout/stderr."
    params = [
        ToolParam("command", "string", "Shell command to execute."),
        ToolParam("cwd", "string", "Working directory.", required=False, default=None),
        ToolParam("timeout", "integer", "Max seconds to wait.", required=False, default=30),
    ]

    def __init__(self, cancel_event: threading.Event | None = None):
        self._cancel = cancel_event or threading.Event()
        self.workspace: str | None = None

    def set_cancel_event(self, event: threading.Event) -> None:
        self._cancel = event

    def execute(self, command: str, cwd: str | None = None,
                timeout: int = 30, **_) -> ToolResult:
        for pattern in _BLOCKED_PATTERNS:
            if pattern.search(command):
                return ToolResult(
                    success=False, output="",
                    error=f"Command blocked by safety filter: {command!r}"
                )

        if sys.platform == "win32":
            command = f"chcp 65001 > nul 2>&1 & {command}"

        effective_cwd = cwd or self.workspace

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        # Windows'ta process group oluştur (Unix'te de aynısı)
        kwargs: dict = dict(
            shell=True,
            cwd=effective_cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        if sys.platform != "win32":
            kwargs["start_new_session"] = True  # yeni process group → killpg

        try:
            proc = subprocess.Popen(command, **kwargs)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

        # Cancel monitörü: cancel_event set edilince process tree'yi öldür
        cancelled = threading.Event()

        def _monitor():
            # cancel_event ya da timeout bekle
            self._cancel.wait(timeout=timeout + 1)
            if not cancelled.is_set():
                _kill_tree(proc)

        monitor_thread = threading.Thread(target=_monitor, daemon=True)
        monitor_thread.start()

        try:
            raw, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            cancelled.set()
            _kill_tree(proc)
            raw, _ = proc.communicate()
            return ToolResult(
                success=False,
                output=_decode(raw),
                error=f"Command timed out after {timeout}s",
            )
        finally:
            cancelled.set()  # monitor thread'e "process bitti, uyuma" sinyali

        if self._cancel.is_set():
            return ToolResult(success=False, output="", error="Cancelled by user.")

        stdout = _decode(raw)
        success = proc.returncode == 0
        return ToolResult(
            success=success,
            output=stdout,
            data={"returncode": proc.returncode},
            error=None if success else f"Exit code {proc.returncode}",
        )
