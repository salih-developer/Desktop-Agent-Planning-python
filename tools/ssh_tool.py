from __future__ import annotations

import logging
import socket
from pathlib import Path

from tools.base import BaseTool, ToolParam, ToolResult

try:
    import paramiko
    _PARAMIKO_AVAILABLE = True
except ImportError:
    _PARAMIKO_AVAILABLE = False

_log = logging.getLogger(__name__)


class SSHRun(BaseTool):
    """Execute shell commands on remote hosts via SSH."""

    name = "ssh_run"
    description = (
        "Execute a command on a remote host via SSH and return stdout/stderr. "
        "Supports password or private-key authentication. "
        "Use this to check Kubernetes pods, Docker images, logs, or any remote shell command."
    )
    params = [
        ToolParam("host", "string", "Remote host IP or hostname."),
        ToolParam("command", "string", "Shell command to execute on the remote host."),
        ToolParam("username", "string", "SSH username.", required=False, default="root"),
        ToolParam(
            "password", "string",
            "SSH password. Leave empty to use key_path.",
            required=False, default=None,
        ),
        ToolParam(
            "key_path", "string",
            "Path to private key file (e.g. ~/.ssh/id_rsa). Used when password is not set.",
            required=False, default=None,
        ),
        ToolParam("port", "integer", "SSH port.", required=False, default=22),
        ToolParam("timeout", "integer", "Connection timeout in seconds.", required=False, default=30),
    ]

    def execute(
        self,
        host: str,
        command: str,
        username: str = "root",
        password: str | None = None,
        key_path: str | None = None,
        port: int = 22,
        timeout: int = 30,
        **_,
    ) -> ToolResult:
        """Run a command on a remote host over SSH.

        Args:
            host: Remote host IP or hostname.
            command: Shell command to execute.
            username: SSH username.
            password: SSH password (mutually exclusive with key_path).
            key_path: Path to private key file (e.g. ``~/.ssh/id_rsa``).
            port: SSH port (default 22).
            timeout: Connection and exec timeout in seconds.

        Returns:
            ToolResult with stdout (and stderr if non-empty) as output.
        """
        if not _PARAMIKO_AVAILABLE:
            return ToolResult(
                success=False, output="",
                error="paramiko is not installed. Run: pip install paramiko",
            )

        connect_kwargs: dict = dict(
            hostname=host,
            port=port,
            username=username,
            timeout=timeout,
            banner_timeout=timeout,
        )
        if password:
            connect_kwargs["password"] = password
        elif key_path:
            connect_kwargs["key_filename"] = str(Path(key_path).expanduser())
        else:
            connect_kwargs["allow_agent"] = True
            connect_kwargs["look_for_keys"] = True

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            client.connect(**connect_kwargs)
        except paramiko.AuthenticationException as exc:
            _log.warning("SSH auth failed for %s@%s: %s", username, host, exc)
            return ToolResult(success=False, output="", error=f"SSH authentication failed: {exc}")
        except (paramiko.SSHException, socket.error, OSError) as exc:
            _log.warning("SSH connect failed to %s:%s — %s", host, port, exc)
            return ToolResult(success=False, output="", error=f"SSH connect failed: {exc}")

        try:
            _, stdout, stderr = client.exec_command(command, timeout=timeout)
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            rc = stdout.channel.recv_exit_status()
        except (paramiko.SSHException, socket.timeout, OSError) as exc:
            _log.warning("SSH exec failed on %s: %s", host, exc)
            return ToolResult(success=False, output="", error=f"SSH exec failed: {exc}")
        finally:
            client.close()

        combined = out
        if err.strip():
            combined += f"\n[stderr]\n{err}"

        success = rc == 0
        _log.debug("ssh_run %s %r → rc=%d, %d chars", host, command[:60], rc, len(combined))
        return ToolResult(
            success=success,
            output=combined,
            data={"returncode": rc, "host": host},
            error=None if success else f"Exit code {rc}",
        )
