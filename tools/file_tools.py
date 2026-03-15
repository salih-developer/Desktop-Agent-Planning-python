from __future__ import annotations

from pathlib import Path

from tools.base import BaseTool, ToolParam, ToolResult


def _replace_normalized(content: str, old_string: str, new_string: str) -> str | None:
    """Try to find old_string in content ignoring trailing whitespace on each line.
    Returns the updated content string, or None if not found.
    """
    content_lines = content.splitlines(keepends=True)
    old_lines = old_string.splitlines()
    if not old_lines:
        return None

    # Normalize: strip trailing whitespace for comparison only
    norm_content = [l.rstrip() for l in content_lines]
    norm_old = [l.rstrip() for l in old_lines]
    n = len(norm_old)

    for i in range(len(norm_content) - n + 1):
        if norm_content[i:i + n] == norm_old:
            # Replace matched lines with new_string lines
            new_lines = new_string.splitlines(keepends=True)
            if new_lines and not new_lines[-1].endswith(("\n", "\r")):
                # Preserve the line ending of the last replaced line
                ending = ""
                for ch in reversed(content_lines[i + n - 1]):
                    if ch in ("\n", "\r"):
                        ending = ch + ending
                    else:
                        break
                new_lines[-1] += ending
            result = content_lines[:i] + new_lines + content_lines[i + n:]
            return "".join(result)
    return None


def _jail(path: str, workspace: str) -> Path:
    """Resolve *path* and verify it stays inside *workspace*.

    - Relative paths are anchored to workspace.
    - Absolute paths must still be inside workspace.
    - Any attempt to escape via ``..`` or symlinks raises PermissionError.
    """
    ws = Path(workspace).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ws / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(ws):
        raise PermissionError(
            f"Access denied: '{path}' resolves to '{resolved}' which is outside "
            f"the workspace '{ws}'."
        )
    return resolved


class FileRead(BaseTool):
    name = "file_read"
    description = "Read a file, optionally limited to a line range."
    params = [
        ToolParam("path", "string", "File path (relative to workspace or absolute within workspace)."),
        ToolParam("start_line", "integer", "1-based start line for chunked reads.", required=False, default=None),
        ToolParam("end_line", "integer", "1-based end line for chunked reads.", required=False, default=None),
    ]

    def __init__(self, workspace: str = "."):
        self.workspace = workspace

    def execute(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        **_,
    ) -> ToolResult:
        try:
            resolved = _jail(path, self.workspace)
            content = resolved.read_text(encoding="utf-8", errors="replace")
            data = {"path": str(resolved), "size": len(content)}

            if start_line is None and end_line is None:
                return ToolResult(success=True, output=content, data=data)

            lines = content.splitlines(keepends=True)
            total_lines = len(lines)
            start = 1 if start_line is None else max(1, start_line)
            end = total_lines if end_line is None else min(total_lines, end_line)
            if start > total_lines:
                return ToolResult(
                    success=True,
                    output=f"(File has only {total_lines} line(s); start_line={start} is beyond end of file.)",
                    data={**data, "start_line": start, "end_line": end, "total_lines": total_lines, "chunked": True},
                )
            if end < start:
                end = start

            chunk = "".join(lines[start - 1 : end]) if total_lines else ""
            data.update(
                {
                    "start_line": start,
                    "end_line": end,
                    "total_lines": total_lines,
                    "chunked": True,
                }
            )
            return ToolResult(success=True, output=chunk, data=data)
        except PermissionError as e:
            return ToolResult(success=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class FileWrite(BaseTool):
    name = "file_write"
    description = "Write (overwrite) a file with given content."
    params = [
        ToolParam("path", "string", "File path (relative to workspace or absolute within workspace)."),
        ToolParam("content", "string", "Content to write."),
    ]

    def __init__(self, workspace: str = "."):
        self.workspace = workspace

    def execute(self, path: str, content: str, **_) -> ToolResult:
        try:
            resolved = _jail(path, self.workspace)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
            return ToolResult(success=True, output=f"Written {len(content)} chars to {resolved}")
        except PermissionError as e:
            return ToolResult(success=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class FileEdit(BaseTool):
    name = "file_edit"
    description = "Replace an exact string in a file with a new string."
    params = [
        ToolParam("path", "string", "File path (relative to workspace or absolute within workspace)."),
        ToolParam("old_string", "string", "Exact string to find and replace."),
        ToolParam("new_string", "string", "Replacement string."),
    ]

    def __init__(self, workspace: str = "."):
        self.workspace = workspace

    def execute(self, path: str, old_string: str, new_string: str, **_) -> ToolResult:
        try:
            resolved = _jail(path, self.workspace)
            original = resolved.read_text(encoding="utf-8")

            # 1. Exact match
            if old_string in original:
                updated = original.replace(old_string, new_string, 1)
                resolved.write_text(updated, encoding="utf-8")
                return ToolResult(success=True, output=f"Replaced in {resolved}")

            # 2. Normalized match (ignore trailing whitespace per line)
            updated = _replace_normalized(original, old_string, new_string)
            if updated is not None:
                resolved.write_text(updated, encoding="utf-8")
                return ToolResult(success=True, output=f"Replaced in {resolved} (whitespace-normalized match)")

            return ToolResult(success=False, output="", error="old_string not found in file.")
        except PermissionError as e:
            return ToolResult(success=False, output="", error=str(e))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
