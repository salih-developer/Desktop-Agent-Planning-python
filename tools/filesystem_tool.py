from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from tools.base import BaseTool, ToolParam, ToolResult


class DirectoryTree(BaseTool):
    name = "dir_tree"
    description = "List the directory tree starting from a path."
    params = [
        ToolParam("path", "string", "Root directory path. If omitted uses workspace.", required=False, default=None),
        ToolParam("max_depth", "integer", "Max depth to traverse.", required=False, default=3),
    ]

    def __init__(self):
        self.workspace: str | None = None

    def execute(self, path: str | None = None, max_depth: int = 3, **_) -> ToolResult:
        base = Path(self.workspace) if self.workspace else Path.home()
        root_path = path or self.workspace or str(Path.home())
        candidate = Path(root_path)
        if not candidate.is_absolute():
            candidate = base / candidate
        path = str(candidate)
        try:
            root = Path(path)
            if not root.exists():
                return ToolResult(success=False, output="", error=f"Path does not exist: {path}")
            lines = [str(root)]
            self._walk(root, "", 0, max_depth, lines)
            output = "\n".join(lines)
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def _walk(self, path: Path, prefix: str, depth: int, max_depth: int, lines: list[str]) -> None:
        if depth >= max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                self._walk(entry, prefix + extension, depth + 1, max_depth, lines)


class FindFiles(BaseTool):
    name = "find_files"
    description = "Find files matching a glob pattern in a directory."
    params = [
        ToolParam("path", "string", "Root directory to search in. If omitted uses workspace.", required=False, default=None),
        ToolParam("pattern", "string", "Glob pattern, e.g. '*.py' or '**/*.ts'."),
        ToolParam("max_results", "integer", "Max number of results.", required=False, default=50),
    ]

    def __init__(self):
        self.workspace: str | None = None

    def execute(self, path: str | None = None, pattern: str = "*", max_results: int = 50, **_) -> ToolResult:
        base = Path(self.workspace) if self.workspace else Path.home()
        root_path = path or self.workspace or str(Path.home())
        candidate = Path(root_path)
        if not candidate.is_absolute():
            candidate = base / candidate
        path = str(candidate)
        try:
            root = Path(path)
            matches = []
            for entry in root.rglob("*"):
                if fnmatch.fnmatch(entry.name, pattern.lstrip("**/")) or fnmatch.fnmatch(
                    str(entry.relative_to(root)), pattern
                ):
                    matches.append(str(entry))
                    if len(matches) >= max_results:
                        break
            output = "\n".join(matches) if matches else "(no matches)"
            return ToolResult(
                success=True,
                output=output,
                data={
                    "count": len(matches),
                    "first_path": matches[0] if matches else "",
                    "has_matches": bool(matches),
                },
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
