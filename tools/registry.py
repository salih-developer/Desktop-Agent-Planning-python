from __future__ import annotations

from tools.base import BaseTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}")
        return self._tools[name]

    def all_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def set_workspace(self, path: str) -> None:
        """Set the workspace path on every tool that supports it.

        Args:
            path: Absolute path to the workspace directory.
        """
        for tool in self._tools.values():
            if hasattr(tool, "workspace"):
                tool.workspace = path

    def to_text_manifest(self) -> str:
        return "\n\n".join(t.to_text_description() for t in self._tools.values())


def build_default_registry(workspace: str | None = None) -> ToolRegistry:
    from tools.file_tools import FileRead, FileWrite, FileEdit
    from tools.shell_tool import ShellRun
    from tools.web_tools import WebSearch, WebFetch
    from tools.filesystem_tool import DirectoryTree, FindFiles
    from tools.ssh_tool import SSHRun
    from tools.process_tool import ProcessStart, ProcessStop, ProcessList
    from tools.browser_tool import BrowserOpen

    registry = ToolRegistry()
    for tool in [
        FileRead(), FileWrite(), FileEdit(),
        ShellRun(),
        WebSearch(), WebFetch(),
        DirectoryTree(), FindFiles(),
        SSHRun(),
        ProcessStart(), ProcessStop(), ProcessList(),
        BrowserOpen(),
    ]:
        registry.register(tool)

    if workspace:
        registry.set_workspace(workspace)

    return registry
