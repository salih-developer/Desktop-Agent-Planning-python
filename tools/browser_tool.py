from __future__ import annotations

import webbrowser
from typing import ClassVar

from tools.base import BaseTool, ToolParam, ToolResult


class BrowserOpen(BaseTool):
    name = "browser_open"
    description = (
        "Open one or more URLs in the user's default browser. "
        "Use this to actually display a web page on screen — not just fetch its content. "
        "Accepts a single URL or a comma-separated list of URLs."
    )
    params = [
        ToolParam(
            "urls",
            "string",
            "URL or comma-separated list of URLs to open (e.g. 'http://localhost:5516' "
            "or 'http://localhost:5516, http://localhost:5500/swagger')",
            required=True,
        ),
    ]

    def execute(self, urls: str, **_) -> ToolResult:  # type: ignore[override]
        url_list = [u.strip() for u in urls.split(",") if u.strip()]
        if not url_list:
            return ToolResult(success=False, output="", error="No URLs provided.")

        opened = []
        failed = []
        for url in url_list:
            try:
                webbrowser.open(url)
                opened.append(url)
            except Exception as exc:
                failed.append(f"{url}: {exc}")

        if failed:
            return ToolResult(
                success=len(opened) > 0,
                output=f"Opened: {', '.join(opened)}" if opened else "",
                error=f"Failed to open: {'; '.join(failed)}",
            )
        return ToolResult(
            success=True,
            output=f"Opened {len(opened)} URL(s) in browser: {', '.join(opened)}",
        )
