"""Playwright-based browser automation tools.

Provides stateful browser control (navigate, click, fill, screenshot, etc.)
using a module-level singleton so the browser persists across tool calls
within a single agent session.

Install: pip install playwright && playwright install chromium
"""
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import ClassVar

from tools.base import BaseTool, ToolParam, ToolResult


# ── Shared browser state ───────────────────────────────────────────────────────

class _BrowserState:
    """Module-level singleton that owns the Playwright browser/page lifecycle."""

    _playwright = None
    _browser = None
    _page = None

    @classmethod
    def page(cls):
        """Return the active page, launching the browser if needed."""
        if cls._page is None or cls._page.is_closed():
            cls._ensure_browser()
            cls._page = cls._browser.new_page()
        return cls._page

    @classmethod
    def _ensure_browser(cls):
        if cls._browser is None or not cls._browser.is_connected():
            from playwright.sync_api import sync_playwright  # lazy import
            cls._playwright = sync_playwright().start()
            cls._browser = cls._playwright.chromium.launch(headless=False)

    @classmethod
    def close(cls):
        try:
            if cls._page and not cls._page.is_closed():
                cls._page.close()
        except Exception:
            pass
        try:
            if cls._browser and cls._browser.is_connected():
                cls._browser.close()
        except Exception:
            pass
        try:
            if cls._playwright:
                cls._playwright.stop()
        except Exception:
            pass
        cls._page = None
        cls._browser = None
        cls._playwright = None


# ── Tool implementations ───────────────────────────────────────────────────────

class PlaywrightNavigate(BaseTool):
    name = "playwright_navigate"
    description = (
        "Navigate the browser to a URL and return the page title and a text "
        "snippet of visible content. Opens the browser automatically if not running. "
        "Use this as the first step before any click/fill/screenshot operations."
    )
    params = [
        ToolParam("url", "string", "Full URL to navigate to (e.g. 'http://localhost:3000')", required=True),
        ToolParam("wait_until", "string",
                  "When to consider navigation done: 'load' (default), 'domcontentloaded', 'networkidle'",
                  required=False, default="load",
                  enum=["load", "domcontentloaded", "networkidle"]),
        ToolParam("timeout", "integer",
                  "Navigation timeout in milliseconds (default 15000)", required=False, default=15000),
    ]

    def execute(self, url: str, wait_until: str = "load", timeout: int = 15000, **_) -> ToolResult:
        try:
            page = _BrowserState.page()
            page.goto(url, wait_until=wait_until, timeout=timeout)
            title = page.title()
            # Grab first ~800 chars of visible body text as a content snapshot
            try:
                body_text = page.inner_text("body")[:800].strip()
            except Exception:
                body_text = ""
            output = f"Navigated to: {url}\nTitle: {title}"
            if body_text:
                output += f"\n\nPage content (first 800 chars):\n{body_text}"
            return ToolResult(success=True, output=output, data={"title": title, "url": page.url})
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class PlaywrightScreenshot(BaseTool):
    name = "playwright_screenshot"
    description = (
        "Take a full-page screenshot of the current browser page. "
        "Saves the image to a file and returns the file path. "
        "Use this to visually verify page state, capture test evidence, or inspect UI."
    )
    params = [
        ToolParam("path", "string",
                  "File path to save the screenshot (e.g. 'screenshot.png'). "
                  "Defaults to 'playwright_screenshot.png' in the current directory.",
                  required=False, default="playwright_screenshot.png"),
        ToolParam("full_page", "boolean",
                  "Capture the full scrollable page (True) or only the visible viewport (False, default)",
                  required=False, default=False),
    ]

    def execute(self, path: str = "playwright_screenshot.png", full_page: bool = False, **_) -> ToolResult:
        try:
            page = _BrowserState.page()
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_bytes = page.screenshot(path=str(out_path), full_page=full_page)
            abs_path = str(out_path.resolve())
            image_b64 = base64.b64encode(screenshot_bytes).decode()
            return ToolResult(
                success=True,
                output=f"Screenshot saved to: {abs_path}",
                data={"path": abs_path, "image_base64": image_b64},
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class PlaywrightClick(BaseTool):
    name = "playwright_click"
    description = (
        "Click an element on the current page identified by a CSS selector or visible text. "
        "Examples: selector='button#submit', selector='text=Login', selector='.nav-link'. "
        "Waits up to the timeout for the element to be visible before clicking."
    )
    params = [
        ToolParam("selector", "string",
                  "CSS selector or Playwright text selector (e.g. 'button#submit', 'text=Login')",
                  required=True),
        ToolParam("timeout", "integer",
                  "How long to wait for the element to appear in ms (default 5000)",
                  required=False, default=5000),
    ]

    def execute(self, selector: str, timeout: int = 5000, **_) -> ToolResult:
        try:
            page = _BrowserState.page()
            page.click(selector, timeout=timeout)
            time.sleep(0.3)  # brief settle after click
            return ToolResult(
                success=True,
                output=f"Clicked element: {selector!r}\nCurrent URL: {page.url}",
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class PlaywrightFill(BaseTool):
    name = "playwright_fill"
    description = (
        "Clear and fill a text input or textarea on the current page. "
        "Locates the element by CSS selector, clears its current value, then types the new value."
    )
    params = [
        ToolParam("selector", "string",
                  "CSS selector for the input field (e.g. 'input[name=username]', '#search')",
                  required=True),
        ToolParam("value", "string", "Text to enter into the field", required=True),
        ToolParam("timeout", "integer",
                  "How long to wait for the element in ms (default 5000)",
                  required=False, default=5000),
    ]

    def execute(self, selector: str, value: str, timeout: int = 5000, **_) -> ToolResult:
        try:
            page = _BrowserState.page()
            page.fill(selector, value, timeout=timeout)
            return ToolResult(
                success=True,
                output=f"Filled {selector!r} with value: {value!r}",
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class PlaywrightGetText(BaseTool):
    name = "playwright_get_text"
    description = (
        "Get the visible text content of an element or the entire page body. "
        "Use this to read form values, verify displayed text, or extract table data."
    )
    params = [
        ToolParam("selector", "string",
                  "CSS selector of the element to read. Use 'body' (default) for the full page.",
                  required=False, default="body"),
        ToolParam("max_chars", "integer",
                  "Maximum characters to return (default 2000)", required=False, default=2000),
    ]

    def execute(self, selector: str = "body", max_chars: int = 2000, **_) -> ToolResult:
        try:
            page = _BrowserState.page()
            text = page.inner_text(selector)[:max_chars]
            return ToolResult(success=True, output=text, data={"length": len(text)})
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class PlaywrightEvaluate(BaseTool):
    name = "playwright_evaluate"
    description = (
        "Execute JavaScript in the browser page context and return the result. "
        "Useful for reading DOM state, triggering events, or running assertions. "
        "Example: script='document.title' or script='window.location.href'."
    )
    params = [
        ToolParam("script", "string",
                  "JavaScript expression or statement to evaluate. "
                  "Return a value with a plain expression or explicit return statement.",
                  required=True),
    ]

    def execute(self, script: str, **_) -> ToolResult:
        try:
            page = _BrowserState.page()
            result = page.evaluate(script)
            output = str(result) if result is not None else "(null)"
            return ToolResult(success=True, output=output, data={"result": result})
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))


class PlaywrightClose(BaseTool):
    name = "playwright_close"
    description = (
        "Close the browser and release all Playwright resources. "
        "Call this when browser testing is finished to free memory."
    )
    params = []

    def execute(self, **_) -> ToolResult:
        try:
            _BrowserState.close()
            return ToolResult(success=True, output="Browser closed successfully.")
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
