from __future__ import annotations

import html.parser
import re
import urllib.parse

import httpx

from tools.base import BaseTool, ToolParam, ToolResult

_MAX_CONTENT = 8000


class _TextExtractor(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self._skip_tags = {"script", "style", "head", "noscript"}
        self._in_skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._in_skip += 1

    def handle_endtag(self, tag):
        if tag in self._skip_tags and self._in_skip:
            self._in_skip -= 1

    def handle_data(self, data):
        if not self._in_skip:
            stripped = data.strip()
            if stripped:
                self.parts.append(stripped)

    def get_text(self) -> str:
        return " ".join(self.parts)


def _html_to_text(html_str: str) -> str:
    parser = _TextExtractor()
    parser.feed(html_str)
    text = parser.get_text()
    text = re.sub(r"\s{2,}", " ", text)
    return text[:_MAX_CONTENT]


class WebSearch(BaseTool):
    name = "web_search"
    description = "Search the web using DuckDuckGo and return a list of result titles and URLs."
    params = [
        ToolParam("query", "string", "Search query."),
        ToolParam("max_results", "integer", "Max results to return.", required=False, default=5),
    ]

    def execute(self, query: str, max_results: int = 5, **_) -> ToolResult:
        try:
            encoded = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded}"
            headers = {"User-Agent": "Mozilla/5.0 (compatible; DesktopAgent/1.0)"}
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)

            class LinkParser(html.parser.HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.results: list[dict] = []
                    self._in_result = False
                    self._current_href = None
                    self._current_title = []

                def handle_starttag(self, tag, attrs):
                    attrs_dict = dict(attrs)
                    if tag == "a" and "result__a" in attrs_dict.get("class", ""):
                        self._in_result = True
                        href = attrs_dict.get("href", "")
                        # DuckDuckGo wraps URLs
                        if href.startswith("//duckduckgo.com/l/?uddg="):
                            parsed = urllib.parse.parse_qs(
                                urllib.parse.urlparse(href).query
                            )
                            href = urllib.parse.unquote(parsed.get("uddg", [href])[0])
                        self._current_href = href
                        self._current_title = []

                def handle_endtag(self, tag):
                    if tag == "a" and self._in_result:
                        self._in_result = False
                        title = "".join(self._current_title).strip()
                        if title and self._current_href:
                            self.results.append({"title": title, "url": self._current_href})

                def handle_data(self, data):
                    if self._in_result:
                        self._current_title.append(data)

            parser = LinkParser()
            parser.feed(resp.text)
            results = parser.results[:max_results]

            lines = [f"{i+1}. {r['title']}\n   {r['url']}" for i, r in enumerate(results)]
            return ToolResult(success=True, output="\n".join(lines), data=results)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class WebFetch(BaseTool):
    name = "web_fetch"
    description = "Fetch a URL and return its text content (max 8000 chars)."
    params = [
        ToolParam("url", "string", "URL to fetch."),
    ]

    def execute(self, url: str, **_) -> ToolResult:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; DesktopAgent/1.0)"}
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
            content_type = resp.headers.get("content-type", "")
            if "html" in content_type:
                text = _html_to_text(resp.text)
            else:
                text = resp.text[:_MAX_CONTENT]
            return ToolResult(success=True, output=text, data={"url": url, "status": resp.status_code})
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
