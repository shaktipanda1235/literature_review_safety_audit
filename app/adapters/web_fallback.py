from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from typing import Any, List
from urllib.parse import urlsplit

from app.config import (
    MAX_RESULTS_PER_SOURCE,
    TAVILY_SEARCH_URL,
    WEB_FALLBACK_ALLOWED_DOMAINS,
    WEB_FALLBACK_MAX_CONTENT_CHARS,
)
from app.schemas import EvidenceItem
from app.sources.base import HttpClient


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
        elif not self._skip_depth and tag in {"br", "div", "li", "p", "section", "tr"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
        elif not self._skip_depth and tag in {"br", "div", "li", "p", "section", "tr"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def _sanitize_text(value: Any, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    parser = _HTMLTextExtractor()
    parser.feed(value)
    parser.close()
    text = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    return text[:max_chars].rstrip()


def _allowed_domain(url: Any, allowlist: tuple[str, ...]) -> str | None:
    if not isinstance(url, str):
        return None
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    hostname = parsed.hostname.casefold().rstrip(".")
    for domain in allowlist:
        allowed = domain.casefold().strip(".")
        if hostname == allowed or hostname.endswith(f".{allowed}"):
            return hostname
    return None


class WebFallbackSource:
    """Tavily search restricted to trusted source domains; results remain untrusted."""

    def __init__(
        self,
        api_key: str | None = None,
        client: HttpClient | None = None,
        allowed_domains: tuple[str, ...] | None = None,
        max_content_chars: int = WEB_FALLBACK_MAX_CONTENT_CHARS,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("TAVILY_API_KEY")
        self.client = client or HttpClient()
        self._owns_client = client is None
        self.allowed_domains = allowed_domains or WEB_FALLBACK_ALLOWED_DOMAINS
        self.max_content_chars = max(1, max_content_chars)
        self.last_error: str | None = None

    async def __aenter__(self) -> WebFallbackSource:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self.client.close()
            self._owns_client = False

    async def search(
        self, query: str, max_results: int = MAX_RESULTS_PER_SOURCE
    ) -> List[EvidenceItem]:
        self.last_error = None
        if not query.strip():
            self.last_error = "Search query is empty."
            return []
        if not self.api_key:
            self.last_error = "TAVILY_API_KEY is not configured."
            return []
        if max_results <= 0:
            return []

        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": min(max_results, 20),
            "include_domains": list(self.allowed_domains),
            "include_domains_mode": "restrict",
            "include_raw_content": "text",
        }
        try:
            response = await self.client.request(
                "POST",
                TAVILY_SEARCH_URL,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                use_cache=False,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as error:
            self.last_error = f"Tavily search failed: {error}"
            return []

        if not isinstance(data, dict):
            self.last_error = "Tavily returned an invalid response."
            return []

        evidence: List[EvidenceItem] = []
        for result in data.get("results") or []:
            if not isinstance(result, dict):
                continue
            result_url = result.get("url")
            hostname = _allowed_domain(result_url, self.allowed_domains)
            if not hostname:
                continue

            summary = _sanitize_text(
                result.get("raw_content") or result.get("content"),
                self.max_content_chars,
            )
            title = _sanitize_text(result.get("title"), 300) or hostname
            if not summary:
                continue

            evidence.append(
                EvidenceItem(
                    source="Web fallback",
                    title=title,
                    summary=summary,
                    url=result_url,
                    publication_date=result.get("published_date"),
                    study_type="web search result",
                    safety_flags=["untrusted"],
                    raw_payload={
                        "query": query,
                        "metadata": {"untrusted": True, "domain": hostname},
                        "score": result.get("score"),
                    },
                )
            )

            if len(evidence) >= min(max_results, 20):
                break

        return evidence