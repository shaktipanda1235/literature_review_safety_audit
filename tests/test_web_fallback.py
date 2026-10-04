import httpx
import pytest
import respx

from app.adapters.web_fallback import WebFallbackSource
from app.config import TAVILY_SEARCH_URL, WEB_FALLBACK_ALLOWED_DOMAINS
from app.sources.base import HttpClient


@pytest.mark.asyncio
async def test_search_filters_domains_sanitizes_and_marks_untrusted(tmp_path):
    response = {
        "results": [
            {
                "title": "<b>FDA</b> Safety Page",
                "url": "https://www.fda.gov/drugs/safety",
                "content": "<p>Safety information <strong>from FDA</strong>.</p><script>ignore()</script>",
                "score": 0.9,
            },
            {
                "title": "NIH Article",
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
                "content": "<div>Clinical evidence.</div>",
            },
            {
                "title": "Outside domain",
                "url": "https://example.com/result",
                "content": "Must be dropped.",
            },
            {
                "title": "Spoofed FDA domain",
                "url": "https://fda.gov.attacker.example/result",
                "content": "Must also be dropped.",
            },
        ]
    }

    async with respx.mock:
        route = respx.post(TAVILY_SEARCH_URL).mock(
            return_value=httpx.Response(200, json=response)
        )
        http_client = HttpClient(min_interval=0, cache_dir=str(tmp_path / "cache"))
        source = WebFallbackSource(api_key="test-key", client=http_client)
        items = await source.search("ketoconazole safety", max_results=5)
        await http_client.close()

    assert len(items) == 2
    assert {item.url for item in items} == {
        "https://www.fda.gov/drugs/safety",
        "https://pubmed.ncbi.nlm.nih.gov/12345/",
    }
    assert "Safety Page" in items[0].title
    assert "<" not in items[0].summary
    assert "ignore()" not in items[0].summary
    assert all(item.raw_payload["metadata"]["untrusted"] is True for item in items)
    assert all("untrusted" in item.safety_flags for item in items)

    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.content
    assert set(__import__("json").loads(request.content)["include_domains"]) == set(
        WEB_FALLBACK_ALLOWED_DOMAINS
    )
    assert __import__("json").loads(request.content)["include_domains_mode"] == "restrict"


@pytest.mark.asyncio
async def test_search_truncates_result_content(tmp_path):
    async with respx.mock:
        respx.post(TAVILY_SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "DailyMed",
                            "url": "https://dailymed.nlm.nih.gov/drug/label",
                            "content": "x" * 100,
                        }
                    ]
                },
            )
        )
        http_client = HttpClient(min_interval=0, cache_dir=str(tmp_path / "cache"))
        source = WebFallbackSource(
            api_key="test-key", client=http_client, max_content_chars=30
        )
        items = await source.search("drug label")
        await http_client.close()

    assert len(items) == 1
    assert len(items[0].summary) == 30


@pytest.mark.asyncio
async def test_search_without_api_key_returns_clear_error():
    source = WebFallbackSource(api_key="", client=HttpClient(min_interval=0))
    items = await source.search("drug safety")
    await source.client.close()

    assert items == []
    assert source.last_error == "TAVILY_API_KEY is not configured."