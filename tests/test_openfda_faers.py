import httpx
import pytest
import respx

from app.adapters.openfda_faers import FAERS_CAVEAT, search_openfda_faers
from app.config import OPENFDA_BASE
from app.sources.base import HttpClient


@pytest.mark.asyncio
async def test_search_openfda_faers_returns_ranked_top_twenty_and_caveat(tmp_path):
    url = f"{OPENFDA_BASE}/event.json"
    response_results = [
        {"term": f"REACTION_{index:02}", "count": 25 - index}
        for index in range(25)
    ]

    async with respx.mock:
        route = respx.get(url).mock(
            return_value=httpx.Response(200, json={"results": response_results})
        )
        client = HttpClient(min_interval=0, cache_dir=str(tmp_path / "cache"))
        items = await search_openfda_faers("ketoconazole", client=client)
        await client.close()

    assert len(items) == 1
    item = items[0]
    reactions = item.raw_payload["reactions"]
    assert len(reactions) == 20
    assert reactions[0] == {"rank": 1, "term": "REACTION_00", "count": 25}
    assert reactions[-1] == {"rank": 20, "term": "REACTION_19", "count": 6}
    assert "REACTION_00" in item.summary
    assert FAERS_CAVEAT in item.summary
    assert item.raw_payload["caveat"] == FAERS_CAVEAT
    assert item.raw_payload["count_field"] == "patient.reaction.reactionmeddrapt.exact"
    assert route.calls[0].request.url.params["count"] == "patient.reaction.reactionmeddrapt.exact"
    assert route.calls[0].request.url.params["limit"] == "20"


@pytest.mark.asyncio
async def test_search_openfda_faers_handles_no_results(tmp_path):
    url = f"{OPENFDA_BASE}/event.json"
    async with respx.mock:
        route = respx.get(url).mock(
            return_value=httpx.Response(
                404,
                json={"error": {"code": "NOT_FOUND", "message": "No matches found!"}},
            )
        )
        client = HttpClient(min_interval=0, cache_dir=str(tmp_path / "cache"))
        items = await search_openfda_faers("unknown-drug", client=client)
        await client.close()

    assert items == []
    assert len(route.calls) == 1