import httpx
import pytest
import respx

from app.adapters.rxnorm import RxNormClient
from app.config import RXNAV_BASE
from app.sources.base import HttpClient


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["Glucophage", "Glucophge"])
async def test_resolve_brand_or_misspelling_to_canonical_ingredient(query, tmp_path):
    approximate_url = f"{RXNAV_BASE}/approximateTerm.json"
    related_url = f"{RXNAV_BASE}/rxcui/151827/related.json"

    async with respx.mock:
        approximate_route = respx.get(approximate_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "approximateGroup": {
                        "candidate": [
                            {
                                "rxcui": "151827",
                                "name": "Glucophage",
                                "score": "14.19",
                                "rank": "1",
                                "source": "RXNORM",
                            }
                        ]
                    }
                },
            )
        )
        respx.get(related_url).mock(
            return_value=httpx.Response(
                200,
                json={
                    "relatedGroup": {
                        "conceptGroup": [
                            {
                                "tty": "IN",
                                "conceptProperties": [
                                    {
                                        "rxcui": "6809",
                                        "name": "metformin",
                                        "synonym": "",
                                        "tty": "IN",
                                    }
                                ],
                            }
                        ]
                    }
                },
            )
        )
        http_client = HttpClient(min_interval=0, cache_dir=str(tmp_path / "cache"))
        rxnorm = RxNormClient(client=http_client)
        resolution = await rxnorm.resolve(query)
        await http_client.close()

    assert resolution is not None
    assert resolution.canonical_name == "metformin"
    assert resolution.rxcui == "151827"
    assert resolution.ingredient_rxcuis == ["6809"]
    assert "Glucophage" in resolution.synonyms
    assert "metformin" in resolution.synonyms
    assert approximate_route.calls[0].request.url.params["term"] == query
    assert approximate_route.calls[0].request.url.params["option"] == "1"


@pytest.mark.asyncio
async def test_unknown_drug_returns_none_with_reason(tmp_path):
    approximate_url = f"{RXNAV_BASE}/approximateTerm.json"
    async with respx.mock:
        respx.get(approximate_url).mock(
            return_value=httpx.Response(200, json={"approximateGroup": {"inputTerm": None}})
        )
        http_client = HttpClient(min_interval=0, cache_dir=str(tmp_path / "cache"))
        rxnorm = RxNormClient(client=http_client)
        resolution = await rxnorm.resolve("notarealdrugnamexyzq")
        await http_client.close()

    assert resolution is None
    assert rxnorm.last_error == "No RxNorm match found for 'notarealdrugnamexyzq'."