import httpx
import pytest
import respx

from app.adapters.openfda import search_openfda
from app.config import OPENFDA_BASE
from app.sources.base import HttpClient


def make_label(boxed_warning=True):
    label = {
        "id": "label-id-1",
        "set_id": "set-id-1",
        "contraindications": ["Contraindication text."],
        "warnings": ["Warnings and precautions text."],
        "adverse_reactions": ["Adverse reaction text."],
        "drug_interactions": ["Drug interaction text."],
        "pregnancy": ["Population text."],
        "pediatric_use": ["Pediatric population text."],
        "openfda": {
            "brand_name": ["KETOCONAZOLE"],
            "generic_name": ["KETOCONAZOLE"],
            "route": ["ORAL"],
        },
    }
    if boxed_warning:
        label["boxed_warning"] = [
            "Serious hepatotoxicity has occurred with oral ketoconazole."
        ]
    return label


@pytest.mark.asyncio
async def test_search_openfda_emits_one_item_per_label_section(tmp_path):
    url = f"{OPENFDA_BASE}/label.json"
    async with respx.mock:
        route = respx.get(url).mock(
            return_value=httpx.Response(200, json={"results": [make_label()]})
        )
        client = HttpClient(min_interval=0, cache_dir=str(tmp_path / "cache"))
        items = await search_openfda("ketoconazole", client=client)
        await client.close()

    sections = {item.raw_payload["section"]: item for item in items}
    assert set(sections) == {
        "boxed_warning",
        "contraindications",
        "warnings_and_cautions",
        "adverse_reactions",
        "drug_interactions",
        "use_in_specific_populations",
    }
    assert "hepatotoxicity" in sections["boxed_warning"].summary.lower()
    assert sections["warnings_and_cautions"].raw_payload["source_fields"] == ["warnings"]
    assert sections["use_in_specific_populations"].raw_payload["source_fields"] == [
        "pregnancy",
        "pediatric_use",
    ]
    assert sections["boxed_warning"].url.startswith(url)
    assert len(route.calls) == 1
    search = route.calls[0].request.url.params["search"]
    assert "openfda.generic_name" in search
    assert "openfda.brand_name" in search


@pytest.mark.asyncio
async def test_search_openfda_finds_ketoconazole_boxed_warning(tmp_path):
    url = f"{OPENFDA_BASE}/label.json"
    oral_label = make_label()
    oral_label["id"] = "label-id-2"
    oral_label["set_id"] = "set-id-2"
    async with respx.mock:
        route = respx.get(url).mock(
            side_effect=[
                httpx.Response(200, json={"results": [make_label(boxed_warning=False)]}),
                httpx.Response(200, json={"results": [oral_label]}),
            ]
        )
        client = HttpClient(min_interval=0, cache_dir=str(tmp_path / "cache"))
        items = await search_openfda("ketoconazole", client=client)
        await client.close()

    boxed_warnings = [
        item for item in items if item.raw_payload["section"] == "boxed_warning"
    ]
    assert len(boxed_warnings) == 1
    assert "hepatotoxicity" in boxed_warnings[0].summary.lower()
    assert len(route.calls) == 2
    assert "boxed_warning:hepatotoxicity" in route.calls[1].request.url.params["search"]


@pytest.mark.asyncio
async def test_search_openfda_returns_empty_for_unknown_drug(tmp_path):
    url = f"{OPENFDA_BASE}/label.json"
    async with respx.mock:
        route = respx.get(url).mock(
            return_value=httpx.Response(
                404,
                json={"error": {"code": "NOT_FOUND", "message": "No matches found!"}},
            )
        )
        client = HttpClient(min_interval=0, cache_dir=str(tmp_path / "cache"))
        items = await search_openfda("unknown-drug", client=client)
        await client.close()

    assert items == []
    assert len(route.calls) == 1