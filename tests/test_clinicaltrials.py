import httpx
import pytest
import respx

from app.adapters.clinicaltrials import search_clinical_trials
from app.config import CLINICALTRIALS_BASE
from app.sources.base import HttpClient


def make_study(nct_id="NCT00000001"):
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "briefTitle": "Study of example drug",
            },
            "statusModule": {
                "overallStatus": "COMPLETED",
                "studyFirstPostDateStruct": {"date": "2024-01-15"},
            },
            "descriptionModule": {"briefSummary": "Study summary."},
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "phases": ["PHASE2"],
                "enrollmentInfo": {"count": 120, "type": "ACTUAL"},
            },
            "conditionsModule": {"conditions": ["Condition A"]},
            "outcomesModule": {
                "primaryOutcomes": [
                    {"measure": "Response rate", "timeFrame": "12 weeks"}
                ]
            },
        },
        "resultsSection": {
            "adverseEventsModule": {
                "description": "Serious events were monitored.",
                "seriousEvents": [
                    {"organSystem": "Gastrointestinal", "term": "Nausea"}
                ],
                "eventGroups": [
                    {
                        "id": "EG1",
                        "title": "Treatment group",
                        "seriousNumAffected": 2,
                        "seriousNumAtRisk": 50,
                    }
                ],
            }
        },
        "hasResults": True,
    }


@pytest.mark.asyncio
async def test_search_clinical_trials_maps_v2_study_fields(tmp_path):
    url = f"{CLINICALTRIALS_BASE}/studies"
    async with respx.mock:
        route = respx.get(url).mock(
            return_value=httpx.Response(200, json={"studies": [make_study()]})
        )
        client = HttpClient(min_interval=0, cache_dir=str(tmp_path / "cache"))
        items = await search_clinical_trials("example drug", client=client)
        await client.close()

    assert len(items) == 1
    item = items[0]
    assert item.url == "https://clinicaltrials.gov/study/NCT00000001"
    assert item.title == "Study of example drug"
    assert item.publication_date == "2024-01-15"
    assert item.raw_payload["phase"] == ["PHASE2"]
    assert item.raw_payload["status"] == "COMPLETED"
    assert item.raw_payload["enrollment_count"] == 120
    assert item.raw_payload["conditions"] == ["Condition A"]
    assert item.raw_payload["primary_outcomes"][0]["measure"] == "Response rate"
    assert item.raw_payload["has_results"] is True
    assert "2/50 participants" in item.raw_payload["serious_adverse_event_summary"]
    assert "Gastrointestinal: Nausea" in item.raw_payload["serious_adverse_event_summary"]
    assert route.calls[0].request.url.params["query.intr"] == "example drug"


@pytest.mark.asyncio
async def test_search_clinical_trials_handles_no_results(tmp_path):
    url = f"{CLINICALTRIALS_BASE}/studies"
    async with respx.mock:
        respx.get(url).mock(return_value=httpx.Response(200, json={"studies": []}))
        client = HttpClient(min_interval=0, cache_dir=str(tmp_path / "cache"))
        items = await search_clinical_trials("no matching intervention", client=client)
        await client.close()

    assert items == []


@pytest.mark.asyncio
async def test_search_clinical_trials_caps_cursor_pagination(tmp_path):
    url = f"{CLINICALTRIALS_BASE}/studies"
    async with respx.mock:
        route = respx.get(url).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "studies": [make_study("NCT00000001"), make_study("NCT00000002")],
                        "nextPageToken": "page-2",
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "studies": [make_study("NCT00000003"), make_study("NCT00000004")],
                        "nextPageToken": "page-3",
                    },
                ),
            ]
        )
        client = HttpClient(min_interval=0, cache_dir=str(tmp_path / "cache"))
        items = await search_clinical_trials(
            "example drug", max_results=10, page_size=2, max_pages=2, client=client
        )
        await client.close()

    assert len(items) == 4
    assert len(route.calls) == 2
    assert route.calls[0].request.url.params["pageSize"] == "2"
    assert "pageToken" not in route.calls[0].request.url.params
    assert route.calls[1].request.url.params["pageToken"] == "page-2"