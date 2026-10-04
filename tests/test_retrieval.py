import asyncio

import pytest

from app import graph
from app.adapters.rxnorm import RxNormResolution
from app.schemas import EvidenceItem


class FakeRxNormClient:
    def __init__(self, resolution, last_error=None):
        self.resolution = resolution
        self.last_error = last_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def resolve(self, drug_name):
        return self.resolution


def make_resolution():
    return RxNormResolution(
        input_name="Glucophage",
        rxcui="151827",
        canonical_name="metformin",
        synonyms=["metformin", "Glucophage"],
        ingredient_rxcuis=["6809"],
        matched_name="Glucophage",
    )


@pytest.mark.asyncio
async def test_normalize_node_stores_canonical_name_and_synonyms(monkeypatch):
    resolver = FakeRxNormClient(make_resolution())
    monkeypatch.setattr(graph, "RxNormClient", lambda: resolver)

    updates = await graph.normalize_node({"drug_query": "Glucophage"})

    assert updates["canonical_name"] == "metformin"
    assert updates["synonyms"] == ["metformin", "Glucophage"]
    assert updates["normalization_error"] == ""
    assert graph.route_after_normalization({"drug_query": "Glucophage", **updates}) == "retrieve_node"


@pytest.mark.asyncio
async def test_normalize_node_stops_with_clear_error_when_unresolved(monkeypatch):
    resolver = FakeRxNormClient(None, last_error="No RxNorm match found for 'unknown'.")
    monkeypatch.setattr(graph, "RxNormClient", lambda: resolver)

    updates = await graph.normalize_node({"drug_query": "unknown", "audit_log": []})

    assert updates["normalization_error"] == "No RxNorm match found for 'unknown'."
    assert updates["audit_log"] == [
        "Normalization failed: No RxNorm match found for 'unknown'."
    ]
    assert graph.route_after_normalization({"drug_query": "unknown", **updates}) == "END"


@pytest.mark.asyncio
async def test_retrieve_node_merges_successes_and_audits_failed_source(monkeypatch):
    started = 0
    all_started = asyncio.Event()

    class FakeHttpClient:
        closed = False

        async def close(self):
            self.closed = True

    client = FakeHttpClient()

    async def fetch(source_name, items, *, fail=False, drug_query=None):
        nonlocal started
        started += 1
        if started == 4:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=0.5)
        assert drug_query == "metformin"
        if fail:
            raise RuntimeError("source unavailable")
        return items

    def item(source_name):
        return EvidenceItem(
            source=source_name,
            title=f"{source_name} finding",
            summary="evidence",
            url="https://example.test/evidence",
        )

    async def fetch_pubmed(drug_query, max_results, client):
        return await fetch("PubMed", [item("PubMed")], drug_query=drug_query)

    async def fetch_trials(drug_query, max_results, client):
        return await fetch("ClinicalTrials.gov", [item("ClinicalTrials.gov")], drug_query=drug_query)

    async def fetch_labels(drug_query, max_results, client):
        return await fetch("openFDA labels", [item("openFDA labels")], drug_query=drug_query)

    async def fetch_faers(drug_query, max_results, client):
        return await fetch("openFDA FAERS", [], fail=True, drug_query=drug_query)

    monkeypatch.setattr(graph, "HttpClient", lambda: client)
    monkeypatch.setattr(graph, "search_pubmed", fetch_pubmed)
    monkeypatch.setattr(graph, "search_clinical_trials", fetch_trials)
    monkeypatch.setattr(graph, "search_openfda", fetch_labels)
    monkeypatch.setattr(graph, "search_openfda_faers", fetch_faers)

    updates = await graph.retrieve_node(
        {"drug_query": "Glucophage", "canonical_name": "metformin", "audit_log": []}
    )

    assert [document.source for document in updates["documents"]] == [
        "PubMed",
        "ClinicalTrials.gov",
        "openFDA labels",
    ]
    assert len(updates["literature_raw_data"]) == 3
    assert len(updates["audit_log"]) == 1
    assert "openFDA FAERS retrieval failed" in updates["audit_log"][0]
    assert "source unavailable" in updates["audit_log"][0]
    assert client.closed is True