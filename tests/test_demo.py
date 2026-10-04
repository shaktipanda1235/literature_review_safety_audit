import json

import pytest

from app import graph
from app.adapters.rxnorm import RxNormResolution
from app.schemas import EvidenceItem


@pytest.mark.asyncio
async def test_compiled_graph_runs(monkeypatch):
    """Run the async graph with mocked external services."""

    class FakeHttpClient:
        async def close(self):
            return None

    class FakeRxNormClient:
        last_error = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

        async def resolve(self, drug_name):
            return RxNormResolution(
                input_name=drug_name,
                rxcui="6809",
                canonical_name="metformin",
                synonyms=["metformin", "Glucophage"],
                ingredient_rxcuis=["6809"],
                matched_name="metformin",
            )

    class FakeStructuredGrader:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            payload = json.loads(messages[-1].content)
            return {
                "document_relevance": [
                    {"doc_id": item["doc_id"], "relevance": "relevant"}
                    for item in payload["documents"]
                ],
                "evidence_level": "moderate",
                "sufficient": True,
                "missing_topics": [],
            }

    def source_result(source_name):
        async def fetch(*args, **kwargs):
            return [
                EvidenceItem(
                    source=source_name,
                    title=f"{source_name} finding",
                    summary="evidence",
                    url="https://example.test/evidence",
                )
            ]

        return fetch

    async def failed_faers(*args, **kwargs):
        raise RuntimeError("FAERS unavailable")

    monkeypatch.setattr(graph, "HttpClient", FakeHttpClient)
    monkeypatch.setattr(graph, "RxNormClient", FakeRxNormClient)
    monkeypatch.setattr(graph, "get_llm", lambda role: FakeStructuredGrader())
    monkeypatch.setattr(graph, "search_pubmed", source_result("PubMed"))
    monkeypatch.setattr(graph, "search_clinical_trials", source_result("ClinicalTrials.gov"))
    monkeypatch.setattr(graph, "search_openfda", source_result("openFDA labels"))
    monkeypatch.setattr(graph, "search_openfda_faers", failed_faers)

    initial_state = {
        "drug_query": "Compound-X",
        "literature_raw_data": [],
        "web_fallback_data": [],
        "safety_violations": [],
        "regulatory_brief": "",
        "grade_decision": "clear",
        "human_approved": False,
        "chat_history": [],
    }

    config = {"configurable": {"thread_id": "test_run_1"}}

    result = await graph.compiled_pharma_graph.ainvoke(initial_state, config=config)

    assert isinstance(result, dict)
    assert result["canonical_name"] == "metformin"
    assert result.get("__interrupted_at") == "generator_node"
    assert [document.source for document in result["documents"]] == [
        "PubMed",
        "ClinicalTrials.gov",
        "openFDA labels",
    ]
    assert any("openFDA FAERS retrieval failed" in entry for entry in result["audit_log"])

