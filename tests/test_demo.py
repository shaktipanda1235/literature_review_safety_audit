import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app import graph
from app.adapters.rxnorm import RxNormResolution
from app.schemas import Brief, CitedClaim, EvidenceItem


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "edit", "reject"])
async def test_compiled_graph_runs_human_review_decisions(monkeypatch, decision):
    """Run the graph to human review and resume each supported decision."""

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

    async def fake_draft(state):
        doc_ids = [document.doc_id for document in state.get("documents", [])]
        brief = Brief(
            summary=CitedClaim(text="Evidence is limited to retrieved sources.", doc_ids=doc_ids),
            evidence_level="moderate",
        )
        return {"brief": brief, "regulatory_brief": "Mock regulatory brief."}

    monkeypatch.setattr(graph, "HttpClient", FakeHttpClient)
    monkeypatch.setattr(graph, "RxNormClient", FakeRxNormClient)
    monkeypatch.setattr(graph, "get_llm", lambda role: FakeStructuredGrader())
    monkeypatch.setattr(graph, "search_pubmed", source_result("PubMed"))
    monkeypatch.setattr(graph, "search_clinical_trials", source_result("ClinicalTrials.gov"))
    monkeypatch.setattr(graph, "search_openfda", source_result("openFDA labels"))
    monkeypatch.setattr(graph, "search_openfda_faers", failed_faers)
    monkeypatch.setattr(graph, "draft_node", fake_draft)

    async def fake_safety_audit(state):
        return {
            "safety_findings": [],
            "risk_level": "unknown",
            "safety_violations": [],
            "audit_log": state.get("audit_log", []),
        }

    compiled_graph = graph.build_compiled_graph(
        InMemorySaver(), node_overrides={"safety_audit_node": fake_safety_audit}
    )

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

    config["configurable"]["thread_id"] = f"human-review-{decision}"
    interrupted = await compiled_graph.ainvoke(initial_state, config=config)

    assert isinstance(interrupted, dict)
    assert interrupted["canonical_name"] == "metformin"
    assert interrupted["__interrupt__"][0].value["brief"] == "Mock regulatory brief."
    assert [document.source for document in interrupted["documents"]] == [
        "PubMed",
        "ClinicalTrials.gov",
        "openFDA labels",
    ]
    assert any("openFDA FAERS retrieval failed" in entry for entry in interrupted["audit_log"])

    response = {
        "decision": decision,
        "feedback": "Please revise the evidence summary." if decision != "approve" else "",
        "edited_brief": "Human-edited brief." if decision == "edit" else None,
    }
    result = await compiled_graph.ainvoke(Command(resume=response), config=config)

    if decision == "edit":
        assert result["__interrupt__"][0].value["human_round"] == 2
        result = await compiled_graph.ainvoke(
            Command(resume={"decision": "approve"}), config=config
        )

    assert result["human_decision"] == ("approve" if decision == "edit" else decision)
    if decision == "approve":
        assert result["human_approved"] is True
        assert result["review_status"] == "approved"
    elif decision == "edit":
        assert result["human_approved"] is True
        assert result["review_status"] == "approved"
        assert result["edited_brief"] == ""
    else:
        assert result["human_approved"] is False
        assert result["review_status"] == "rejected"

