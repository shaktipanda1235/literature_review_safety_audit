import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app import graph
from app.adapters.rxnorm import RxNormResolution
from app.config import MAX_SEARCH_RETRIES
from app.schemas import GradeResult


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
            matched_name="Glucophage",
        )


class FakeStructuredRewriter:
    def __init__(self, queries):
        self.queries = list(queries)
        self.inputs = []

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.inputs.append(json.loads(messages[-1].content))
        return {"search_query": self.queries.pop(0)}


@pytest.mark.asyncio
async def test_insufficient_evidence_rewrites_exactly_max_retries_then_falls_back(monkeypatch):
    retrieval_queries = []
    fallback_calls = []
    rewriter = FakeStructuredRewriter(
        ["metformin hepatic injury", "metformin liver toxicity Glucophage"]
    )

    async def retrieve(state):
        retrieval_queries.append(state.get("search_query"))
        return {"documents": [], "literature_raw_data": ["No matching evidence."], "audit_log": []}

    async def grade(state):
        return {
            "grade_decision": "fallback",
            "grade_result": GradeResult(
                document_relevance=[],
                evidence_level="weak",
                sufficient=False,
                missing_topics=["hepatic injury"],
            ),
        }

    async def fallback(state):
        fallback_calls.append(state.get("search_attempts"))
        return {"web_fallback_data": ["fallback result"]}

    monkeypatch.setattr(graph, "RxNormClient", lambda: FakeRxNormClient())
    monkeypatch.setattr(graph, "get_llm", lambda role: rewriter)
    compiled_graph = graph.build_compiled_graph(
        InMemorySaver(),
        node_overrides={
            "retrieve_node": retrieve,
            "grade_node": grade,
            "web_fallback_node": fallback,
            "safety_audit_node": lambda state: {
                "safety_violations": [],
                "safety_findings": [],
                "risk_level": "unknown",
            },
        },
    )

    initial_state = {
        "drug_query": "Glucophage",
        "literature_raw_data": [],
        "web_fallback_data": [],
        "safety_violations": [],
        "regulatory_brief": "",
        "grade_decision": "clear",
        "human_approved": False,
        "chat_history": [],
    }
    result = await compiled_graph.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": "query-retry-bound-test"}},
    )

    assert len(retrieval_queries) == MAX_SEARCH_RETRIES + 1
    assert retrieval_queries == [
        "metformin",
        "metformin hepatic injury",
        "metformin liver toxicity Glucophage",
    ]
    assert len(rewriter.inputs) == MAX_SEARCH_RETRIES
    assert all("hepatic injury" in request["missing_topics"] for request in rewriter.inputs)
    assert result["search_attempts"] == MAX_SEARCH_RETRIES
    assert fallback_calls == [MAX_SEARCH_RETRIES]


def test_route_after_grading_respects_retry_bound():
    assert graph.route_after_grading(
        {"grade_decision": "fallback", "search_attempts": MAX_SEARCH_RETRIES - 1}
    ) == "rewrite_node"
    assert graph.route_after_grading(
        {"grade_decision": "fallback", "search_attempts": MAX_SEARCH_RETRIES}
    ) == "web_fallback_node"