import pytest

from app import graph


def test_compiled_graph_runs():
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

    result = graph.compiled_pharma_graph.invoke(initial_state, config=config)

    # Expect the state to have been updated and to include the finished marker
    assert isinstance(result, dict)
    assert result.get("__finished") is True or result.get("__interrupted_at") is not None
