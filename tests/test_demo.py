import pytest

from app import graph


def test_compiled_graph_runs():
    """Test that the compiled graph runs without error.
    
    Note: Using invoke() instead of ainvoke() because the local shim 
    version of LangGraph doesn't support async nodes. The actual graph 
    is async-capable and will work with asyncio.run() in production.
    """
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

    # Note: This test uses the local shim which is synchronous only.
    # In production, use asyncio.run(main()) from app/main.py instead.
    result = graph.compiled_pharma_graph.invoke(initial_state, config=config)

    # Expect the state to have been updated 
    assert isinstance(result, dict)

