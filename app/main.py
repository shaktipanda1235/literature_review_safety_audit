from app.graph import PharmaGraphState, compiled_pharma_graph


if __name__ == "__main__":
    initial_state: PharmaGraphState = {
        "drug_query": "Compound-X",
        "literature_raw_data": [],
        "web_fallback_data": [],
        "safety_violations": [],
        "regulatory_brief": "",
        "grade_decision": "clear",
        "human_approved": False,
        "chat_history": [],
    }

    config = {"configurable": {"thread_id": "pharma_demo_1"}}

    print("\nStarting workflow...")
    result = compiled_pharma_graph.invoke(initial_state, config=config)
    print("\nFinal workflow result:")
    print(result)
