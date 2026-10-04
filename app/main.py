import asyncio

from app.graph import PharmaGraphState, compiled_pharma_graph


async def main():
    """Run the pharma workflow with async support."""
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
    result = await compiled_pharma_graph.ainvoke(initial_state, config=config)
    print("\nFinal workflow result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
