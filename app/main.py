import asyncio

from app.checkpointing import create_checkpointer
from app.config import DEFAULT_DRUG_QUERY
from app.graph import PharmaGraphState, build_compiled_graph


async def main():
    """Run the pharma workflow with async support."""
    initial_state: PharmaGraphState = {
        "drug_query": DEFAULT_DRUG_QUERY,
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
    async with create_checkpointer() as checkpointer:
        compiled_pharma_graph = build_compiled_graph(checkpointer)
        result = await compiled_pharma_graph.ainvoke(initial_state, config=config)
    print("\nFinal workflow result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
