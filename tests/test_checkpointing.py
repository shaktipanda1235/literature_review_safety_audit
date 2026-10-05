from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.checkpointing import create_checkpointer


class ReviewState(TypedDict):
    brief: str
    decision: str


def build_review_graph(checkpointer):
    def review_node(state: ReviewState):
        response = interrupt({"brief": state["brief"], "question": "Approve?"})
        return {"decision": response["decision"]}

    workflow = StateGraph(ReviewState)
    workflow.add_node("review", review_node)
    workflow.add_edge(START, "review")
    workflow.add_edge("review", END)
    return workflow.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_sqlite_checkpoint_resumes_after_saver_and_graph_recreation(tmp_path):
    database_path = tmp_path / "durable-checkpoints.sqlite"
    config = {"configurable": {"thread_id": "restart-resume-test"}}

    async with create_checkpointer("sqlite", sqlite_path=database_path) as checkpointer:
        first_graph = build_review_graph(checkpointer)
        paused = await first_graph.ainvoke(
            {"brief": "Saved draft", "decision": ""}, config
        )
        assert paused["__interrupt__"][0].value["brief"] == "Saved draft"

    async with create_checkpointer("sqlite", sqlite_path=database_path) as checkpointer:
        restarted_graph = build_review_graph(checkpointer)
        resumed = await restarted_graph.ainvoke(
            Command(resume={"decision": "approve"}), config
        )

    assert resumed["decision"] == "approve"


@pytest.mark.asyncio
async def test_postgres_backend_requires_connection_url():
    with pytest.raises(ValueError, match="CHECKPOINT_POSTGRES_URL or DATABASE_URL"):
        async with create_checkpointer("postgres", postgres_url=""):
            pass


@pytest.mark.asyncio
async def test_unknown_backend_has_clear_error():
    with pytest.raises(ValueError, match="Unsupported CHECKPOINTER_BACKEND"):
        async with create_checkpointer("memory"):
            pass