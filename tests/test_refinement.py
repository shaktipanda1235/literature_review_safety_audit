import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app import graph
from app.schemas import SafetyFinding


def build_refinement_graph(checkpointer, generated_states):
    async def generate(state):
        generated_states.append(state)
        return {}

    workflow = StateGraph(graph.PharmaGraphState)
    workflow.add_node("human_review_node", graph.human_review_node)
    workflow.add_node("refine_node", graph.refine_node)
    workflow.add_node("generator_node", generate)
    workflow.add_edge(START, "human_review_node")
    workflow.add_conditional_edges(
        "human_review_node",
        graph.route_after_human_review,
        {"finalize_node": END, "refine_node": "refine_node"},
    )
    workflow.add_conditional_edges(
        "refine_node",
        graph.route_after_refine,
        {"generator_node": "generator_node", "END": END},
    )
    workflow.add_edge("generator_node", "human_review_node")
    return workflow.compile(checkpointer=checkpointer)


@pytest.mark.asyncio
async def test_update_state_override_reaches_redraft_and_human_rounds_are_bounded():
    original_finding = SafetyFinding(
        category="hepatic",
        severity="high",
        summary="Original label finding.",
        supporting_doc_ids=["label-1"],
        quote="Original warning text.",
    )
    manual_finding = original_finding.model_copy(
        update={
            "summary": "Manually corrected label finding.",
            "quote": "Corrected warning text.",
        }
    )
    generated_states = []
    config = {"configurable": {"thread_id": "manual-refinement-override"}}
    initial_state = {
        "drug_query": "ketoconazole",
        "literature_raw_data": [],
        "web_fallback_data": [],
        "safety_violations": [],
        "regulatory_brief": "Draft brief.",
        "grade_decision": "clear",
        "human_approved": False,
        "chat_history": [],
        "safety_findings": [original_finding],
        "risk_level": "high",
    }

    async with InMemorySaver() as checkpointer:
        compiled = build_refinement_graph(checkpointer, generated_states)
        paused = await compiled.ainvoke(initial_state, config)
        assert paused["__interrupt__"]

        await compiled.aupdate_state(config, {"safety_findings": [manual_finding]})
        for round_number in range(1, graph.MAX_HUMAN_ROUNDS + 1):
            result = await compiled.ainvoke(
                Command(
                    resume={
                        "decision": "edit",
                        "feedback": f"Apply review note {round_number}.",
                    }
                ),
                config,
            )

    assert len(generated_states) == graph.MAX_HUMAN_ROUNDS - 1
    assert generated_states[0]["human_feedback"] == "Apply review note 1."
    assert generated_states[0]["safety_findings"][0].summary == (
        "Manually corrected label finding."
    )
    assert result["human_rounds"] == graph.MAX_HUMAN_ROUNDS
    assert result["review_status"] == "needs_manual_review"
    assert result["human_approved"] is False