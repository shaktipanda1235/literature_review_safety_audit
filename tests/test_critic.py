import pytest
from langgraph.graph import END, START, StateGraph

from app import graph
from app.critic import critic_node
from app.config import MAX_REDRAFTS
from app.graph import route_after_critic
from app.schemas import Brief, CitedClaim, EvidenceItem, SafetyFinding


def make_brief(summary, doc_ids=("doc-1",)):
    return Brief(
        summary=CitedClaim(text=summary, doc_ids=list(doc_ids)),
        evidence_level="moderate",
    )


def critical_finding():
    return SafetyFinding(
        category="hepatic",
        severity="critical",
        summary="The label reports serious hepatotoxicity.",
        supporting_doc_ids=["doc-1"],
        quote="Serious hepatotoxicity has occurred.",
    )


def compile_critic_loop():
    workflow = StateGraph(dict)
    workflow.add_node("draft", graph.generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_edge(START, "draft")
    workflow.add_edge("draft", "critic")
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {"generator_node": "draft", "END": END},
    )
    return workflow.compile()


def critic_loop_state():
    return {
        "brief": make_brief("Initial draft omits the safety finding."),
        "documents": [
            EvidenceItem(
                source="openFDA",
                title="Boxed warning",
                summary="Serious hepatotoxicity has occurred.",
                url="https://api.fda.gov/drug/label.json?id=label-1",
                doc_id="doc-1",
            )
        ],
        "safety_findings": [critical_finding()],
        "redraft_count": 0,
    }


def test_critic_rejects_brief_that_omits_critical_finding_and_routes_for_redraft():
    state = {
        "brief": make_brief("No specific finding is summarized."),
        "documents": [type("Doc", (), {"doc_id": "doc-1"})()],
        "safety_findings": [critical_finding()],
        "redraft_count": 0,
    }

    updates = critic_node(state)
    state.update(updates)

    assert state["critic_passed"] is False
    assert state["critic_unresolved"] is False
    assert state["redraft_count"] == 1
    assert any("critical hepatic finding is not mentioned" in item for item in state["critic_feedback"])
    assert route_after_critic(state) == "generator_node"


def test_critic_rejects_uncited_and_unsupported_safety_assurance():
    state = {
        "brief": Brief(
            summary=CitedClaim(text="The compound is safe and presents no risk."),
            evidence_level="moderate",
        ),
        "documents": [type("Doc", (), {"doc_id": "doc-1"})()],
        "safety_findings": [],
        "redraft_count": 0,
    }

    updates = critic_node(state)

    assert updates["critic_passed"] is False
    assert any("Citation validation failed" in item for item in updates["critic_feedback"])
    assert any("Unsupported safety assurance" in item for item in updates["critic_feedback"])


def test_critic_keeps_best_brief_and_marks_unresolved_at_redraft_bound():
    state = {
        "brief": make_brief("No specific finding is summarized."),
        "documents": [type("Doc", (), {"doc_id": "doc-1"})()],
        "safety_findings": [critical_finding()],
        "redraft_count": MAX_REDRAFTS,
        "audit_log": [],
    }

    updates = critic_node(state)
    state.update(updates)

    assert state["brief"].summary.text == "No specific finding is summarized."
    assert state["redraft_count"] == MAX_REDRAFTS
    assert state["critic_unresolved"] is True
    assert state["audit_log"] == ["Critic issues remain after the redraft limit was reached."]
    assert route_after_critic(state) == "END"


def test_critic_passes_when_critical_finding_is_cited_and_mentioned():
    state = {
        "brief": make_brief("The label reports serious hepatotoxicity.", ("doc-1",)),
        "documents": [type("Doc", (), {"doc_id": "doc-1"})()],
        "safety_findings": [critical_finding()],
        "redraft_count": 1,
    }

    updates = critic_node(state)

    assert updates == {
        "critic_feedback": [],
        "critic_passed": True,
        "critic_unresolved": False,
    }


@pytest.mark.asyncio
async def test_critic_graph_redrafts_omitted_critical_finding(monkeypatch):
    drafts = [
        "Initial draft omits the safety finding.",
        "Serious hepatotoxicity is reported.",
    ]
    draft_feedback = []

    async def fake_draft(state):
        text = drafts.pop(0)
        draft_feedback.append(list(state.get("critic_feedback", [])))
        return {"brief": make_brief(text)}

    monkeypatch.setattr(graph, "draft_node", fake_draft)
    result = await compile_critic_loop().ainvoke(critic_loop_state())

    assert len(draft_feedback) == 2
    assert draft_feedback[0] == []
    assert any("critical hepatic finding" in item for item in draft_feedback[1])
    assert result["brief"].summary.text == "Serious hepatotoxicity is reported."
    assert result["critic_passed"] is True
    assert result["critic_unresolved"] is False
    assert result["redraft_count"] == 1


@pytest.mark.asyncio
async def test_critic_graph_stops_at_redraft_bound_with_best_draft(monkeypatch):
    best_draft = "Best available draft omits the serious finding."
    draft_calls = []

    async def fake_draft(state):
        draft_calls.append(best_draft)
        return {"brief": make_brief(best_draft)}

    monkeypatch.setattr(graph, "draft_node", fake_draft)
    result = await compile_critic_loop().ainvoke(critic_loop_state())

    assert len(draft_calls) == MAX_REDRAFTS + 1
    assert result["brief"].summary.text == best_draft
    assert result["critic_unresolved"] is True
    assert result["redraft_count"] == MAX_REDRAFTS