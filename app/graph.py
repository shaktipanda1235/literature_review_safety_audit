from typing import Dict, List, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.adapters.clinicaltrials import search_clinical_trials
from app.adapters.openfda import search_openfda
from app.adapters.pubmed import search_pubmed


class PharmaGraphState(TypedDict):
    """State for the pharmaceutical review workflow."""

    drug_query: str
    literature_raw_data: List[str]
    web_fallback_data: List[str]
    safety_violations: List[str]
    regulatory_brief: str
    grade_decision: Literal["clear", "fallback", "fail"]
    human_approved: bool
    chat_history: List[BaseMessage]
    pubmed_results: List[Dict]
    clinical_trials_results: List[Dict]
    fda_results: List[Dict]
    normalized_evidence: List[Dict]
    quality_score: int


def literature_search_node(state: PharmaGraphState) -> Dict:
    """Fetch primary evidence from PubMed and normalize it."""
    query = state["drug_query"]
    print(f"[Node: Literature Search] Fetching PubMed data for: {query}")

    pubmed_items = search_pubmed(query)
    if not pubmed_items:
        fallback = ["PubMed returned no usable literature for this query."]
        return {
            "literature_raw_data": fallback,
            "pubmed_results": [],
            "normalized_evidence": [],
            "quality_score": 0,
        }

    summaries = [item.summary for item in pubmed_items]
    normalized = [item.to_dict() for item in pubmed_items]

    return {
        "literature_raw_data": summaries,
        "pubmed_results": normalized,
        "normalized_evidence": normalized,
    }


def fact_grader_node(state: PharmaGraphState) -> Dict:
    """Score the usability of the evidence before moving to audit or fallback."""
    data = state["literature_raw_data"]
    evidence = state.get("normalized_evidence", [])
    print(f"[Node: Fact Grader] Evidence count: {len(data)}")

    if not data or not evidence:
        return {"grade_decision": "fail", "quality_score": 0}

    score = min(100, len(evidence) * 25)
    if "no usable literature" in " ".join(data).lower() or score < 25:
        return {"grade_decision": "fallback", "quality_score": score}

    return {"grade_decision": "clear", "quality_score": score}


def web_fallback_node(state: PharmaGraphState) -> Dict:
    """Query external safety sources when primary literature is weak."""
    query = state["drug_query"]
    print(f"[Node: Web Fallback] Triggering ClinicalTrials.gov + openFDA for: {query}")

    ct_items = search_clinical_trials(query)
    fda_items = search_openfda(query)

    ct_data = [item.to_dict() for item in ct_items]
    fda_data = [item.to_dict() for item in fda_items]
    combined = [item["summary"] for item in ct_data + fda_data if item.get("summary")]

    return {
        "web_fallback_data": combined,
        "clinical_trials_results": ct_data,
        "fda_results": fda_data,
        "normalized_evidence": state.get("normalized_evidence", []) + ct_data + fda_data,
    }


def safety_audit_node(state: PharmaGraphState) -> Dict:
    """Audit safety concerns in all normalized evidence sources."""
    print("[Node: Safety Audit] Evaluating safety signals across sources...")

    raw_info = state["literature_raw_data"] + state.get("web_fallback_data", [])
    evidence = list(state.get("normalized_evidence", []))
    violations = []

    for item in evidence:
        text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        if "hepatotoxicity" in text:
            violations.append(
                "CRITICAL WARNING: Detected implicit hepatotoxicity flags at elevated therapeutic intervals."
            )
        if "inconclusive" in text or "insufficient evidence" in text:
            violations.append("WARNING: Trial evidence is inconclusive and may require further confirmation.")

    for text in raw_info:
        lowered = text.lower()
        if "hepatotoxicity" in lowered:
            violations.append(
                "CRITICAL WARNING: Detected implicit hepatotoxicity flags at elevated therapeutic intervals."
            )
        if "inconclusive" in lowered:
            violations.append("WARNING: Trial evidence is inconclusive and may require further confirmation.")

    unique = []
    seen = set()
    for item in violations:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return {"safety_violations": unique}


def generator_node(state: PharmaGraphState) -> Dict:
    """Generate the final regulatory brief from the collected evidence."""
    print("[Node: Brief Generator] Synthesis of regulatory compliance files underway...")

    violations = state.get("safety_violations", [])
    quality = state.get("quality_score", 0)
    brief = "--- REGULATORY BRIEF FOR COMPOUND ---\n"
    brief += f"Query: {state['drug_query']}\n"
    brief += f"Evidence Quality Score: {quality}/100\n"
    brief += f"Primary Violations Flagged: {len(violations)}\n"

    if violations:
        brief += f"Flags Raised: {violations[0]}\n"

    brief += "Status: Pending Human-in-the-Loop Verification."
    return {"regulatory_brief": brief}


def route_after_grading(state: PharmaGraphState) -> Literal["web_fallback_node", "safety_audit_node"]:
    """Choose fallback or direct safety review based on evidence quality."""
    if state.get("grade_decision") == "fallback":
        print("  -> Routing to Web Fallback...")
        return "web_fallback_node"

    print("  -> Routing to Safety Audit...")
    return "safety_audit_node"


def route_after_audit(state: PharmaGraphState) -> Literal["generator_node", "literature_search_node", "END"]:
    """Branch after safety review."""
    if state.get("safety_violations"):
        print("  -> Violations detected. Re-routing for review and possible re-draft.")
        return "literature_search_node"

    print("  -> No critical violations found. Generating brief.")
    return "generator_node"


workflow = StateGraph(PharmaGraphState)

workflow.add_node("literature_search_node", literature_search_node)
workflow.add_node("fact_grader_node", fact_grader_node)
workflow.add_node("web_fallback_node", web_fallback_node)
workflow.add_node("safety_audit_node", safety_audit_node)
workflow.add_node("generator_node", generator_node)

workflow.add_edge(START, "literature_search_node")
workflow.add_edge("literature_search_node", "fact_grader_node")

workflow.add_conditional_edges(
    "fact_grader_node",
    route_after_grading,
    {
        "web_fallback_node": "web_fallback_node",
        "safety_audit_node": "safety_audit_node",
    },
)

workflow.add_edge("web_fallback_node", "safety_audit_node")

workflow.add_conditional_edges(
    "safety_audit_node",
    route_after_audit,
    {
        "generator_node": "generator_node",
        "literature_search_node": "literature_search_node",
        "END": END,
    },
)

workflow.add_edge("generator_node", END)

memory = MemorySaver()
compiled_pharma_graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["generator_node"],
)

print("[Graph Setup] LangGraph Workflow Compiled Successfully with persistent MemorySaver.")
