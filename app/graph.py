from typing import Dict, List, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.adapters.pubmed import search_pubmed
from app.adapters.clinicaltrials import search_clinical_trials
from app.adapters.openfda import search_openfda
from app.sources.base import HttpClient


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


async def literature_search_node(state: PharmaGraphState) -> Dict:
    """Fetch primary literature from PubMed, ClinicalTrials.gov, and openFDA."""
    query = state["drug_query"]
    print(f"[Node: Literature Search] Fetching clinical data for: {query}")

    # Create shared HTTP client for all adapters
    client = HttpClient()
    
    try:
        # Query all three sources in parallel
        pubmed_results = await search_pubmed(query, max_results=5, client=client)
        ct_results = await search_clinical_trials(query, max_results=5, client=client)
        fda_results = await search_openfda(query, max_results=5, client=client)
        
        # Combine results
        all_results = pubmed_results + ct_results + fda_results
        
        # Format as strings for state (preserving details)
        data = [
            f"{item.source}: {item.title} - {item.summary}"
            for item in all_results
        ]
        
        if not data:
            data = ["No results found from external sources for: " + query]
            
        print(f"  Retrieved {len(data)} items from external sources")
        return {"literature_raw_data": data}
    except Exception as e:
        print(f"  Error querying external sources: {e}")
        return {"literature_raw_data": [f"Error fetching data: {str(e)}"]}
    finally:
        await client.close()


def fact_grader_node(state: PharmaGraphState) -> Dict:
    """Decide whether the literature is usable or needs fallback."""
    data = state["literature_raw_data"]
    print(f"[Node: Fact Grader] Grading data richness: {data}")

    if not data:
        return {"grade_decision": "fail"}

    first = data[0].lower()
    if "no localized proprietary data" in first or "inconclusive" in first:
        return {"grade_decision": "fallback"}

    return {"grade_decision": "clear"}


def web_fallback_node(state: PharmaGraphState) -> Dict:
    """Query external sources when internal data is insufficient."""
    query = state["drug_query"]
    print(f"[Node: Web Fallback] Triggering external APIs for: {query}")

    external_data = [
        "Web scraping source: Phase I data suggests low affinity but alternative metabolic clearance pathways."
    ]
    return {"web_fallback_data": external_data}


def safety_audit_node(state: PharmaGraphState) -> Dict:
    """Audit safety concerns in both internal and external data."""
    print("[Node: Safety Audit] Evaluating structural contraindications...")

    raw_info = state["literature_raw_data"] + state.get("web_fallback_data", [])
    violations = []

    for text in raw_info:
        lowered = text.lower()
        if "hepatotoxicity" in lowered:
            violations.append(
                "CRITICAL WARNING: Detected implicit hepatotoxicity flags at elevated therapeutic intervals."
            )
        if "inconclusive" in lowered:
            violations.append("WARNING: Trial evidence is inconclusive and may require further confirmation.")

    return {"safety_violations": violations}


def generator_node(state: PharmaGraphState) -> Dict:
    """Generate the regulatory brief from the collected evidence."""
    print("[Node: Brief Generator] Synthesis of regulatory compliance files underway...")

    violations = state.get("safety_violations", [])
    brief = "--- REGULATORY BRIEF FOR COMPOUND ---\n"
    brief += f"Query: {state['drug_query']}\n"
    brief += f"Primary Violations Flagged: {len(violations)}\n"

    if violations:
        brief += f"Flags Raised: {violations[0]}\n"

    brief += "Status: Pending Human-in-the-Loop Verification."
    return {"regulatory_brief": brief}


def route_after_grading(state: PharmaGraphState) -> Literal["web_fallback_node", "safety_audit_node"]:
    """Choose fallback or direct safety review based on data quality."""
    if state["grade_decision"] == "fallback":
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
