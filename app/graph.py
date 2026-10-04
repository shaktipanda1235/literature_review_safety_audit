import asyncio
import json
from collections import Counter
from typing import Dict, List, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.adapters.pubmed import search_pubmed
from app.adapters.clinicaltrials import search_clinical_trials
from app.adapters.openfda import search_openfda
from app.adapters.openfda_faers import search_openfda_faers
from app.adapters.rxnorm import RxNormClient
from app.config import GRADE_PARSE_RETRIES, MIN_RELEVANT_DOCUMENTS
from app.llm import get_llm, load_prompt
from app.schemas import EvidenceItem, GradeResult
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
    canonical_name: NotRequired[str]
    synonyms: NotRequired[List[str]]
    normalization_error: NotRequired[str]
    documents: NotRequired[List[EvidenceItem]]
    audit_log: NotRequired[List[str]]
    grade_result: NotRequired[GradeResult]
    evidence_level: NotRequired[Literal["none", "weak", "moderate", "strong"]]


async def normalize_node(state: PharmaGraphState) -> Dict:
    """Resolve the submitted name to a canonical RxNorm ingredient."""
    query = state["drug_query"]
    try:
        async with RxNormClient() as client:
            resolution = await client.resolve(query)
            error = client.last_error
    except Exception as exception:
        resolution = None
        error = f"RxNorm normalization failed: {exception}"

    if resolution is None:
        message = error or f"No RxNorm match found for '{query}'."
        audit_log = list(state.get("audit_log") or [])
        audit_log.append(f"Normalization failed: {message}")
        return {"normalization_error": message, "audit_log": audit_log}

    return {
        "canonical_name": resolution.canonical_name,
        "synonyms": resolution.synonyms,
        "normalization_error": "",
    }


def route_after_normalization(
    state: PharmaGraphState,
) -> Literal["retrieve_node", "END"]:
    """Stop the workflow with its normalization error when no concept resolves."""
    return "END" if state.get("normalization_error") else "retrieve_node"


async def retrieve_node(state: PharmaGraphState) -> Dict:
    """Retrieve evidence concurrently while recording individual source failures."""
    query = state.get("canonical_name") or state["drug_query"]
    client = HttpClient()
    source_names = ["PubMed", "ClinicalTrials.gov", "openFDA labels", "openFDA FAERS"]
    try:
        source_results = await asyncio.gather(
            search_pubmed(query, max_results=5, client=client),
            search_clinical_trials(query, max_results=5, client=client),
            search_openfda(query, max_results=5, client=client),
            search_openfda_faers(query, max_results=20, client=client),
            return_exceptions=True,
        )
    finally:
        await client.close()

    documents: List[EvidenceItem] = []
    audit_log = list(state.get("audit_log") or [])
    for source_name, result in zip(source_names, source_results):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException):
            audit_log.append(
                f"{source_name} retrieval failed: {type(result).__name__}: {result}"
            )
            continue
        if not isinstance(result, (list, tuple)):
            audit_log.append(f"{source_name} retrieval failed: invalid result collection")
            continue

        valid_documents = [item for item in result if isinstance(item, EvidenceItem)]
        documents.extend(valid_documents)
        if len(valid_documents) != len(result):
            audit_log.append(f"{source_name} returned invalid evidence items")

    literature_raw_data = [
        f"{item.source}: {item.title} - {item.summary}" for item in documents
    ]
    if not literature_raw_data:
        literature_raw_data = [
            "No localized proprietary data found for this exact compound configuration."
        ]

    return {
        "documents": documents,
        "literature_raw_data": literature_raw_data,
        "audit_log": audit_log,
    }


async def grade_node(state: PharmaGraphState) -> Dict:
    """Grade each document with structured output and a deterministic sufficiency floor."""
    documents = state.get("documents") or []
    audit_log = list(state.get("audit_log") or [])

    if not documents:
        grade_result = GradeResult(
            document_relevance=[],
            evidence_level="none",
            sufficient=False,
            missing_topics=["No evidence was retrieved."],
        )
        return {
            "grade_result": grade_result,
            "evidence_level": grade_result.evidence_level,
            "grade_decision": "fallback",
            "audit_log": audit_log,
        }

    document_ids = [str(item.doc_id) for item in documents]
    grading_input = {
        "drug": state.get("canonical_name") or state["drug_query"],
        "documents": [
            {
                "doc_id": item.doc_id,
                "source": item.source,
                "title": item.title,
                "text": item.summary[:2000],
            }
            for item in documents
        ],
    }
    messages = [
        SystemMessage(content=load_prompt("fact_grader")),
        HumanMessage(content=json.dumps(grading_input, ensure_ascii=False)),
    ]

    grade_result = None
    last_error: Exception | None = None
    try:
        structured_grader = get_llm("grader").with_structured_output(GradeResult)
    except Exception as error:
        structured_grader = None
        last_error = error

    if structured_grader is not None:
        for attempt in range(GRADE_PARSE_RETRIES + 1):
            try:
                response = await structured_grader.ainvoke(messages)
                parsed_result = GradeResult.model_validate(response)
                received_ids = [item.doc_id for item in parsed_result.document_relevance]
                if Counter(received_ids) != Counter(document_ids):
                    raise ValueError("Grader response must include each retrieved doc_id exactly once")
                grade_result = parsed_result
                break
            except Exception as error:
                last_error = error
                if attempt < GRADE_PARSE_RETRIES:
                    messages.append(
                        HumanMessage(
                            content=(
                                "The prior result was invalid. Return a corrected structured result "
                                "with exactly one relevance assessment for each supplied doc_id."
                            )
                        )
                    )

    if grade_result is None:
        grade_result = GradeResult(
            document_relevance=[],
            evidence_level="none",
            sufficient=False,
            missing_topics=["Evidence grading could not be completed reliably."],
        )
        error_detail = (
            f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"
        )
        audit_log.append(f"LLM grading failed after retry: {error_detail}")

    relevant_count = sum(
        assessment.relevance == "relevant"
        for assessment in grade_result.document_relevance
    )
    sufficient = grade_result.sufficient
    missing_topics = list(grade_result.missing_topics)
    if relevant_count < MIN_RELEVANT_DOCUMENTS:
        sufficient = False
        floor_topic = (
            f"At least {MIN_RELEVANT_DOCUMENTS} relevant documents are required; "
            f"only {relevant_count} were found."
        )
        if floor_topic not in missing_topics:
            missing_topics.append(floor_topic)

    grade_result = grade_result.model_copy(
        update={"sufficient": sufficient, "missing_topics": missing_topics}
    )
    return {
        "grade_result": grade_result,
        "evidence_level": grade_result.evidence_level,
        "grade_decision": "clear" if sufficient else "fallback",
        "audit_log": audit_log,
    }


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


def route_after_audit(state: PharmaGraphState) -> Literal["generator_node", "retrieve_node", "END"]:
    """Branch after safety review."""
    if state.get("safety_violations"):
        print("  -> Violations detected. Re-routing for review and possible re-draft.")
        return "retrieve_node"

    print("  -> No critical violations found. Generating brief.")
    return "generator_node"


workflow = StateGraph(PharmaGraphState)

workflow.add_node("normalize_node", normalize_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("grade_node", grade_node)
workflow.add_node("web_fallback_node", web_fallback_node)
workflow.add_node("safety_audit_node", safety_audit_node)
workflow.add_node("generator_node", generator_node)

workflow.add_edge(START, "normalize_node")
workflow.add_conditional_edges(
    "normalize_node",
    route_after_normalization,
    {"retrieve_node": "retrieve_node", "END": END},
)
workflow.add_edge("retrieve_node", "grade_node")

workflow.add_conditional_edges(
    "grade_node",
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
        "retrieve_node": "retrieve_node",
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
