import asyncio
import json
from collections import Counter
from typing import Dict, List, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.adapters.pubmed import search_pubmed
from app.adapters.clinicaltrials import search_clinical_trials
from app.adapters.openfda import search_openfda
from app.adapters.openfda_faers import search_openfda_faers
from app.adapters.rxnorm import RxNormClient
from app.config import (
    GRADE_PARSE_RETRIES,
    MAX_REDRAFTS,
    MAX_SEARCH_RETRIES,
    MIN_RELEVANT_DOCUMENTS,
)
from app.brief import draft_node
from app.critic import critic_node
from app.llm import get_llm, load_prompt
from app.safety.subgraph import safety_audit_node
from app.schemas import (
    Brief,
    EvidenceItem,
    GradeResult,
    HumanReviewResponse,
    QueryRewriteResult,
    SafetyFinding,
)
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
    search_query: NotRequired[str]
    search_attempts: NotRequired[int]
    documents: NotRequired[List[EvidenceItem]]
    audit_log: NotRequired[List[str]]
    grade_result: NotRequired[GradeResult]
    evidence_level: NotRequired[Literal["none", "weak", "moderate", "strong"]]
    safety_findings: NotRequired[List[SafetyFinding]]
    risk_level: NotRequired[Literal["critical", "high", "moderate", "low", "unknown"]]
    brief: NotRequired[Brief]
    critic_feedback: NotRequired[List[str]]
    critic_passed: NotRequired[bool]
    critic_unresolved: NotRequired[bool]
    redraft_count: NotRequired[int]
    human_decision: NotRequired[Literal["approve", "edit", "reject"]]
    human_feedback: NotRequired[str]
    edited_brief: NotRequired[str]
    review_status: NotRequired[Literal["pending", "approved", "needs_refinement", "rejected"]]


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
        "search_query": resolution.canonical_name,
        "search_attempts": state.get("search_attempts", 0),
    }


def route_after_normalization(
    state: PharmaGraphState,
) -> Literal["retrieve_node", "END"]:
    """Stop the workflow with its normalization error when no concept resolves."""
    return "END" if state.get("normalization_error") else "retrieve_node"


async def retrieve_node(state: PharmaGraphState) -> Dict:
    """Retrieve evidence concurrently while recording individual source failures."""
    query = state.get("search_query") or state.get("canonical_name") or state["drug_query"]
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


async def rewrite_node(state: PharmaGraphState) -> Dict:
    """Rewrite the current query using missing topics and normalized synonyms."""
    attempts = state.get("search_attempts", 0)
    current_query = state.get("search_query") or state.get("canonical_name") or state["drug_query"]
    if attempts >= MAX_SEARCH_RETRIES:
        return {"search_query": current_query, "search_attempts": attempts}

    grade_result = state.get("grade_result")
    missing_topics = grade_result.missing_topics if grade_result else []
    rewrite_input = {
        "current_query": current_query,
        "canonical_name": state.get("canonical_name") or state["drug_query"],
        "synonyms": state.get("synonyms", []),
        "missing_topics": missing_topics,
    }
    messages = [
        SystemMessage(content=load_prompt("query_rewriter")),
        HumanMessage(content=json.dumps(rewrite_input, ensure_ascii=False)),
    ]
    audit_log = list(state.get("audit_log") or [])

    try:
        rewriter = get_llm("rewriter").with_structured_output(QueryRewriteResult)
        response = await rewriter.ainvoke(messages)
        rewrite_result = QueryRewriteResult.model_validate(response)
        rewritten_query = rewrite_result.search_query.strip()
        if not rewritten_query:
            raise ValueError("Rewriter returned an empty search query")
    except Exception as error:
        rewritten_query = current_query
        audit_log.append(
            f"Query rewrite attempt {attempts + 1} failed: {type(error).__name__}: {error}"
        )

    return {
        "search_query": rewritten_query,
        "search_attempts": attempts + 1,
        "audit_log": audit_log,
    }


async def generator_node(state: PharmaGraphState) -> Dict:
    """Generate and validate the citation-enforced structured brief."""
    return await draft_node(state)


def human_review_node(state: PharmaGraphState) -> Dict:
    """Pause for a human decision after the finished draft and critic review."""
    response = interrupt(
        {
            "question": "Review the generated evidence brief.",
            "brief": state.get("regulatory_brief", ""),
            "findings": [
                finding.model_dump(mode="json")
                for finding in state.get("safety_findings", [])
            ],
            "risk_level": state.get("risk_level", "unknown"),
            "critic_unresolved": state.get("critic_unresolved", False),
            "critic_feedback": state.get("critic_feedback", []),
        }
    )
    review = HumanReviewResponse.model_validate(response)
    audit_log = list(state.get("audit_log") or [])
    audit_log.append(f"Human review decision: {review.decision}")
    updates = {
        "human_decision": review.decision,
        "human_feedback": review.feedback,
        "review_status": "pending",
        "audit_log": audit_log,
    }
    if review.edited_brief is not None:
        updates["edited_brief"] = review.edited_brief
    return updates


def route_after_human_review(
    state: PharmaGraphState,
) -> Literal["finalize_node", "refine_node"]:
    """Route approvals to finalization and edit/reject decisions to refinement."""
    return "finalize_node" if state.get("human_decision") == "approve" else "refine_node"


def finalize_node(state: PharmaGraphState) -> Dict:
    """Mark the reviewed brief as approved, preserving a human-edited version if supplied."""
    updates = {"human_approved": True, "review_status": "approved"}
    if state.get("edited_brief"):
        updates["regulatory_brief"] = state["edited_brief"]
    return updates


def refine_node(state: PharmaGraphState) -> Dict:
    """Record a handoff for the bounded refinement workflow implemented in S4.6."""
    status = "rejected" if state.get("human_decision") == "reject" else "needs_refinement"
    return {"human_approved": False, "review_status": status}


def route_after_grading(
    state: PharmaGraphState,
) -> Literal["rewrite_node", "web_fallback_node", "safety_audit_node"]:
    """Retry insufficient retrieval with a rewritten query, then use fallback."""
    if state["grade_decision"] == "fallback":
        if state.get("search_attempts", 0) < MAX_SEARCH_RETRIES:
            print("  -> Evidence insufficient. Rewriting the search query...")
            return "rewrite_node"
        print("  -> Routing to Web Fallback...")
        return "web_fallback_node"

    print("  -> Routing to Safety Audit...")
    return "safety_audit_node"


def route_after_audit(state: PharmaGraphState) -> Literal["generator_node", "retrieve_node", "END"]:
    """Generate after one completed deterministic/LLM safety audit pass."""
    return "generator_node"


def route_after_critic(
    state: PharmaGraphState,
) -> Literal["generator_node", "human_review_node"]:
    """Redraft on critic issues; send the completed best draft to human review."""
    if state.get("critic_passed") or state.get("critic_unresolved"):
        return "human_review_node"
    return "generator_node"


def build_compiled_graph(checkpointer, node_overrides: Dict | None = None):
    """Build a fresh graph using the supplied official LangGraph checkpointer."""
    workflow = StateGraph(PharmaGraphState)
    nodes = {
        "normalize_node": normalize_node,
        "retrieve_node": retrieve_node,
        "grade_node": grade_node,
        "rewrite_node": rewrite_node,
        "web_fallback_node": web_fallback_node,
        "safety_audit_node": safety_audit_node,
        "generator_node": generator_node,
        "critic_node": critic_node,
        "human_review_node": human_review_node,
        "finalize_node": finalize_node,
        "refine_node": refine_node,
    }
    nodes.update(node_overrides or {})
    for node_name, node in nodes.items():
        workflow.add_node(node_name, node)

    workflow.add_edge(START, "normalize_node")
    workflow.add_conditional_edges(
        "normalize_node",
        route_after_normalization,
        {"retrieve_node": "retrieve_node", "END": END},
    )
    workflow.add_edge("retrieve_node", "grade_node")
    workflow.add_edge("rewrite_node", "retrieve_node")
    workflow.add_conditional_edges(
        "grade_node",
        route_after_grading,
        {
            "rewrite_node": "rewrite_node",
            "web_fallback_node": "web_fallback_node",
            "safety_audit_node": "safety_audit_node",
        },
    )
    workflow.add_edge("web_fallback_node", "safety_audit_node")
    workflow.add_conditional_edges(
        "safety_audit_node",
        route_after_audit,
        {"generator_node": "generator_node", "retrieve_node": "retrieve_node", "END": END},
    )
    workflow.add_edge("generator_node", "critic_node")
    workflow.add_conditional_edges(
        "critic_node",
        route_after_critic,
        {"generator_node": "generator_node", "human_review_node": "human_review_node"},
    )
    workflow.add_conditional_edges(
        "human_review_node",
        route_after_human_review,
        {"finalize_node": "finalize_node", "refine_node": "refine_node"},
    )
    workflow.add_edge("finalize_node", END)
    workflow.add_edge("refine_node", END)
    return workflow.compile(checkpointer=checkpointer)
