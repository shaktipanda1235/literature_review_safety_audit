from __future__ import annotations

from typing import Dict, List, Literal, NotRequired, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.retrieval import EvidenceRetriever
from app.safety.extractor import extract_safety_findings
from app.safety.rules import findings_from_evidence, load_safety_watchlist
from app.schemas import EvidenceItem, SafetyFinding


SAFETY_CATEGORY_QUERIES = {
    "hepatic": "hepatotoxicity liver injury hepatic failure",
    "cardiac": "cardiac arrest cardiotoxicity QT prolongation torsades",
    "renal": "renal failure nephrotoxicity kidney injury",
    "interactions": "drug-drug interactions contraindications coadministration",
    "special_populations": "pregnancy pediatric hepatic impairment renal impairment",
}
CATEGORY_NODE_NAMES = [f"{category}_check" for category in SAFETY_CATEGORY_QUERIES]
GENERAL_LABEL_RULE_CATEGORIES = {"boxed_warning", "contraindication"}


class SafetyAuditState(TypedDict):
    documents: List[EvidenceItem]
    category_chunks: Dict[str, List[Document]]
    watchlist: Dict[str, List[str]]
    hepatic_findings: NotRequired[List[SafetyFinding]]
    cardiac_findings: NotRequired[List[SafetyFinding]]
    renal_findings: NotRequired[List[SafetyFinding]]
    interactions_findings: NotRequired[List[SafetyFinding]]
    special_populations_findings: NotRequired[List[SafetyFinding]]
    hepatic_audit_log: NotRequired[List[str]]
    cardiac_audit_log: NotRequired[List[str]]
    renal_audit_log: NotRequired[List[str]]
    interactions_audit_log: NotRequired[List[str]]
    special_populations_audit_log: NotRequired[List[str]]
    safety_findings: NotRequired[List[SafetyFinding]]
    risk_level: NotRequired[Literal["critical", "high", "moderate", "low", "unknown"]]
    audit_log: NotRequired[List[str]]


def _category_node(category: str):
    findings_key = f"{category}_findings"
    audit_key = f"{category}_audit_log"

    async def run_category(state: SafetyAuditState) -> Dict:
        chunks = state.get("category_chunks", {}).get(category, [])
        retrieved_ids = {
            chunk.metadata.get("doc_id")
            for chunk in chunks
            if isinstance(chunk.metadata.get("doc_id"), str)
        }
        category_documents = [
            document
            for document in state.get("documents", [])
            if document.doc_id in retrieved_ids
        ]

        rule_findings = [
            finding
            for finding in findings_from_evidence(
                category_documents, state.get("watchlist", {})
            )
            if finding.category == category
            or finding.category in GENERAL_LABEL_RULE_CATEGORIES
        ]
        findings = list(rule_findings)
        audit_entries: List[str] = []

        if chunks:
            try:
                llm_findings = await extract_safety_findings(
                    category,
                    chunks,
                    category_documents,
                )
                findings.extend(llm_findings)
            except Exception as error:
                audit_entries.append(
                    f"{category} LLM safety extraction failed: "
                    f"{type(error).__name__}: {error}"
                )

        if not chunks and not findings:
            findings.append(
                SafetyFinding(
                    category=category,
                    severity="no_evidence",
                    summary=f"No evidence was retrieved for the {category} safety category.",
                )
            )

        return {findings_key: findings, audit_key: audit_entries}

    return run_category


def _merge_findings(state: SafetyAuditState) -> Dict:
    findings: List[SafetyFinding] = []
    audit_log: List[str] = []
    seen = set()

    for category in SAFETY_CATEGORY_QUERIES:
        audit_log.extend(state.get(f"{category}_audit_log", []))
        for finding in state.get(f"{category}_findings", []):
            identity = (
                finding.category,
                finding.severity,
                tuple(sorted(finding.supporting_doc_ids)),
                " ".join((finding.quote or finding.summary).split()).casefold(),
            )
            if identity not in seen:
                seen.add(identity)
                findings.append(finding)

    risk_priority = {"critical": 4, "high": 3, "moderate": 2, "info": 1}
    risk_findings = [finding for finding in findings if finding.severity != "no_evidence"]
    risk_level = (
        max(risk_findings, key=lambda finding: risk_priority[finding.severity]).severity
        if risk_findings
        else "unknown"
    )

    return {
        "safety_findings": findings,
        "risk_level": risk_level,
        "audit_log": audit_log,
    }


def build_safety_audit_subgraph():
    """Create a five-branch safety graph with a shared merge node."""
    workflow = StateGraph(SafetyAuditState)
    for category, node_name in zip(SAFETY_CATEGORY_QUERIES, CATEGORY_NODE_NAMES):
        workflow.add_node(node_name, _category_node(category))
        workflow.add_edge(START, node_name)

    workflow.add_node("merge_findings", _merge_findings)
    workflow.add_edge(CATEGORY_NODE_NAMES, "merge_findings")
    workflow.add_edge("merge_findings", END)
    return workflow.compile()


compiled_safety_audit_subgraph = build_safety_audit_subgraph()


async def safety_audit_node(state: dict) -> Dict:
    """Prepare per-category retrieval context and invoke the safety subgraph."""
    documents = state.get("documents") or []
    if documents:
        with EvidenceRetriever() as retriever:
            retriever.add_evidence(documents)
            category_chunks = {
                category: retriever.top_k(query, k=5)
                for category, query in SAFETY_CATEGORY_QUERIES.items()
            }
    else:
        category_chunks = {category: [] for category in SAFETY_CATEGORY_QUERIES}

    subgraph_result = await compiled_safety_audit_subgraph.ainvoke(
        {
            "documents": documents,
            "category_chunks": category_chunks,
            "watchlist": load_safety_watchlist(),
        }
    )
    findings = subgraph_result.get("safety_findings", [])
    audit_log = list(state.get("audit_log") or [])
    audit_log.extend(subgraph_result.get("audit_log", []))
    safety_violations = [
        f"{finding.severity.upper()}: {finding.summary}"
        for finding in findings
        if finding.severity != "no_evidence"
    ]

    return {
        "safety_findings": findings,
        "risk_level": subgraph_result.get("risk_level", "unknown"),
        "audit_log": audit_log,
        "safety_violations": safety_violations,
    }