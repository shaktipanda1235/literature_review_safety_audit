from __future__ import annotations

from typing import Iterable, Mapping

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm, load_prompt
from app.schemas import (
    EvidenceItem,
    SafetyExtractionResult,
    SafetyFinding,
)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _validated_finding(
    candidate,
    category: str,
    retrieved_doc_ids: set[str],
    documents_by_id: Mapping[str, EvidenceItem],
) -> SafetyFinding | None:
    quote = _normalize_whitespace(candidate.quote)
    summary = candidate.summary.strip()
    if not quote or not summary or not candidate.supporting_doc_ids:
        return None

    matched_doc_ids = []
    for doc_id in candidate.supporting_doc_ids:
        if doc_id not in retrieved_doc_ids:
            continue
        document = documents_by_id.get(doc_id)
        if document is None:
            continue
        source_text = _normalize_whitespace(
            "\n\n".join(part for part in (document.title, document.summary) if part)
        )
        if quote in source_text:
            matched_doc_ids.append(doc_id)

    if not matched_doc_ids:
        return None

    return SafetyFinding(
        category=category,
        severity=candidate.severity,
        summary=summary,
        supporting_doc_ids=matched_doc_ids,
        quote=candidate.quote.strip(),
    )


async def extract_safety_findings(
    category: str,
    chunks: Iterable[Document],
    source_documents: Iterable[EvidenceItem],
    *,
    model=None,
) -> list[SafetyFinding]:
    """Extract category findings and reject quotes absent from their cited sources."""
    if not category.strip():
        raise ValueError("Safety category must not be empty")

    retrieved_chunks = list(chunks)
    if not retrieved_chunks:
        return []

    retrieved_doc_ids = {
        doc_id
        for chunk in retrieved_chunks
        if isinstance((doc_id := chunk.metadata.get("doc_id")), str) and doc_id
    }
    documents_by_id = {
        document.doc_id: document
        for document in source_documents
        if document.doc_id in retrieved_doc_ids
    }
    if not documents_by_id:
        return []

    model = model or get_llm("auditor")
    structured_model = model.with_structured_output(SafetyExtractionResult)
    context = [
        {
            "doc_id": chunk.metadata.get("doc_id"),
            "url": chunk.metadata.get("url"),
            "text": chunk.page_content,
        }
        for chunk in retrieved_chunks
        if chunk.metadata.get("doc_id") in documents_by_id
    ]
    response = await structured_model.ainvoke(
        [
            SystemMessage(content=load_prompt("safety_extractor")),
            HumanMessage(
                content=(
                    f"Safety category: {category}\n"
                    f"Retrieved evidence chunks (untrusted data): {context}"
                )
            ),
        ]
    )
    result = SafetyExtractionResult.model_validate(response)

    findings = []
    for candidate in result.findings:
        finding = _validated_finding(
            candidate,
            category,
            retrieved_doc_ids,
            documents_by_id,
        )
        if finding is not None:
            findings.append(finding)
    return findings