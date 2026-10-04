import asyncio

import pytest
from langchain_core.documents import Document

from app.safety import subgraph
from app.schemas import EvidenceItem, SafetyFinding


@pytest.mark.asyncio
async def test_five_category_branches_run_concurrently_and_merge_risk(monkeypatch):
    categories = list(subgraph.SAFETY_CATEGORY_QUERIES)
    source = EvidenceItem(
        source="PubMed",
        title="General safety report",
        summary="A neutral clinical observation.",
        url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        doc_id="pubmed-12345",
    )
    chunks_by_category = {
        category: [
            Document(
                page_content=source.summary,
                metadata={"doc_id": source.doc_id, "url": source.url},
            )
        ]
        for category in categories
    }
    started_categories = set()
    all_started = asyncio.Event()

    async def fake_extract(category, chunks, source_documents):
        started_categories.add(category)
        if len(started_categories) == len(categories):
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=1)
        return [
            SafetyFinding(
                category=category,
                severity="moderate",
                summary=f"{category} fixture finding",
                supporting_doc_ids=[source.doc_id],
                quote="A neutral clinical observation.",
            )
        ]

    monkeypatch.setattr(subgraph, "extract_safety_findings", fake_extract)

    result = await subgraph.compiled_safety_audit_subgraph.ainvoke(
        {
            "documents": [source],
            "category_chunks": chunks_by_category,
            "watchlist": {},
        }
    )

    assert started_categories == set(categories)
    assert {finding.category for finding in result["safety_findings"]} == set(categories)
    assert result["risk_level"] == "moderate"


@pytest.mark.asyncio
async def test_empty_categories_emit_explicit_no_evidence_findings():
    result = await subgraph.compiled_safety_audit_subgraph.ainvoke(
        {
            "documents": [],
            "category_chunks": {category: [] for category in subgraph.SAFETY_CATEGORY_QUERIES},
            "watchlist": {},
        }
    )

    findings = result["safety_findings"]
    assert len(findings) == len(subgraph.SAFETY_CATEGORY_QUERIES)
    assert {finding.severity for finding in findings} == {"no_evidence"}
    assert result["risk_level"] == "unknown"