import pytest
from langchain_core.documents import Document

from app.safety.extractor import extract_safety_findings
from app.schemas import EvidenceItem


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        return self.response


def source_document():
    return EvidenceItem(
        source="openFDA",
        title="Warnings and precautions",
        summary="Serious hepatotoxicity has occurred with oral treatment.",
        url="https://api.fda.gov/drug/label.json?id=label-1",
        doc_id="label-1-warnings",
    )


def evidence_chunk(document):
    return Document(
        page_content=f"{document.title}\n\n{document.summary}",
        metadata={"doc_id": document.doc_id, "url": document.url},
    )


@pytest.mark.asyncio
async def test_valid_verbatim_quote_returns_cited_finding():
    document = source_document()
    model = FakeStructuredModel(
        {
            "findings": [
                {
                    "category": "hepatic",
                    "severity": "high",
                    "summary": "The label reports hepatotoxicity.",
                    "supporting_doc_ids": [document.doc_id],
                    "quote": "Serious hepatotoxicity has occurred with oral treatment.",
                }
            ]
        }
    )

    findings = await extract_safety_findings(
        "hepatic", [evidence_chunk(document)], [document], model=model
    )

    assert len(findings) == 1
    assert findings[0].category == "hepatic"
    assert findings[0].supporting_doc_ids == [document.doc_id]
    assert findings[0].quote == "Serious hepatotoxicity has occurred with oral treatment."


@pytest.mark.asyncio
async def test_hallucinated_quote_is_rejected():
    document = source_document()
    model = FakeStructuredModel(
        {
            "findings": [
                {
                    "category": "hepatic",
                    "severity": "critical",
                    "summary": "The label says the drug is completely safe.",
                    "supporting_doc_ids": [document.doc_id],
                    "quote": "The drug is completely safe.",
                }
            ]
        }
    )

    findings = await extract_safety_findings(
        "hepatic", [evidence_chunk(document)], [document], model=model
    )

    assert findings == []


@pytest.mark.asyncio
async def test_quote_validation_normalizes_whitespace_and_rejects_unknown_doc_ids():
    document = source_document()
    model = FakeStructuredModel(
        {
            "findings": [
                {
                    "category": "hepatic",
                    "severity": "high",
                    "summary": "Source quote is supported.",
                    "supporting_doc_ids": ["not-retrieved", document.doc_id],
                    "quote": "Serious hepatotoxicity\n has occurred with oral\ttreatment.",
                }
            ]
        }
    )

    findings = await extract_safety_findings(
        "hepatic", [evidence_chunk(document)], [document], model=model
    )

    assert len(findings) == 1
    assert findings[0].supporting_doc_ids == [document.doc_id]