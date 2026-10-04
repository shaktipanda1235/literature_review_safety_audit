import pytest

from app.brief import (
    BriefValidationError,
    draft_node,
    render_brief_markdown,
    validate_brief,
)
from app.schemas import Brief, CitedClaim, EvidenceItem, NON_MEDICAL_ADVICE_DISCLAIMER


class ScriptedStructuredModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def brief_data(doc_id, *, summary_doc_ids=None, key_doc_ids=None):
    return {
        "summary": {
            "text": "Evidence is limited to the retrieved sources.",
            "doc_ids": summary_doc_ids if summary_doc_ids is not None else [doc_id],
        },
        "evidence_level": "moderate",
        "key_findings": [
            {
                "text": "The label reports hepatic risk.",
                "doc_ids": key_doc_ids if key_doc_ids is not None else [doc_id],
            }
        ],
        "safety_findings_by_severity": {
            "critical": [
                {"text": "Hepatotoxicity is reported.", "doc_ids": [doc_id]}
            ]
        },
        "evidence_gaps": ["Long-term trial evidence was not retrieved."],
    }


def source_document():
    return EvidenceItem(
        source="openFDA",
        title="Boxed warning",
        summary="Serious hepatotoxicity has been reported.",
        url="https://api.fda.gov/drug/label.json?id=label-1",
        doc_id="label-1-boxed-warning",
    )


def test_validate_brief_rejects_uncited_and_unknown_claims():
    uncited = Brief(
        summary=CitedClaim(text="Unsupported claim."),
        evidence_level="weak",
    )
    with pytest.raises(BriefValidationError, match="no document citations"):
        validate_brief(uncited, {"known-doc"})

    unknown = Brief(
        summary=CitedClaim(text="Claim.", doc_ids=["unknown-doc"]),
        evidence_level="moderate",
    )
    with pytest.raises(BriefValidationError, match="unknown doc_ids"):
        validate_brief(unknown, {"known-doc"})


@pytest.mark.asyncio
async def test_draft_node_regenerates_after_uncited_claim_then_renders_sections():
    document = source_document()
    model = ScriptedStructuredModel(
        [
            brief_data(document.doc_id, key_doc_ids=[]),
            brief_data(document.doc_id),
        ]
    )

    result = await draft_node(
        {
            "drug_query": "ketoconazole",
            "canonical_name": "ketoconazole",
            "documents": [document],
            "evidence_level": "moderate",
            "grade_result": None,
            "safety_findings": [],
        },
        model=model,
    )

    assert model.calls == 2
    assert isinstance(result["brief"], Brief)
    assert result["brief"].sources[0].doc_id == document.doc_id
    assert result["brief"].disclaimer == NON_MEDICAL_ADVICE_DISCLAIMER
    for section in (
        "## Summary",
        "## Evidence Level",
        "## Key Findings",
        "## Safety Findings by Severity",
        "## Evidence Gaps",
        "## Sources",
        "## Disclaimer",
    ):
        assert section in result["regulatory_brief"]
    assert f"`{document.doc_id}`" in result["regulatory_brief"]


@pytest.mark.asyncio
async def test_draft_node_without_documents_returns_explicit_evidence_gap():
    result = await draft_node(
        {
            "drug_query": "unknown compound",
            "documents": [],
            "evidence_level": "none",
            "grade_result": None,
            "safety_findings": [],
        }
    )

    assert result["brief"].summary is None
    assert "No evidence documents were retrieved." in result["brief"].evidence_gaps
    assert "## Evidence Gaps" in result["regulatory_brief"]
    assert "not medical advice" in result["regulatory_brief"]