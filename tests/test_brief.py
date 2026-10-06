import json

import pytest

from app.brief import (
    BriefValidationError,
    INSUFFICIENT_EVIDENCE_HEADLINE,
    compute_evidence_level,
    draft_node,
    render_brief_markdown,
    validate_brief,
)
from app.schemas import (
    Brief,
    CitedClaim,
    DocumentRelevance,
    EvidenceItem,
    GradeResult,
    NON_MEDICAL_ADVICE_DISCLAIMER,
)


class ScriptedStructuredModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.message_batches = []

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        self.message_batches.append(messages)
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


def make_grade(documents, level, relevant_count):
    return GradeResult(
        document_relevance=[
            DocumentRelevance(
                doc_id=document.doc_id,
                relevance="relevant" if index < relevant_count else "partial",
            )
            for index, document in enumerate(documents)
        ],
        evidence_level=level,
        sufficient=relevant_count >= 2,
        missing_topics=[],
    )


def test_compute_evidence_level_combines_volume_diversity_trial_phase_and_label():
    assert compute_evidence_level([], None) == "none"

    sparse_documents = [source_document()]
    assert compute_evidence_level(
        sparse_documents, make_grade(sparse_documents, "strong", relevant_count=1)
    ) == "weak"

    moderate_documents = [
        EvidenceItem(
            source="PubMed",
            title="Study",
            summary="Relevant safety evidence.",
            url="https://pubmed.ncbi.nlm.nih.gov/1/",
            doc_id="pubmed-1",
        ),
        EvidenceItem(
            source="ClinicalTrials.gov",
            title="Phase 2 trial",
            summary="Relevant trial evidence.",
            url="https://clinicaltrials.gov/study/NCT1",
            raw_payload={"phase": ["PHASE2"]},
            doc_id="trial-1",
        ),
    ]
    assert compute_evidence_level(
        moderate_documents, make_grade(moderate_documents, "weak", relevant_count=2)
    ) == "moderate"

    strong_documents = [
        EvidenceItem(
            source=source,
            title=f"Evidence {index}",
            summary="Relevant evidence.",
            url=f"https://example.org/{index}",
            raw_payload=payload,
            doc_id=f"doc-{index}",
        )
        for index, (source, payload) in enumerate(
            [
                ("PubMed", {}),
                ("PubMed", {}),
                ("ClinicalTrials.gov", {"phase": ["PHASE3"]}),
                ("ClinicalTrials.gov", {"phase": ["PHASE2"]}),
                ("openFDA", {"section": "boxed_warning"}),
                ("openFDA", {"section": "adverse_reactions"}),
                ("openFDA FAERS", {}),
                ("PubMed", {}),
            ]
        )
    ]
    assert compute_evidence_level(
        strong_documents, make_grade(strong_documents, "strong", relevant_count=8)
    ) == "strong"


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
    assert result["brief"].evidence_level == "none"
    assert "## Evidence Gaps" in result["regulatory_brief"]
    assert INSUFFICIENT_EVIDENCE_HEADLINE in result["regulatory_brief"]
    assert "not medical advice" in result["regulatory_brief"]


@pytest.mark.asyncio
async def test_draft_prompt_includes_human_feedback_and_edited_brief():
    document = source_document()
    model = ScriptedStructuredModel([brief_data(document.doc_id)])

    await draft_node(
        {
            "drug_query": "ketoconazole",
            "documents": [document],
            "grade_result": None,
            "safety_findings": [],
            "human_feedback": "Clarify the hepatic warning.",
            "edited_brief": "Please distinguish warnings from observed incidence.",
            "human_rounds": 1,
        },
        model=model,
    )

    prompt_payload = json.loads(model.message_batches[0][-1].content)
    assert prompt_payload["human_feedback"] == "Clarify the hepatic warning."
    assert prompt_payload["edited_brief"] == (
        "Please distinguish warnings from observed incidence."
    )
    assert prompt_payload["human_rounds"] == 1


@pytest.mark.asyncio
async def test_weak_evidence_overrides_reassuring_model_headline():
    document = source_document()
    model = ScriptedStructuredModel(
        [
            {
                "summary": {
                    "text": "No violations were found, so the compound is safe.",
                    "doc_ids": [document.doc_id],
                },
                "evidence_level": "strong",
                "key_findings": [],
                "safety_findings_by_severity": {},
                "evidence_gaps": [],
            }
        ]
    )

    result = await draft_node(
        {
            "drug_query": "ketoconazole",
            "documents": [document],
            "grade_result": make_grade([document], "strong", relevant_count=1),
            "safety_findings": [],
        },
        model=model,
    )

    assert result["brief"].evidence_level == "weak"
    assert result["brief"].summary.text == INSUFFICIENT_EVIDENCE_HEADLINE
    assert "No violations were found" not in result["regulatory_brief"]