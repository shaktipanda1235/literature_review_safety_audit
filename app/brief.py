from __future__ import annotations

import json
from typing import Iterable, Mapping

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm, load_prompt
from app.schemas import (
    NON_MEDICAL_ADVICE_DISCLAIMER,
    Brief,
    BriefSource,
    CitedClaim,
    EvidenceItem,
    GradeResult,
    SafetyFinding,
)
from app.config import MIN_RELEVANT_DOCUMENTS


MAX_BRIEF_REGENERATION_ATTEMPTS = 1
SAFETY_SEVERITY_ORDER = ("critical", "high", "moderate", "info", "no_evidence")
INSUFFICIENT_EVIDENCE_HEADLINE = "Insufficient evidence to assess safety."


class BriefValidationError(ValueError):
    """Raised when generated claims fail citation or content validation."""


def compute_evidence_level(
    documents: Iterable[EvidenceItem], grade: GradeResult | None
) -> Literal["none", "weak", "moderate", "strong"]:
    """Combine graded relevance with evidence volume, source diversity, trials, and labels."""
    evidence = list(documents)
    if not evidence:
        return "none"

    relevance = grade.document_relevance if grade else []
    relevant_count = sum(item.relevance == "relevant" for item in relevance)
    if relevant_count < MIN_RELEVANT_DOCUMENTS:
        return "weak"

    score = 0
    if len(evidence) >= 3:
        score += 1
    if len(evidence) >= 8:
        score += 1

    source_count = len({item.source.casefold() for item in evidence if item.source})
    if source_count >= 2:
        score += 1
    if source_count >= 3:
        score += 1

    has_trial_phase = any(
        item.source.casefold() == "clinicaltrials.gov"
        and any(
            str(phase).strip().casefold() not in {"", "na", "n/a", "unknown"}
            for phase in (
                item.raw_payload.get("phase")
                if isinstance(item.raw_payload.get("phase"), list)
                else [item.raw_payload.get("phase")]
            )
        )
        for item in evidence
    )
    has_label = any(
        isinstance(item.raw_payload.get("section"), str)
        and bool(item.raw_payload["section"].strip())
        for item in evidence
    )
    score += int(has_trial_phase) + int(has_label)

    if grade and grade.evidence_level == "moderate":
        score += 1
    elif grade and grade.evidence_level == "strong":
        score += 2

    if score >= 5:
        return "strong"
    if score >= 2:
        return "moderate"
    return "weak"


def _claims_in_brief(brief: Brief) -> Iterable[tuple[str, CitedClaim]]:
    if brief.summary is not None:
        yield "summary", brief.summary
    for index, claim in enumerate(brief.key_findings):
        yield f"key_findings[{index}]", claim
    for severity, claims in brief.safety_findings_by_severity.items():
        for index, claim in enumerate(claims):
            yield f"safety_findings_by_severity[{severity}][{index}]", claim


def validate_brief(brief: Brief, known_doc_ids: Iterable[str]) -> Brief:
    """Reject claims with missing or unknown citations and malformed safety groups."""
    known_ids = set(known_doc_ids)
    if known_ids and brief.summary is None:
        raise BriefValidationError("A summary claim is required when evidence documents exist")

    for location, claim in _claims_in_brief(brief):
        if not claim.text.strip():
            raise BriefValidationError(f"{location} must not be empty")
        if not claim.doc_ids:
            raise BriefValidationError(f"{location} has no document citations")
        unknown_ids = set(claim.doc_ids) - known_ids
        if unknown_ids:
            raise BriefValidationError(
                f"{location} cites unknown doc_ids: {', '.join(sorted(unknown_ids))}"
            )

    invalid_severities = set(brief.safety_findings_by_severity) - set(SAFETY_SEVERITY_ORDER)
    if invalid_severities:
        raise BriefValidationError(
            f"Unsupported safety finding severities: {', '.join(sorted(invalid_severities))}"
        )

    return brief


def _format_claim(claim: CitedClaim) -> str:
    citations = ", ".join(f"`{doc_id}`" for doc_id in claim.doc_ids)
    return f"{claim.text} [{citations}]"


def render_brief_markdown(brief: Brief) -> str:
    """Render every required brief section, including an explicit evidence-gaps section."""
    lines = ["# Regulatory Evidence Brief", "", "## Summary"]
    if brief.summary is None:
        lines.append("Insufficient evidence to assess safety.")
    else:
        lines.append(_format_claim(brief.summary))

    lines.extend(["", "## Evidence Level", brief.evidence_level, "", "## Key Findings"])
    if brief.key_findings:
        lines.extend(f"- {_format_claim(claim)}" for claim in brief.key_findings)
    else:
        lines.append("No supported key findings were generated.")

    lines.extend(["", "## Safety Findings by Severity"])
    any_safety_findings = False
    for severity in SAFETY_SEVERITY_ORDER:
        claims = brief.safety_findings_by_severity.get(severity, [])
        if not claims:
            continue
        any_safety_findings = True
        lines.extend(["", f"### {severity.title()}"])
        lines.extend(f"- {_format_claim(claim)}" for claim in claims)
    if not any_safety_findings:
        lines.append("No cited safety findings were generated.")

    lines.extend(["", "## Evidence Gaps"])
    if brief.evidence_gaps:
        lines.extend(f"- {gap}" for gap in brief.evidence_gaps)
    else:
        lines.append("- No specific evidence gaps identified.")

    lines.extend(["", "## Sources"])
    if brief.sources:
        lines.extend(
            f"- `{source.doc_id}` — {source.source}: [{source.title}]({source.url})"
            for source in brief.sources
        )
    else:
        lines.append("No source documents were retrieved.")

    lines.extend(["", "## Disclaimer", brief.disclaimer])
    return "\n".join(lines)


def _claims_payload(items: Iterable[EvidenceItem]) -> list[dict[str, str]]:
    return [
        {
            "doc_id": item.doc_id,
            "source": item.source,
            "title": item.title,
            "url": item.url,
            "text": item.summary[:4000],
        }
        for item in items
    ]


def _sources_for(documents: Iterable[EvidenceItem]) -> list[BriefSource]:
    return [
        BriefSource(
            doc_id=item.doc_id,
            source=item.source,
            title=item.title,
            url=item.url,
        )
        for item in documents
    ]


def _evidence_gaps(state: Mapping, safety_findings: list[SafetyFinding]) -> list[str]:
    gaps = list((state.get("grade_result") and state["grade_result"].missing_topics) or [])
    gaps.extend(
        finding.summary
        for finding in safety_findings
        if finding.severity == "no_evidence"
    )
    if not state.get("documents"):
        gaps.append("No evidence documents were retrieved.")
    for entry in state.get("audit_log") or []:
        if "failed" in entry.casefold() and entry not in gaps:
            gaps.append(entry)
    return list(dict.fromkeys(gap for gap in gaps if gap))


def _brief_context(
    state: Mapping,
    documents: list[EvidenceItem],
    evidence_level: Literal["none", "weak", "moderate", "strong"],
) -> dict:
    grade_result = state.get("grade_result")
    return {
        "drug": state.get("canonical_name") or state.get("drug_query", "Unknown"),
        "evidence_level": evidence_level,
        "documents": _claims_payload(documents),
        "grade_result": grade_result.model_dump() if grade_result else None,
        "safety_findings": [
            finding.model_dump() for finding in state.get("safety_findings", [])
        ],
        "critic_feedback": state.get("critic_feedback", []),
        "redraft_count": state.get("redraft_count", 0),
        "human_feedback": state.get("human_feedback", ""),
        "edited_brief": state.get("edited_brief", ""),
        "human_rounds": state.get("human_rounds", 0),
        "evidence_gaps": _evidence_gaps(state, state.get("safety_findings", [])),
        "required_disclaimer": NON_MEDICAL_ADVICE_DISCLAIMER,
    }


async def draft_node(state: Mapping, *, model=None) -> dict:
    """Create a citation-validated Brief and regenerate once when validation fails."""
    documents = list(state.get("documents") or [])
    known_doc_ids = {item.doc_id for item in documents}
    evidence_level = compute_evidence_level(documents, state.get("grade_result"))
    context = _brief_context(state, documents, evidence_level)

    if not documents:
        brief = Brief(
            summary=None,
            evidence_level="none",
            key_findings=[],
            safety_findings_by_severity={},
            evidence_gaps=context["evidence_gaps"],
            sources=[],
            disclaimer=NON_MEDICAL_ADVICE_DISCLAIMER,
        )
        return {"brief": brief, "regulatory_brief": render_brief_markdown(brief)}

    messages = [
        SystemMessage(content=load_prompt("brief_drafter")),
        HumanMessage(content=json.dumps(context, ensure_ascii=False)),
    ]
    structured_model = (model or get_llm("drafter")).with_structured_output(Brief)
    last_error: Exception | None = None

    for attempt in range(MAX_BRIEF_REGENERATION_ATTEMPTS + 1):
        try:
            response = await structured_model.ainvoke(messages)
            brief = Brief.model_validate(response)
            brief = brief.model_copy(
                update={
                    "evidence_level": evidence_level,
                    "evidence_gaps": list(
                        dict.fromkeys(brief.evidence_gaps + context["evidence_gaps"])
                    ),
                    "sources": _sources_for(documents),
                    "disclaimer": NON_MEDICAL_ADVICE_DISCLAIMER,
                }
            )
            if evidence_level in {"none", "weak"}:
                brief = brief.model_copy(
                    update={
                        "summary": CitedClaim(
                            text=INSUFFICIENT_EVIDENCE_HEADLINE,
                            doc_ids=sorted(known_doc_ids),
                        )
                    }
                )
            validate_brief(brief, known_doc_ids)
            return {"brief": brief, "regulatory_brief": render_brief_markdown(brief)}
        except Exception as error:
            last_error = error
            if attempt < MAX_BRIEF_REGENERATION_ATTEMPTS:
                messages.append(
                    HumanMessage(
                        content=(
                            f"Brief validation failed: {error}. Regenerate with citations for every "
                            "summary, key finding, and safety claim. Use only the supplied doc_ids."
                        )
                    )
                )

    raise BriefValidationError(
        f"Brief generation failed validation after regeneration: {last_error}"
    ) from last_error