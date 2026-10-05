from __future__ import annotations

import re
from typing import Iterable

from app.brief import BriefValidationError, validate_brief
from app.config import MAX_REDRAFTS
from app.schemas import Brief, SafetyFinding


UNSUPPORTED_ASSURANCE_PATTERN = re.compile(
    r"\b(?:safe|no\s+risk|risk[- ]free|no\s+known\s+risk)\b", re.IGNORECASE
)
GENERIC_FINDING_TERMS = {
    "about",
    "after",
    "also",
    "because",
    "could",
    "evidence",
    "finding",
    "from",
    "have",
    "into",
    "reported",
    "result",
    "serious",
    "should",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "with",
}


def _claim_texts(brief: Brief) -> list[str]:
    texts = []
    if brief.summary is not None:
        texts.append(brief.summary.text)
    texts.extend(claim.text for claim in brief.key_findings)
    texts.extend(
        claim.text
        for claims in brief.safety_findings_by_severity.values()
        for claim in claims
    )
    return texts


def _finding_is_mentioned(finding: SafetyFinding, claim_texts: Iterable[str]) -> bool:
    normalized_claims = [" ".join(text.casefold().split()) for text in claim_texts]
    anchors = [finding.quote, finding.summary]
    for anchor in anchors:
        normalized_anchor = " ".join((anchor or "").casefold().split())
        if normalized_anchor and any(normalized_anchor in text for text in normalized_claims):
            return True

    finding_text = " ".join(
        part for part in (finding.category, finding.quote, finding.summary) if part
    ).casefold()
    terms = {
        term
        for term in re.findall(r"[a-z0-9]+", finding_text)
        if len(term) >= 5 and term not in GENERIC_FINDING_TERMS
    }
    if not terms:
        return False
    claim_terms = {
        term
        for text in normalized_claims
        for term in re.findall(r"[a-z0-9]+", text)
    }
    return bool(terms & claim_terms)


def critic_node(state: dict) -> dict:
    """Deterministically validate citations, key risks, and safety assurances."""
    feedback: list[str] = []
    brief = state.get("brief")
    documents = state.get("documents") or []
    known_doc_ids = {document.doc_id for document in documents}

    if not isinstance(brief, Brief):
        feedback.append("A structured brief is missing.")
    else:
        try:
            validate_brief(brief, known_doc_ids)
        except BriefValidationError as error:
            feedback.append(f"Citation validation failed: {error}")

        claim_texts = _claim_texts(brief)
        for finding in state.get("safety_findings") or []:
            if finding.severity not in {"critical", "high"}:
                continue
            if not _finding_is_mentioned(finding, claim_texts):
                feedback.append(
                    f"The {finding.severity} {finding.category} finding is not mentioned: "
                    f"{finding.summary}"
                )

        for claim_text in claim_texts:
            match = UNSUPPORTED_ASSURANCE_PATTERN.search(claim_text)
            if match:
                feedback.append(
                    f"Unsupported safety assurance '{match.group(0)}' appears in a claim."
                )

    if not feedback:
        return {
            "critic_feedback": [],
            "critic_passed": True,
            "critic_unresolved": False,
        }

    redraft_count = state.get("redraft_count", 0)
    if redraft_count >= MAX_REDRAFTS:
        audit_log = list(state.get("audit_log") or [])
        audit_log.append("Critic issues remain after the redraft limit was reached.")
        return {
            "critic_feedback": feedback,
            "critic_passed": False,
            "critic_unresolved": True,
            "audit_log": audit_log,
        }

    return {
        "critic_feedback": feedback,
        "critic_passed": False,
        "critic_unresolved": False,
        "redraft_count": redraft_count + 1,
    }