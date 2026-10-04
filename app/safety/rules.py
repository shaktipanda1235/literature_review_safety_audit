from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import yaml

from app.adapters.openfda_faers import FAERS_CAVEAT
from app.schemas import EvidenceItem, SafetyFinding


DEFAULT_WATCHLIST_PATH = Path(__file__).with_name("watchlist.yaml")


def load_safety_watchlist(
    path: str | Path = DEFAULT_WATCHLIST_PATH,
) -> dict[str, list[str]]:
    """Load and validate category-to-term safety rules from YAML."""
    with Path(path).open("r", encoding="utf-8") as watchlist_file:
        raw_watchlist = yaml.safe_load(watchlist_file)

    if not isinstance(raw_watchlist, dict):
        raise ValueError("Safety watchlist must be a mapping of categories to terms")

    watchlist: dict[str, list[str]] = {}
    for category, terms in raw_watchlist.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Safety watchlist categories must be non-empty strings")
        if not isinstance(terms, list) or not all(
            isinstance(term, str) and term.strip() for term in terms
        ):
            raise ValueError(f"Safety watchlist category '{category}' must contain non-empty terms")
        watchlist[category.strip()] = [term.strip() for term in terms]

    return watchlist


def findings_from_label(item: EvidenceItem) -> list[SafetyFinding]:
    """Create critical/high findings from boxed-warning and contraindication sections."""
    section = item.raw_payload.get("section")
    if section == "boxed_warning":
        return [
            SafetyFinding(
                category="boxed_warning",
                severity="critical",
                summary=f"Boxed warning in {item.title}: {item.summary}",
                supporting_doc_ids=[item.doc_id],
                quote=item.summary,
            )
        ]
    if section == "contraindications":
        return [
            SafetyFinding(
                category="contraindication",
                severity="high",
                summary=f"Contraindication in {item.title}: {item.summary}",
                supporting_doc_ids=[item.doc_id],
                quote=item.summary,
            )
        ]
    return []


def findings_from_faers(
    item: EvidenceItem,
    watchlist: Mapping[str, list[str]],
) -> list[SafetyFinding]:
    """Create moderate, caveated findings for FAERS reaction terms on the watchlist."""
    if "FAERS" not in item.safety_flags and "faers" not in item.source.casefold():
        return []

    reactions = item.raw_payload.get("reactions")
    if not isinstance(reactions, list):
        return []

    drug = item.raw_payload.get("drug_query") or item.title
    caveat = item.raw_payload.get("caveat") or FAERS_CAVEAT
    findings: list[SafetyFinding] = []
    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        term = reaction.get("term")
        if not isinstance(term, str) or not term.strip():
            continue
        normalized_term = term.casefold()
        try:
            count = int(reaction.get("count"))
        except (TypeError, ValueError):
            continue

        for category, terms in watchlist.items():
            if any(watch_term.casefold() in normalized_term for watch_term in terms):
                findings.append(
                    SafetyFinding(
                        category=category,
                        severity="moderate",
                        summary=(
                            f"FAERS reports {term} {count} times for {drug}. {caveat}"
                        ),
                        supporting_doc_ids=[item.doc_id],
                        quote=f"{term}: {count}",
                        caveat=caveat,
                    )
                )

    return findings


def findings_from_evidence(
    documents: Iterable[EvidenceItem],
    watchlist: Mapping[str, list[str]] | None = None,
) -> list[SafetyFinding]:
    """Apply deterministic label and FAERS rules to evidence documents."""
    active_watchlist = watchlist if watchlist is not None else load_safety_watchlist()
    findings: list[SafetyFinding] = []
    for document in documents:
        findings.extend(findings_from_label(document))
        findings.extend(findings_from_faers(document, active_watchlist))
    return findings