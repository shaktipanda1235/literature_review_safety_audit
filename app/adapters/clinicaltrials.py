from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx

from app.config import API_TIMEOUT_SECONDS, CLINICALTRIALS_BASE, MAX_RESULTS_PER_SOURCE
from app.schemas import EvidenceItem


def search_clinical_trials(drug_query: str, max_results: int = MAX_RESULTS_PER_SOURCE) -> List[EvidenceItem]:
    """Query ClinicalTrials.gov and normalize trial metadata."""
    url = f"{CLINICALTRIALS_BASE}/study_fields"
    params = {
        "expr": drug_query,
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,StudyType,PrimaryOutcome,StudyFirstPostDate,BriefSummary",
        "min_rnk": 1,
        "max_rnk": max_results,
        "fmt": "json",
    }

    try:
        response = httpx.get(url, params=params, timeout=API_TIMEOUT_SECONDS)
        response.raise_for_status()
    except Exception:
        return []

    payload = response.json()
    studies = payload.get("StudyFieldsResponse", {}).get("StudyFields", [])
    items: List[EvidenceItem] = []

    for study in studies:
        try:
            nct_id = study.get("NCTId", [""])[0]
            title = study.get("BriefTitle", ["Untitled"])[0]
            summary = study.get("BriefSummary", ["No summary available."])[0]
            phase = ", ".join(study.get("Phase", [])) or "Unknown"
            study_type = ", ".join(study.get("StudyType", [])) or "Unknown"
            items.append(
                EvidenceItem(
                    source="ClinicalTrials.gov",
                    title=title,
                    summary=summary,
                    url=f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
                    publication_date=study.get("StudyFirstPostDate", [None])[0],
                    study_type=study_type,
                    safety_flags=[phase] if phase else [],
                    raw_payload={"nct_id": nct_id, "phase": phase},
                )
            )
        except Exception:
            continue

    return items
