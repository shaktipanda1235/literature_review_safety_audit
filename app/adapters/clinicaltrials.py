from __future__ import annotations

from typing import List

from app.config import CLINICALTRIALS_BASE, MAX_RESULTS_PER_SOURCE
from app.schemas import EvidenceItem
from app.sources.base import HttpClient


async def search_clinical_trials(
    drug_query: str, max_results: int = MAX_RESULTS_PER_SOURCE, client: HttpClient | None = None
) -> List[EvidenceItem]:
    """Query ClinicalTrials.gov and normalize trial metadata."""
    own_client = False
    if client is None:
        client = HttpClient()
        own_client = True

    url = f"{CLINICALTRIALS_BASE}/study_fields"
    params = {
        "expr": drug_query,
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,StudyType,PrimaryOutcome,StudyFirstPostDate,BriefSummary",
        "min_rnk": 1,
        "max_rnk": max_results,
        "fmt": "json",
    }

    try:
        response = await client.request("GET", url, params=params, use_cache=True)
    except Exception as e:
        print(f"[ClinicalTrials] Error searching: {e}")
        if own_client:
            await client.close()
        return []

    try:
        payload = response.json()
    except Exception:
        if own_client:
            await client.close()
        return []

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

    if own_client:
        await client.close()
    return items
