from __future__ import annotations

from typing import Any, Dict, List

from app.config import (
    CLINICALTRIALS_BASE,
    CLINICALTRIALS_MAX_PAGES,
    CLINICALTRIALS_PAGE_SIZE,
    MAX_RESULTS_PER_SOURCE,
)
from app.schemas import EvidenceItem
from app.sources.base import HttpClient


def _serious_adverse_event_summary(module: Dict[str, Any]) -> str:
    parts = [str(module.get("description") or "").strip()]

    for event in module.get("seriousEvents") or []:
        label = ": ".join(
            value for value in (event.get("organSystem"), event.get("term")) if value
        )
        if label:
            parts.append(label)

    for group in module.get("eventGroups") or []:
        affected = group.get("seriousNumAffected")
        at_risk = group.get("seriousNumAtRisk")
        if affected is not None:
            group_name = group.get("title") or group.get("id") or "Unknown group"
            count = f"{affected}/{at_risk}" if at_risk is not None else str(affected)
            parts.append(f"{count} participants in {group_name}")

    return " ".join(part for part in parts if part)


async def search_clinical_trials(
    drug_query: str,
    max_results: int = MAX_RESULTS_PER_SOURCE,
    client: HttpClient | None = None,
    *,
    page_size: int = CLINICALTRIALS_PAGE_SIZE,
    max_pages: int = CLINICALTRIALS_MAX_PAGES,
) -> List[EvidenceItem]:
    """Query ClinicalTrials.gov API v2 and normalize study records."""
    if not drug_query.strip() or max_results <= 0 or max_pages <= 0:
        return []

    own_client = False
    if client is None:
        client = HttpClient()
        own_client = True

    items: List[EvidenceItem] = []
    page_token = None
    effective_page_size = min(max(page_size, 1), 1000, max_results)
    url = f"{CLINICALTRIALS_BASE}/studies"

    try:
        for _ in range(max_pages):
            params = {
                "query.intr": drug_query,
                "pageSize": effective_page_size,
                "format": "json",
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                response = await client.request("GET", url, params=params, use_cache=True)
                payload = response.json()
            except Exception as error:
                print(f"[ClinicalTrials] Error searching: {error}")
                break

            studies = payload.get("studies") or []
            for study in studies:
                if len(items) >= max_results:
                    break

                try:
                    protocol = study.get("protocolSection") or {}
                    identification = protocol.get("identificationModule") or {}
                    status = protocol.get("statusModule") or {}
                    description = protocol.get("descriptionModule") or {}
                    design = protocol.get("designModule") or {}
                    enrollment = design.get("enrollmentInfo") or {}
                    conditions_module = protocol.get("conditionsModule") or {}
                    outcomes_module = protocol.get("outcomesModule") or {}
                    results = study.get("resultsSection") or {}
                    adverse_events = results.get("adverseEventsModule") or {}

                    nct_id = identification.get("nctId", "")
                    phase = design.get("phases") or []
                    conditions = conditions_module.get("conditions") or []
                    primary_outcomes = outcomes_module.get("primaryOutcomes") or []
                    serious_events = adverse_events.get("seriousEvents") or []
                    serious_summary = _serious_adverse_event_summary(adverse_events)
                    brief_summary = description.get("briefSummary") or "No summary available."
                    summary = brief_summary
                    if serious_summary:
                        summary = f"{summary}\n\nSerious adverse events: {serious_summary}"

                    items.append(
                        EvidenceItem(
                            source="ClinicalTrials.gov",
                            title=identification.get("briefTitle") or "Untitled study",
                            summary=summary,
                            url=f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
                            publication_date=(status.get("studyFirstPostDateStruct") or {}).get("date"),
                            study_type=design.get("studyType"),
                            raw_payload={
                                "nct_id": nct_id,
                                "phase": phase,
                                "status": status.get("overallStatus"),
                                "enrollment_count": enrollment.get("count"),
                                "enrollment_type": enrollment.get("type"),
                                "conditions": conditions,
                                "primary_outcomes": primary_outcomes,
                                "has_results": study.get("hasResults"),
                                "serious_adverse_event_summary": serious_summary,
                                "serious_adverse_events": serious_events,
                            },
                        )
                    )
                except (AttributeError, TypeError):
                    continue

            if len(items) >= max_results:
                break
            page_token = payload.get("nextPageToken")
            if not page_token:
                break
    finally:
        if own_client:
            await client.close()

    return items
