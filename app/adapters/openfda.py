from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx

from app.config import API_TIMEOUT_SECONDS, MAX_RESULTS_PER_SOURCE, OPENFDA_BASE
from app.schemas import EvidenceItem


def search_openfda(drug_query: str, max_results: int = MAX_RESULTS_PER_SOURCE) -> List[EvidenceItem]:
    """Query openFDA labels/adverse events and normalize safety signals."""
    endpoints = [
        ("/drug/event.json", "receivedate:[20040101+TO+20250131]"),
        ("/drug/label.json", f"openfda.brand_name:{drug_query} OR openfda.generic_name:{drug_query}"),
    ]
    results: List[EvidenceItem] = []

    for endpoint, search_term in endpoints:
        params = {
            "search": search_term,
            "limit": str(max_results),
        }
        try:
            response = httpx.get(f"{OPENFDA_BASE}{endpoint}", params=params, timeout=API_TIMEOUT_SECONDS)
            response.raise_for_status()
        except Exception:
            continue

        payload = response.json()
        items = payload.get("results", [])[:max_results]
        for item in items:
            title = item.get("patient", {}).get("drug", [{}])[0].get("medicinalproduct") or "FDA entry"
            summary = json.dumps(item, ensure_ascii=False)[:700]
            results.append(
                EvidenceItem(
                    source="openFDA",
                    title=title,
                    summary=summary,
                    url=f"{OPENFDA_BASE}{endpoint}",
                    study_type="safety",
                    safety_flags=["FDA-safety-check"],
                    raw_payload=item,
                )
            )

    return results
