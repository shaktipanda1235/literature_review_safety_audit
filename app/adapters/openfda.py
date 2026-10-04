from __future__ import annotations

import json
from typing import List

from app.config import MAX_RESULTS_PER_SOURCE, OPENFDA_BASE
from app.schemas import EvidenceItem
from app.sources.base import HttpClient


async def search_openfda(
    drug_query: str, max_results: int = MAX_RESULTS_PER_SOURCE, client: HttpClient | None = None
) -> List[EvidenceItem]:
    """Query openFDA labels/adverse events and normalize safety signals."""
    own_client = False
    if client is None:
        client = HttpClient()
        own_client = True

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
            response = await client.request(
                "GET", f"{OPENFDA_BASE}{endpoint}", params=params, use_cache=True
            )
        except Exception as e:
            print(f"[openFDA] Error querying {endpoint}: {e}")
            continue

        try:
            payload = response.json()
        except Exception:
            continue

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

    if own_client:
        await client.close()
    return results
