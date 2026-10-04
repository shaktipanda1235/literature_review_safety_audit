from __future__ import annotations

from typing import Any, List
from urllib.parse import urlencode

from app.config import MAX_RESULTS_PER_SOURCE, OPENFDA_BASE
from app.schemas import EvidenceItem
from app.sources.base import HttpClient


REACTION_COUNT_FIELD = "patient.reaction.reactionmeddrapt.exact"
FAERS_CAVEAT = (
    "FAERS reports are spontaneous and do not establish that a drug caused a reaction; "
    "report counts are not incidence rates or measures of risk."
)


async def search_openfda_faers(
    drug_query: str,
    max_results: int = 20,
    client: HttpClient | None = None,
) -> List[EvidenceItem]:
    """Return ranked FAERS reaction counts with an explicit interpretation caveat."""
    if not drug_query.strip() or max_results <= 0:
        return []

    own_client = False
    if client is None:
        client = HttpClient()
        own_client = True

    escaped_query = drug_query.replace("\\", "\\\\").replace('"', '\\"')
    params = {
        "search": f'patient.drug.medicinalproduct:"{escaped_query}"',
        "count": REACTION_COUNT_FIELD,
        "limit": min(max_results, 1000),
    }
    url = f"{OPENFDA_BASE}/event.json"

    try:
        try:
            response = await client.request("GET", url, params=params, use_cache=True)
            payload = response.json()
        except Exception as error:
            print(f"[openFDA FAERS] Error querying adverse events: {error}")
            return []

        if not isinstance(payload, dict):
            return []

        reactions = []
        for result in payload.get("results") or []:
            if not isinstance(result, dict):
                continue
            term = result.get("term")
            try:
                count = int(result.get("count"))
            except (TypeError, ValueError):
                continue
            if isinstance(term, str) and term.strip() and count >= 0:
                reactions.append({"term": term.strip(), "count": count})

        reactions.sort(key=lambda reaction: (-reaction["count"], reaction["term"].casefold()))
        reactions = reactions[: min(max_results, 1000)]
        if not reactions:
            return []

        ranked_reactions = [
            {"rank": rank, **reaction}
            for rank, reaction in enumerate(reactions, start=1)
        ]
        table_lines = ["| Rank | Reaction term | Reports |", "| ---: | --- | ---: |"]
        table_lines.extend(
            f"| {reaction['rank']} | {reaction['term']} | {reaction['count']} |"
            for reaction in ranked_reactions
        )
        summary = "\n".join(table_lines) + f"\n\nCaveat: {FAERS_CAVEAT}"
        citation_url = f"{url}?{urlencode(params)}"

        return [
            EvidenceItem(
                source="openFDA FAERS",
                title=f"FAERS reaction counts for {drug_query}",
                summary=summary,
                url=citation_url,
                study_type="FAERS reaction count",
                safety_flags=["FAERS"],
                raw_payload={
                    "drug_query": drug_query,
                    "reactions": ranked_reactions,
                    "caveat": FAERS_CAVEAT,
                    "count_field": REACTION_COUNT_FIELD,
                },
            )
        ]
    finally:
        if own_client:
            await client.close()