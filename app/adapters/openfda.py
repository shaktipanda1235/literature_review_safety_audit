from __future__ import annotations

from typing import Any, List
from urllib.parse import urlencode

from app.config import MAX_RESULTS_PER_SOURCE, OPENFDA_BASE
from app.schemas import EvidenceItem
from app.sources.base import HttpClient


LABEL_SECTIONS = (
    ("boxed_warning", "Boxed warning", ("boxed_warning",)),
    ("contraindications", "Contraindications", ("contraindications",)),
    (
        "warnings_and_cautions",
        "Warnings and precautions",
        ("warnings_and_cautions", "warnings", "precautions", "general_precautions"),
    ),
    ("adverse_reactions", "Adverse reactions", ("adverse_reactions",)),
    ("drug_interactions", "Drug interactions", ("drug_interactions",)),
    (
        "use_in_specific_populations",
        "Use in specific populations",
        (
            "use_in_specific_populations",
            "pregnancy",
            "nursing_mothers",
            "pediatric_use",
            "geriatric_use",
        ),
    ),
)


def _string_values(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def _name_search(drug_query: str) -> str:
    escaped_query = drug_query.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'(openfda.generic_name:"{escaped_query}" '
        f'OR openfda.brand_name:"{escaped_query}")'
    )


async def _fetch_labels(
    client: HttpClient, search_term: str, limit: int
) -> List[dict]:
    try:
        response = await client.request(
            "GET",
            f"{OPENFDA_BASE}/label.json",
            params={"search": search_term, "limit": limit},
            use_cache=True,
        )
        payload = response.json()
    except Exception as error:
        print(f"[openFDA label] Error querying labels: {error}")
        return []

    if not isinstance(payload, dict):
        return []
    records = payload.get("results") or []
    return [record for record in records if isinstance(record, dict)]


def _first_string(value: Any) -> str:
    values = _string_values(value)
    return values[0] if values else ""


async def search_openfda(
    drug_query: str, max_results: int = MAX_RESULTS_PER_SOURCE, client: HttpClient | None = None
) -> List[EvidenceItem]:
    """Query openFDA drug labels and emit a separate evidence item per section."""
    if not drug_query.strip() or max_results <= 0:
        return []

    own_client = False
    if client is None:
        client = HttpClient()
        own_client = True

    name_search = _name_search(drug_query)
    record_limit = min(max_results, 100)
    try:
        labels = await _fetch_labels(client, name_search, record_limit)

        has_hepatotoxicity_box_warning = any(
            "hepatotoxicity" in value.lower()
            for label in labels
            for value in _string_values(label.get("boxed_warning"))
        )
        if labels and not has_hepatotoxicity_box_warning:
            safety_search = f"boxed_warning:hepatotoxicity AND {name_search}"
            additional_labels = await _fetch_labels(client, safety_search, record_limit)
            seen_ids = {label.get("id") or label.get("set_id") for label in labels}
            labels.extend(
                label
                for label in additional_labels
                if (label.get("id") or label.get("set_id")) not in seen_ids
            )

        results: List[EvidenceItem] = []
        for label in labels:
            openfda = label.get("openfda") or {}
            product_name = (
                _first_string(openfda.get("brand_name"))
                or _first_string(openfda.get("generic_name"))
                or drug_query
            )
            label_id = label.get("id") or label.get("set_id")
            source_url = f"{OPENFDA_BASE}/label.json"
            if label_id:
                source_url += "?" + urlencode({"search": f'id:"{label_id}"'})

            for section_key, section_title, fields in LABEL_SECTIONS:
                source_fields = []
                section_text = []
                for field in fields:
                    values = _string_values(label.get(field))
                    if values:
                        source_fields.append(field)
                        section_text.extend(values)

                if not section_text:
                    continue

                results.append(
                    EvidenceItem(
                        source="openFDA",
                        title=f"{product_name} - {section_title}",
                        summary="\n\n".join(section_text),
                        url=source_url,
                        study_type="drug_label_section",
                        safety_flags=[section_key],
                        raw_payload={
                            "section": section_key,
                            "source_fields": source_fields,
                            "label_id": label.get("id"),
                            "set_id": label.get("set_id"),
                            "brand_names": _string_values(openfda.get("brand_name")),
                            "generic_names": _string_values(openfda.get("generic_name")),
                            "route": _string_values(openfda.get("route")),
                        },
                    )
                )

        return results
    finally:
        if own_client:
            await client.close()
