from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import httpx

from app.config import API_TIMEOUT_SECONDS, MAX_RESULTS_PER_SOURCE, PUBMED_BASE
from app.schemas import EvidenceItem


def search_pubmed(drug_query: str, max_results: int = MAX_RESULTS_PER_SOURCE) -> List[EvidenceItem]:
    """Query PubMed for literature around a drug and normalize results."""
    params = {
        "db": "pubmed",
        "term": f"{drug_query} AND (clinical trial OR review OR safety)",
        "retmode": "xml",
        "retmax": str(max_results),
        "sort": "relevance",
    }

    try:
        response = httpx.get(
            f"{PUBMED_BASE}/esearch.fcgi",
            params=params,
            timeout=API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception:
        return []

    root = ET.fromstring(response.text)
    ids = [node.text for node in root.findall("./IdList/Id") if node is not None]
    if not ids:
        return []

    fetch_url = f"{PUBMED_BASE}/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
    }

    try:
        fetch_resp = httpx.get(fetch_url, params=fetch_params, timeout=API_TIMEOUT_SECONDS)
        fetch_resp.raise_for_status()
    except Exception:
        return []

    tree = ET.fromstring(fetch_resp.text)
    items: List[EvidenceItem] = []
    for article in tree.findall("./PubmedArticleSet/PubmedArticle"):
        med = article.find("MedlineCitation")
        if med is None:
            continue
        article_meta = med.find("Article")
        if article_meta is None:
            continue

        title = (article_meta.findtext("ArticleTitle") or "Untitled").strip()
        abstract_node = article_meta.find("Abstract")
        summary = ""
        if abstract_node is not None:
            abstract_text = " ".join(
                (text.strip() for text in abstract_node.itertext() if text and text.strip())
            )
            summary = abstract_text[:700]

        if not summary:
            summary = "No abstract available."

        pub_year = med.find("DateCompleted")
        pub_date = None
        if pub_year is not None:
            year = pub_year.findtext("Year")
            month = pub_year.findtext("Month")
            day = pub_year.findtext("Day")
            if year:
                pub_date = "-".join(part for part in [year, month or "01", day or "01"] if part)

        items.append(
            EvidenceItem(
                source="PubMed",
                title=title,
                summary=summary,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{ids[0]}/",
                publication_date=pub_date,
                study_type="literature",
                raw_payload={"article_id": ids[0] if ids else None},
            )
        )

    return items
