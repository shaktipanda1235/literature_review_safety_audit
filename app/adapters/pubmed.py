from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List

from app.config import MAX_RESULTS_PER_SOURCE, PUBMED_BASE
from app.schemas import EvidenceItem
from app.sources.base import HttpClient


async def search_pubmed(
    drug_query: str, max_results: int = MAX_RESULTS_PER_SOURCE, client: HttpClient | None = None
) -> List[EvidenceItem]:
    """Query PubMed for literature around a drug and normalize results."""
    own_client = False
    if client is None:
        client = HttpClient()
        own_client = True

    params = {
        "db": "pubmed",
        "term": f"{drug_query} AND (clinical trial OR review OR safety)",
        "retmode": "xml",
        "retmax": str(max_results),
        "sort": "relevance",
    }

    try:
        response = await client.request(
            "GET",
            f"{PUBMED_BASE}/esearch.fcgi",
            params=params,
            use_cache=True,
        )
    except Exception as e:
        print(f"[PubMed] Error searching: {e}")
        if own_client:
            await client.close()
        return []

    try:
        root = ET.fromstring(response.text)
    except Exception:
        if own_client:
            await client.close()
        return []

    ids = [node.text for node in root.findall("./IdList/Id") if node is not None]
    if not ids:
        if own_client:
            await client.close()
        return []

    fetch_url = f"{PUBMED_BASE}/efetch.fcgi"
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
    }

    try:
        fetch_resp = await client.request("GET", fetch_url, params=fetch_params, use_cache=True)
    except Exception as e:
        print(f"[PubMed] Error fetching articles: {e}")
        if own_client:
            await client.close()
        return []

    try:
        tree = ET.fromstring(fetch_resp.text)
    except Exception:
        if own_client:
            await client.close()
        return []

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

    if own_client:
        await client.close()
    return items
