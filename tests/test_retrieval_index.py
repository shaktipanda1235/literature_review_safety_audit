from langchain_core.embeddings import Embeddings

from app.retrieval import EvidenceRetriever
from app.schemas import EvidenceItem


class KeywordEmbeddings(Embeddings):
    def _embed(self, text):
        lowered = text.casefold()
        return [
            1.0 if "hepatotoxicity" in lowered or "liver injury" in lowered else 0.0,
            1.0 if "qt prolongation" in lowered else 0.0,
            1.0 if "renal failure" in lowered else 0.0,
        ]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_top_k_ranks_matching_safety_chunk_and_retains_provenance():
    evidence = [
        EvidenceItem(
            source="openFDA",
            title="Boxed warning",
            summary="Serious hepatotoxicity and liver injury have been reported.",
            url="https://api.fda.gov/drug/label.json?id=label-1",
            raw_payload={"section": "boxed_warning"},
            doc_id="label-1-boxed-warning",
        ),
        EvidenceItem(
            source="ClinicalTrials.gov",
            title="Cardiac monitoring",
            summary="The protocol monitors QT prolongation during treatment.",
            url="https://clinicaltrials.gov/study/NCT00000001",
            doc_id="NCT00000001",
        ),
    ]

    with EvidenceRetriever(
        embeddings=KeywordEmbeddings(), chunk_size=200, chunk_overlap=20
    ) as retriever:
        assert retriever.add_evidence(evidence) == 2
        results = retriever.top_k("hepatotoxicity", k=1)

    assert len(results) == 1
    assert "hepatotoxicity" in results[0].page_content
    assert results[0].metadata["doc_id"] == "label-1-boxed-warning"
    assert results[0].metadata["url"] == evidence[0].url
    assert results[0].metadata["section"] == "boxed_warning"


def test_long_evidence_is_split_and_every_chunk_keeps_source_id_and_url():
    evidence = EvidenceItem(
        source="PubMed",
        title="Safety report",
        summary=("hepatotoxicity evidence " * 30),
        url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        doc_id="pmid-12345",
    )

    with EvidenceRetriever(
        embeddings=KeywordEmbeddings(), chunk_size=80, chunk_overlap=10
    ) as retriever:
        chunk_count = retriever.add_evidence([evidence])
        results = retriever.top_k("hepatotoxicity", k=chunk_count)

    assert chunk_count > 1
    assert len(results) == chunk_count
    assert all(result.metadata["doc_id"] == "pmid-12345" for result in results)
    assert all(result.metadata["url"] == evidence.url for result in results)


def test_evidence_item_generates_stable_doc_id_from_provenance():
    first = EvidenceItem(
        source="PubMed",
        title="Example article",
        summary="First instance",
        url="https://pubmed.ncbi.nlm.nih.gov/12345/",
    )
    second = EvidenceItem(
        source="PubMed",
        title="Example article",
        summary="Updated summary",
        url="https://pubmed.ncbi.nlm.nih.gov/12345/",
    )

    assert first.doc_id
    assert first.doc_id == second.doc_id