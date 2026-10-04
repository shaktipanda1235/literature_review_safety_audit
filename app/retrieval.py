from __future__ import annotations

from hashlib import sha256
from typing import Iterable, List
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_CHUNK_OVERLAP,
    RETRIEVAL_CHUNK_SIZE,
    RETRIEVAL_DEFAULT_TOP_K,
)
from app.schemas import EvidenceItem


class EvidenceRetriever:
    """Chunk and rank evidence in an ephemeral, per-instance Chroma collection."""

    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        chunk_size: int = RETRIEVAL_CHUNK_SIZE,
        chunk_overlap: int = RETRIEVAL_CHUNK_OVERLAP,
        embeddings: Embeddings | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

        if embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        self.embedding_model = embedding_model
        self.embeddings = embeddings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        self.collection_name = f"evidence_{uuid4().hex}"
        self._vector_store = None
        self._chunk_count = 0

    def __enter__(self) -> EvidenceRetriever:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _get_vector_store(self):
        if self._vector_store is None:
            from langchain_chroma import Chroma

            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        return self._vector_store

    def add_evidence(self, evidence_items: Iterable[EvidenceItem]) -> int:
        """Chunk and index evidence, retaining source provenance on every chunk."""
        chunks: List[Document] = []
        chunk_ids: List[str] = []

        for item in evidence_items:
            text = "\n\n".join(part for part in (item.title, item.summary) if part).strip()
            if not text:
                continue

            metadata = {
                "doc_id": item.doc_id,
                "url": item.url,
                "source": item.source,
                "title": item.title,
            }
            section = item.raw_payload.get("section")
            if isinstance(section, str) and section:
                metadata["section"] = section

            item_chunks = self.splitter.create_documents([text], metadatas=[metadata])
            for chunk_index, chunk in enumerate(item_chunks):
                chunk.metadata["chunk_index"] = chunk_index
                offset = chunk.metadata.get("start_index", 0)
                chunk_ids.append(
                    sha256(
                        f"{item.doc_id}\0{offset}\0{chunk.page_content}".encode("utf-8")
                    ).hexdigest()
                )
                chunks.append(chunk)

        if not chunks:
            return 0

        self._get_vector_store().add_documents(documents=chunks, ids=chunk_ids)
        self._chunk_count += len(chunks)
        return len(chunks)

    def top_k(self, query: str, k: int = RETRIEVAL_DEFAULT_TOP_K) -> List[Document]:
        """Return the most relevant chunks, including source doc_id and URL metadata."""
        if not query.strip() or k <= 0 or self._chunk_count == 0:
            return []
        return self._get_vector_store().similarity_search(query, k=k)

    def close(self) -> None:
        """Delete this run's in-memory collection and release its reference."""
        if self._vector_store is not None:
            self._vector_store.delete_collection()
            self._vector_store = None
        self._chunk_count = 0