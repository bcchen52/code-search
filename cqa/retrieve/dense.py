"""Dense retrieval: embed the question and search the vector store."""

from __future__ import annotations

from cqa.types import Embedder, Scored, VectorStore


class DenseRetriever:
    """Ranks chunks by cosine similarity to the question."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, question: str, k: int) -> list[Scored]:
        """Embed the question in query mode and return the top ``k`` chunks with ``source="dense"``."""
        if k <= 0:
            return []
        query = self.embedder.embed([question], "query")[0]
        return [
            Scored(chunk_id=cid, score=score, rank=i + 1, source="dense")
            for i, (cid, score) in enumerate(self.store.search(query, k))
        ]
