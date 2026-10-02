"""The retrieval pipeline: retrieve from each source, fuse, rerank, and gate."""

from __future__ import annotations

from dataclasses import dataclass, field

from cqa.config import Config
from cqa.index.build import IndexHandle
from cqa.types import Chunk, Embedder, Reranker, Retriever, Scored


@dataclass
class Retrieval:
    """The result of one retrieval, with every intermediate list.

    Attributes:
        lists: Each retriever's ranked list, before fusion.
        fused: The fused candidates.
        reranked: The candidates kept after reranking.
        chunks: Every chunk referenced above, by id.
        top_score: The best rerank score, or None without a model-backed reranker.
        low_confidence: Whether the confidence gate fired.
        timings_ms: Milliseconds per stage.
    """

    lists: dict[str, list[Scored]]
    fused: list[Scored]
    reranked: list[Scored]
    chunks: dict[int, Chunk]
    top_score: float | None
    low_confidence: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


class RetrievalPipeline:
    """Runs the configured retrievers, fusion, reranker, and gate over one index."""

    def __init__(
        self, cfg: Config, retrievers: dict[str, Retriever], reranker: Reranker, index: IndexHandle
    ) -> None:
        self.cfg = cfg
        self.retrievers = retrievers
        self.reranker = reranker
        self.index = index

    @classmethod
    def from_config(cls, cfg: Config, index: IndexHandle, embedder: Embedder) -> RetrievalPipeline:
        """Build the retrievers in ``cfg.retrieve.lists`` and the reranker in ``cfg.rerank``."""
        raise NotImplementedError

    def run(self, question: str) -> Retrieval:
        """Retrieve for a question, timing each stage.

        Retrievers run concurrently. With more than one list, results are
        fused with ``rrf``; a single list is used as is. The top
        ``cfg.retrieve.candidates`` are reranked, and the gate is applied to
        the best rerank score.
        """
        raise NotImplementedError
