"""The retrieval pipeline: retrieve from each source, fuse, rerank, and gate."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from cqa.config import Config
from cqa.errors import ConfigError
from cqa.index.build import IndexHandle
from cqa.retrieve.dense import DenseRetriever
from cqa.retrieve.gate import is_low_confidence
from cqa.retrieve.rerank import make_reranker
from cqa.types import Chunk, Embedder, Reranker, Retriever, Scored

AVAILABLE_LISTS = frozenset({"dense"})
"""Retrievers this version can build; the others raise ``ConfigError``."""


def _ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000


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
        """Build the retrievers in ``cfg.retrieve.lists`` and the reranker in ``cfg.rerank``.

        Raises:
            ConfigError: If a configured retriever or reranker is not available
                yet (only ``dense`` and ``none`` are, for now), or the index has
                no vector store.
        """
        unavailable = sorted(set(cfg.retrieve.lists) - AVAILABLE_LISTS)
        if unavailable:
            raise ConfigError(f"retrieve.lists: {', '.join(unavailable)} not available yet")
        if index.store is None:
            raise ConfigError("retrieve.lists: dense retrieval needs an index opened with a vector store")
        retrievers: dict[str, Retriever] = {"dense": DenseRetriever(embedder, index.store)}
        return cls(cfg, retrievers, make_reranker(cfg.rerank), index)

    def run(self, question: str) -> Retrieval:
        """Retrieve for a question, timing each stage.

        Retrievers run concurrently. With more than one list, results are
        fused with ``rrf``; a single list is used as is. Either way the
        candidates are cut to ``cfg.retrieve.candidates``, so the candidate
        stage has the same depth whether or not fusion ran. The candidates are
        reranked, and the gate is applied to the best rerank score, which
        exists only for a model-backed reranker.

        ``timings_ms`` records each retriever by name, then ``load``,
        ``rerank``, and ``total``.
        """
        started = time.perf_counter()
        r = self.cfg.retrieve
        depth = {"dense": r.dense_k, "bm25": r.bm25_k, "symbol": r.symbol_refs_k}
        timings: dict[str, float] = {}

        def timed(name: str, retriever: Retriever) -> tuple[str, list[Scored], float]:
            t = time.perf_counter()
            return name, retriever.retrieve(question, depth[name]), _ms(t)

        lists: dict[str, list[Scored]] = {}
        with ThreadPoolExecutor(max_workers=len(self.retrievers)) as pool:
            for name, ranked, ms in pool.map(lambda item: timed(*item), self.retrievers.items()):
                lists[name], timings[name] = ranked, ms
        if len(lists) != 1:
            raise ConfigError("retrieve.lists: fusing several lists is not available yet")
        fused = next(iter(lists.values()))[: r.candidates]

        t = time.perf_counter()
        chunks = self.index.chunks([s.chunk_id for ranked in lists.values() for s in ranked])
        timings["load"] = _ms(t)

        t = time.perf_counter()
        candidates = [chunks[s.chunk_id] for s in fused if s.chunk_id in chunks]
        reranked = self.reranker.rerank(question, candidates, self.cfg.rerank.keep)
        timings["rerank"] = _ms(t)

        top_score = reranked[0].score if reranked and self.cfg.rerank.model != "none" else None
        timings["total"] = _ms(started)
        return Retrieval(
            lists=lists,
            fused=fused,
            reranked=reranked,
            chunks=chunks,
            top_score=top_score,
            low_confidence=is_low_confidence(top_score, self.cfg.rerank.gate_tau),
            timings_ms=timings,
        )
