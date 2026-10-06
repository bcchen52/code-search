"""Rerankers: rescoring fused candidates jointly with the question.

A cross-encoder reads the question and a chunk together, which ranks better
than comparing separate embeddings but costs one model pass per pair, so it
only sees the fused candidates. Its scores are model-specific, so any
threshold on them, such as the confidence gate's, is recalibrated whenever
the reranker changes.
"""

from __future__ import annotations

from typing import Any

from cqa.config import RerankConfig
from cqa.errors import ConfigError
from cqa.types import Chunk, Reranker, Scored


class NoopReranker:
    """Keeps candidates in their fused order."""

    def rerank(self, question: str, chunks: list[Chunk], keep: int) -> list[Scored]:
        """Return the first ``keep`` chunks in the given order with ``source="rerank"`` and score 0.0."""
        return [
            Scored(chunk_id=c.id, score=0.0, rank=i + 1, source="rerank") for i, c in enumerate(chunks[:keep])
        ]


class CrossEncoderReranker:
    """A sentence-transformers cross-encoder, loaded on first use."""

    def __init__(self, model_id: str, batch_size: int = 40) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self._model: Any = None

    def rerank(self, question: str, chunks: list[Chunk], keep: int) -> list[Scored]:
        """Score ``(question, embed_text)`` pairs and return the top ``keep``, best first."""
        raise NotImplementedError


class LlmReranker:
    """Asks a language model to order the candidates."""

    def __init__(self, model: str) -> None:
        self.model = model

    def rerank(self, question: str, chunks: list[Chunk], keep: int) -> list[Scored]:
        """Return the top ``keep`` chunks in the order the model ranks them."""
        raise NotImplementedError


def make_reranker(cfg: RerankConfig) -> Reranker:
    """Return the reranker selected by ``cfg.model``: ``none``, ``cross-encoder``, or ``llm``.

    Raises:
        ConfigError: If a model-backed reranker has no ``model_id``, or is not
            available yet (only ``none`` is, for now).
    """
    if cfg.model == "none":
        return NoopReranker()
    if cfg.model_id is None:
        raise ConfigError(f"rerank.model_id: the {cfg.model!r} reranker needs a model id")
    raise ConfigError(f"rerank.model: the {cfg.model!r} reranker is not available yet")
