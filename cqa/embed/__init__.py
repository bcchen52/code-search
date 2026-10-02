"""Text embedding models and the embedding cache."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

from cqa.config import IndexConfig
from cqa.types import Embedder


@dataclass(frozen=True)
class ModelSpec:
    """An embedding model the index can use.

    Attributes:
        backend: ``local`` (sentence-transformers) or ``voyage`` (hosted API).
        provider_id: The model's id at its provider.
        dims: Native output dimensions.
        query_prefix: Text prepended to queries.
        document_prefix: Text prepended to documents.
        trust_remote_code: Whether the model's own loading code must run.
    """

    backend: Literal["local", "voyage"]
    provider_id: str
    dims: int
    query_prefix: str = ""
    document_prefix: str = ""
    trust_remote_code: bool = False


MODELS: dict[str, ModelSpec] = {
    "nomic-embed-text": ModelSpec(
        "local",
        "nomic-ai/nomic-embed-text-v1.5",
        768,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        trust_remote_code=True,
    ),
    "qwen3-embedding-0.6b": ModelSpec(
        "local",
        "Qwen/Qwen3-Embedding-0.6B",
        1024,
        query_prefix=(
            "Instruct: Given a question about a code repository, retrieve the code that answers it\nQuery:"
        ),
    ),
    "voyage-4": ModelSpec("voyage", "voyage-4", 1024),
}
"""Embedder names usable in ``index.embedder``."""


def make_embedder(cfg: IndexConfig, cache: sqlite3.Connection | None) -> Embedder:
    """Return the embedder named by ``cfg.embedder`` at ``cfg.dims``.

    With a cache connection, the embedder is wrapped in ``CachedEmbedder``.
    Requesting fewer dimensions than the model's native size truncates its
    vectors (Matryoshka truncation).

    Raises:
        ConfigError: If the embedder is unknown or ``cfg.dims`` exceeds its native size.
    """
    raise NotImplementedError
