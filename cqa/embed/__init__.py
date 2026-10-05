"""Text embedding models and the embedding cache."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

from cqa.config import IndexConfig
from cqa.embed.cache import CachedEmbedder
from cqa.embed.local import LocalEmbedder
from cqa.errors import ConfigError
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
        revision: The model repository commit to load; required for local
            models, so the weights and any code they bring never change under
            an existing cache. See docs/decisions/D50-embedder-pins.md.
    """

    backend: Literal["local", "voyage"]
    provider_id: str
    dims: int
    query_prefix: str = ""
    document_prefix: str = ""
    trust_remote_code: bool = False
    revision: str | None = None


MODELS: dict[str, ModelSpec] = {
    "modernbert-embed-base": ModelSpec(
        "local",
        "nomic-ai/modernbert-embed-base",
        768,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        revision="d556a88e332558790b210f7bdbe87da2fa94a8d8",
    ),
    "qwen3-embedding-0.6b": ModelSpec(
        "local",
        "Qwen/Qwen3-Embedding-0.6B",
        1024,
        query_prefix=(
            "Instruct: Given a question about a code repository, retrieve the code that answers it\nQuery:"
        ),
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
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
        ConfigError: If the embedder is unknown, ``cfg.dims`` is not between 1
            and its native size, or its backend is not available yet (only
            ``local`` is, for now).
    """
    spec = MODELS.get(cfg.embedder)
    if spec is None:
        names = ", ".join(sorted(MODELS))
        raise ConfigError(f"index.embedder: unknown {cfg.embedder!r}; choose one of {names}")
    if not 0 < cfg.dims <= spec.dims:
        raise ConfigError(f"index.dims: {cfg.embedder} supports 1 to {spec.dims} dimensions, got {cfg.dims}")
    if spec.backend != "local":
        raise ConfigError(f"index.embedder: the {spec.backend!r} backend is not available yet")
    inner = LocalEmbedder(
        cfg.embedder,
        spec.provider_id,
        cfg.dims,
        query_prefix=spec.query_prefix,
        document_prefix=spec.document_prefix,
        trust_remote_code=spec.trust_remote_code,
        revision=spec.revision,
    )
    return CachedEmbedder(inner, cache) if cache is not None else inner
