"""A content-addressed cache in front of any embedder.

Keys are ``sha256(model_id | dims | mode | text)``, so a re-index embeds only
chunks whose text changed. The mode is part of the key because query and
document prefixes give the same text different vectors. See
docs/decisions/D41-cache-key-mode.md.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Literal

import numpy as np

from cqa.types import Embedder


class CachedEmbedder:
    """Wraps an embedder with a SQLite-backed vector cache."""

    def __init__(self, inner: Embedder, conn: sqlite3.Connection) -> None:
        self.inner = inner
        self.conn = conn
        self.model_id = inner.model_id
        self.dims = inner.dims
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model_id: str, dims: int, mode: str, text: str) -> str:
        """Return the SHA-256 hex digest of ``"{model_id}|{dims}|{mode}|{text}"``."""
        raise NotImplementedError

    def embed(self, texts: Sequence[str], mode: Literal["query", "document"]) -> np.ndarray:
        """Embed texts, sending only cache misses to the inner embedder, in a single call.

        No call is made when every text is cached. New vectors are stored as
        float32 bytes. ``hits`` and ``misses`` count texts across calls.

        Returns:
            Vectors in input order.
        """
        raise NotImplementedError
