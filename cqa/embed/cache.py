"""A content-addressed cache in front of any embedder.

Keys are ``sha256(model_id | dims | mode | text)``, so a re-index embeds only
chunks whose text changed. The mode is part of the key because query and
document prefixes give the same text different vectors. See
docs/decisions/D41-cache-key-mode.md.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from typing import Literal

import numpy as np

from cqa.embed.base import batched
from cqa.types import Embedder

LOOKUP_BATCH = 500
"""Keys per lookup query, well under SQLite's limit on bound parameters."""


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
        return hashlib.sha256(f"{model_id}|{dims}|{mode}|{text}".encode()).hexdigest()

    def embed(self, texts: Sequence[str], mode: Literal["query", "document"]) -> np.ndarray:
        """Embed texts, sending only cache misses to the inner embedder, in a single call.

        No call is made when every text is cached. New vectors are stored as
        float32 bytes. ``hits`` and ``misses`` count texts across calls.

        Returns:
            Vectors in input order.

        Raises:
            ValueError: If the inner embedder returns the wrong shape; nothing is cached then.
        """
        keys = [self.key(self.model_id, self.dims, mode, t) for t in texts]
        found = self._lookup(list(dict.fromkeys(keys)))
        hits = sum(k in found for k in keys)
        self.hits += hits
        self.misses += len(keys) - hits

        missing: dict[str, str] = {}
        for k, t in zip(keys, texts, strict=True):
            if k not in found:
                missing.setdefault(k, t)
        if missing:
            fresh = self.inner.embed(list(missing.values()), mode)
            if fresh.shape != (len(missing), self.dims):
                raise ValueError(f"expected {(len(missing), self.dims)} vectors, got {fresh.shape}")
            rows = [(k, v.astype("<f4").tobytes()) for k, v in zip(missing, fresh, strict=True)]
            insert = "INSERT OR REPLACE INTO embedding_cache (key, vector) VALUES (?, ?)"
            with self.conn:
                self.conn.executemany(insert, rows)
            found.update(zip(missing, fresh.astype(np.float32), strict=True))

        if not keys:
            return np.zeros((0, self.dims), dtype=np.float32)
        return np.stack([found[k] for k in keys]).astype(np.float32, copy=False)

    def _lookup(self, keys: list[str]) -> dict[str, np.ndarray]:
        """Fetch cached vectors by key, a batch of keys per query. Rows are read-only views."""
        out: dict[str, np.ndarray] = {}
        for batch in batched(keys, LOOKUP_BATCH):
            marks = ",".join("?" * len(batch))
            query = f"SELECT key, vector FROM embedding_cache WHERE key IN ({marks})"
            for row in self.conn.execute(query, list(batch)):
                out[row[0]] = np.frombuffer(row[1], dtype="<f4")
        return out
