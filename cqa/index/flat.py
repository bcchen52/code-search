"""Exact nearest-neighbor search by matrix-vector product."""

from __future__ import annotations

import numpy as np


class FlatNumpyStore:
    """Brute-force search. Exact, so it is also the reference for approximate stores."""

    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.ids: np.ndarray | None = None

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Keep references to the vectors and ids; memory-mapped arrays are not copied."""
        raise NotImplementedError

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs, best first.

        Scores are ``vectors @ query``. Only the top ``k`` are sorted, after an
        ``np.argpartition``.
        """
        raise NotImplementedError
