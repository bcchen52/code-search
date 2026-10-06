"""Exact nearest-neighbor search by matrix-vector product."""

from __future__ import annotations

import numpy as np


class FlatNumpyStore:
    """Brute-force search. Exact, so it is also the reference for approximate stores."""

    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.ids: np.ndarray | None = None

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Keep references to the vectors and ids; memory-mapped arrays are not copied.

        Raises:
            ValueError: If ``vectors`` is not 2-D or ``ids`` does not hold one id per row.
        """
        if vectors.ndim != 2 or ids.shape != (vectors.shape[0],):
            raise ValueError(f"need an (N, d) matrix and N ids, got {vectors.shape} and {ids.shape}")
        self.vectors, self.ids = vectors, ids

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs, best first.

        Scores are ``vectors @ query``. ``np.partition`` finds the ``k``-th best
        score without sorting every row; every row scoring at least that much
        becomes a candidate, so rows tied at the cutoff are all considered
        rather than one picked arbitrarily. Only the candidates are sorted, by
        score and then chunk id, and the first ``k`` are returned. A search
        therefore returns the same chunks every time, even across duplicated
        chunks.

        Example:
            Ids 42, 5, and 19 with identical vectors that tie at the cutoff
            for ``k=2`` all become candidates, and the result keeps 5 and 19.

        Raises:
            ValueError: If ``query`` is not a vector of the stored dimension.
        """
        if self.vectors is None or self.ids is None or k <= 0 or len(self.ids) == 0:
            return []
        if query.shape != (self.vectors.shape[1],):
            raise ValueError(f"query has shape {query.shape}, expected ({self.vectors.shape[1]},)")
        scores = np.asarray(self.vectors @ query.astype(np.float32, copy=False))
        k = min(k, len(scores))
        kth = np.partition(scores, len(scores) - k)[len(scores) - k]
        candidates = np.flatnonzero(scores >= kth)
        order = np.lexsort((self.ids[candidates], -scores[candidates]))[:k]
        return [(int(self.ids[i]), float(scores[i])) for i in candidates[order]]
