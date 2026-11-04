"""Approximate nearest-neighbor search with hnswlib."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

GRAPH_FILE = "graph.bin"


class HnswlibStore:
    """An HNSW graph over inner-product space, which is cosine for normalized vectors.

    ``M`` and ``ef_construction`` shape the graph; ``ef_search`` is how many
    candidates a search keeps, trading speed for recall. ``threads`` builds
    the graph in parallel (-1 uses every core); a parallel build is not
    deterministic, so tests build with one thread.
    """

    def __init__(
        self, M: int = 16, ef_construction: int = 200, ef_search: int = 64, threads: int = -1
    ) -> None:
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.threads = threads
        self._index: Any = None
        self._count = 0
        self._dims = 0

    @property
    def cache_name(self) -> str:
        """Directory name for a saved graph; it names every parameter that shapes the graph."""
        return f"hnswlib-M{self.M}-efc{self.ef_construction}"

    @property
    def count(self) -> int:
        """Number of vectors in the graph."""
        return self._count

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Build the graph, labeling each vector with its chunk id.

        Raises:
            ValueError: If ``vectors`` is not 2-D or ``ids`` does not hold one id per row.
        """
        if vectors.ndim != 2 or ids.shape != (vectors.shape[0],):
            raise ValueError(f"need an (N, d) matrix and N ids, got {vectors.shape} and {ids.shape}")
        import hnswlib

        n, dims = vectors.shape
        index = hnswlib.Index(space="ip", dim=dims)
        index.init_index(max_elements=max(n, 1), ef_construction=self.ef_construction, M=self.M)
        index.set_num_threads(self.threads)
        if n:
            index.add_items(np.ascontiguousarray(vectors, dtype=np.float32), np.asarray(ids, dtype=np.uint64))
        self._index, self._count, self._dims = index, n, dims

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs, best first, with ``ef = max(ef_search, k)``.

        Scores are ``1 - distance``, the dot product. Results are ordered by
        score and then chunk id, as flat search orders them; ``k`` beyond the
        number of vectors returns them all.

        Raises:
            ValueError: If ``query`` is not a vector of the stored dimension.
        """
        if self._index is None or k <= 0 or self._count == 0:
            return []
        if query.shape != (self._dims,):
            raise ValueError(f"query has shape {query.shape}, expected ({self._dims},)")
        k = min(k, self._count)
        self._index.set_ef(max(self.ef_search, k))
        labels, distances = self._index.knn_query(np.asarray(query, dtype=np.float32).reshape(1, -1), k=k)
        pairs = [(int(c), float(1.0 - d)) for c, d in zip(labels[0], distances[0], strict=True)]
        return sorted(pairs, key=lambda p: (-p[1], p[0]))

    def save(self, ann_dir: Path) -> None:
        """Write the graph to ``ann_dir``, replacing any earlier file in one rename.

        Raises:
            ValueError: If the graph has not been built.
        """
        if self._index is None:
            raise ValueError("nothing to save: build the graph first")
        ann_dir.mkdir(parents=True, exist_ok=True)
        partial = ann_dir / f"{GRAPH_FILE}.{os.getpid()}.tmp"
        self._index.save_index(str(partial))
        os.replace(partial, ann_dir / GRAPH_FILE)

    def load(self, ann_dir: Path, dims: int) -> None:
        """Load a graph written by ``save``.

        Raises:
            FileNotFoundError: If ``ann_dir`` holds no graph.
        """
        path = ann_dir / GRAPH_FILE
        if not path.is_file():
            raise FileNotFoundError(f"no saved graph at {path}")
        import hnswlib

        index = hnswlib.Index(space="ip", dim=dims)
        index.load_index(str(path))
        index.set_num_threads(self.threads)
        self._index, self._count, self._dims = index, index.get_current_count(), dims
