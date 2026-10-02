"""Vector files and the vector-store registry.

Each index stores ``vectors.f32`` (an N x d float32 matrix, row-major,
L2-normalized) and ``chunk_ids.u64`` (row i of the matrix is ``chunk_ids[i]``),
both little-endian with no header. Every store builds from these two files,
including the native one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cqa.types import VectorStore

VECTORS_FILE = "vectors.f32"
IDS_FILE = "chunk_ids.u64"


def write_vectors(index_dir: Path, vectors: np.ndarray, ids: np.ndarray) -> None:
    """Write the vector and id files for an index."""
    raise NotImplementedError


def read_vectors(index_dir: Path, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Memory-map an index's vectors, shape ``(N, dims)`` float32, and ids, shape ``(N,)`` uint64."""
    raise NotImplementedError


def make_store(name: str) -> VectorStore:
    """Return an empty store: ``flat``, ``hnswlib``, or ``cpp``.

    Optional backends are imported only when selected.

    Raises:
        ValueError: If the store name is unknown.
    """
    raise NotImplementedError
