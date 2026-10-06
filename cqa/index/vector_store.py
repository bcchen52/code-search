"""Vector files and the vector-store registry.

Each index stores ``vectors.f32`` (an N x d float32 matrix, row-major,
L2-normalized) and ``chunk_ids.u64`` (row i of the matrix is ``chunk_ids[i]``),
both little-endian with no header. Every store builds from these two files,
including the native one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cqa.index.flat import FlatNumpyStore
from cqa.types import VectorStore

VECTORS_FILE = "vectors.f32"
IDS_FILE = "chunk_ids.u64"


def write_vectors(index_dir: Path, vectors: np.ndarray, ids: np.ndarray) -> None:
    """Write the vector and id files for an index.

    Raises:
        ValueError: If ``vectors`` is not 2-D or ``ids`` does not hold one id per row.
    """
    if vectors.ndim != 2 or ids.shape != (vectors.shape[0],):
        raise ValueError(f"need an (N, d) matrix and N ids, got {vectors.shape} and {ids.shape}")
    np.ascontiguousarray(vectors, dtype="<f4").tofile(index_dir / VECTORS_FILE)
    np.ascontiguousarray(ids, dtype="<u8").tofile(index_dir / IDS_FILE)


def read_vectors(index_dir: Path, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Memory-map an index's vectors, shape ``(N, dims)`` float32, and ids, shape ``(N,)`` uint64.

    Both arrays are read-only. An index with no vectors gives empty arrays.

    Raises:
        ValueError: If the vector file is not a whole number of ``dims``-wide
            rows, or its row count differs from the number of ids.
    """
    vectors_path, ids_path = index_dir / VECTORS_FILE, index_dir / IDS_FILE
    n_values, n_ids = vectors_path.stat().st_size // 4, ids_path.stat().st_size // 8
    if n_values % dims or n_values // dims != n_ids:
        raise ValueError(f"{vectors_path} holds {n_values} values, not {n_ids} rows of {dims}")
    if n_ids == 0:
        return np.zeros((0, dims), dtype=np.float32), np.zeros(0, dtype=np.uint64)
    vectors = np.memmap(vectors_path, dtype="<f4", mode="r").reshape(n_ids, dims)
    ids = np.memmap(ids_path, dtype="<u8", mode="r")
    return vectors, ids


def make_store(name: str) -> VectorStore:
    """Return an empty store: ``flat``, ``hnswlib``, or ``cpp``.

    Optional backends are imported only when selected.

    Raises:
        ValueError: If the store name is unknown, or the store is not available
            yet (only ``flat`` is, for now).
    """
    if name == "flat":
        return FlatNumpyStore()
    if name in ("hnswlib", "cpp"):
        raise ValueError(f"index.store: the {name!r} store is not available yet")
    raise ValueError(f"index.store: unknown store {name!r}")
