"""Helpers shared by embedders."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Scale each row to unit length as float32; all-zero rows stay zero.

    Normalized vectors make cosine similarity a dot product.
    """
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    unit: np.ndarray = (x / np.maximum(norms, 1e-12)).astype(np.float32)
    return unit


def truncate(x: np.ndarray, dims: int) -> np.ndarray:
    """Keep the first ``dims`` columns and renormalize each row."""
    return l2_normalize(x[:, :dims])


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield consecutive slices of at most ``size`` items."""
    for i in range(0, len(items), size):
        yield items[i : i + size]
