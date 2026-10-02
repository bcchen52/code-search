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
    raise NotImplementedError


def truncate(x: np.ndarray, dims: int) -> np.ndarray:
    """Keep the first ``dims`` columns and renormalize each row."""
    raise NotImplementedError


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield consecutive slices of at most ``size`` items."""
    raise NotImplementedError
