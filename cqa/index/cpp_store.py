"""Adapter for a native C++ vector index loaded as a pybind11 extension module."""

from __future__ import annotations

from typing import Any

import numpy as np

EXTENSION_MODULE = "cqa_vdb"
"""Import name of the extension module that provides the native index."""


class CppStore:
    """Builds the native index from the shared vector files and maps results to chunk ids."""

    def __init__(self, **params: Any) -> None:
        self.params = params
        self._index: Any = None

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Import the extension and pass it C-contiguous arrays without copying.

        Raises:
            ImportError: If the extension module is not installed.
        """
        raise NotImplementedError

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs, best first."""
        raise NotImplementedError
