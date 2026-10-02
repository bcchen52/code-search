"""Splitting files into chunks, and preparing chunk text for embedding and BM25."""

from __future__ import annotations

from cqa.config import IndexConfig
from cqa.types import Chunker


def make_chunker(cfg: IndexConfig) -> Chunker:
    """Return the chunker selected by ``cfg.chunker``.

    ``fixed`` returns a ``FixedWindowChunker``; ``ast`` returns an ``AstChunker``,
    which itself falls back to fixed windows for files it cannot parse.
    """
    raise NotImplementedError
