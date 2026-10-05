"""Splitting files into chunks, and preparing chunk text for embedding and BM25."""

from __future__ import annotations

from cqa.chunking.fixed import FixedWindowChunker
from cqa.config import IndexConfig
from cqa.errors import ConfigError
from cqa.types import Chunker


def make_chunker(cfg: IndexConfig) -> Chunker:
    """Return the chunker selected by ``cfg.chunker``.

    ``fixed`` returns a ``FixedWindowChunker``; ``ast`` returns an ``AstChunker``,
    which itself falls back to fixed windows for files it cannot parse.

    Raises:
        ConfigError: If the chunker is not available; only ``fixed`` is, for now.
    """
    if cfg.chunker == "fixed":
        return FixedWindowChunker(cfg.fallback_window_lines, cfg.fallback_overlap_lines)
    raise ConfigError(f"index.chunker: the {cfg.chunker!r} chunker is not available yet")
