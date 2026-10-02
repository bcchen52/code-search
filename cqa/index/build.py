"""Building and opening indexes.

An index is immutable and content-addressed: its id hashes the repository,
commit, stage versions, and index configuration, so building the same inputs
twice reuses the first index. Chunking is file-local, files record their git
blob SHA, and every query store is derived from the written chunks, which
keeps incremental re-indexing possible without changing the data model.
Chunk ids are stable within an index and meaningless across indexes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from cqa.config import IndexConfig
from cqa.ingest.walker import SkippedFile
from cqa.types import Chunk, Chunker, Embedder, SourceFile, VectorStore

OnStage = Callable[[str, float], None]
"""Progress callback: (stage name, fraction complete)."""


@dataclass
class IndexHandle:
    """A ready index, opened for queries."""

    index_id: str
    repo_name: str
    commit: str
    checkout: Path
    conn: sqlite3.Connection
    store: VectorStore | None

    def chunks(self, ids: Sequence[int]) -> dict[int, Chunk]:
        """Load chunks by id in one query."""
        raise NotImplementedError

    def file_lines(self, path: str, start: int, end: int) -> list[str]:
        """Return lines ``start`` through ``end`` of a file in this index, at the indexed commit.

        Raises:
            NotFoundError: If the path is not an indexed file; skipped files are
                not served. Paths are looked up in the index, never resolved
                against the filesystem, which rules out path traversal.
        """
        raise NotImplementedError


def index_identity(repo_url: str, commit: str, cfg: IndexConfig, stage_versions: str) -> str:
    """Return the index id: a short SHA-256 over the repository, commit, stage versions, and index config.

    ``stage_versions`` joins the version of every stage whose output the index
    stores. ``cfg.store`` is excluded because every vector store reads the
    same vectors.
    """
    raise NotImplementedError


def build_index(
    repo_url: str,
    commit: str,
    checkout: Path,
    cfg: IndexConfig,
    conn: sqlite3.Connection,
    data_dir: Path,
    chunker: Chunker,
    embedder: Embedder,
    on_stage: OnStage | None = None,
) -> str:
    """Build an index, or return the id of a ready index with the same identity.

    Walks the commit, records every file, chunks and enriches each file,
    writes the chunks, builds the configured stores, and publishes the index.

    A version left ``building`` or ``failed`` by an earlier attempt is
    rebuilt from scratch. Callers ensure one build of an index at a time; for
    the server, the job queue does, by keeping at most one active job per index.

    Returns:
        The index id.

    Raises:
        IndexBuildError: If any stage fails. The index version is marked
            failed; rebuilding reuses cached embeddings.
    """
    raise NotImplementedError


def write_files(
    conn: sqlite3.Connection, index_id: str, kept: list[SourceFile], skipped: list[SkippedFile]
) -> dict[str, int]:
    """Insert a row per file, skipped files included with their reason.

    Returns:
        File ids keyed by path.
    """
    raise NotImplementedError


def write_chunks(
    conn: sqlite3.Connection, index_id: str, file_ids: dict[str, int], chunks: list[Chunk]
) -> list[Chunk]:
    """Insert chunks and return them with their assigned ids."""
    raise NotImplementedError


def build_stores(
    conn: sqlite3.Connection,
    index_id: str,
    cfg: IndexConfig,
    data_dir: Path,
    embedder: Embedder,
    chunks: list[Chunk],
    files: list[SourceFile],
) -> None:
    """Build each store listed in ``cfg.stores`` from the written chunks.

    ``vectors`` embeds ``embed_text`` in document mode and writes the vector
    files.

    Raises:
        IndexBuildError: If the embedder rejects a chunk's ``embed_text`` as
            longer than its input limit.
    """
    raise NotImplementedError


def publish(conn: sqlite3.Connection, index_id: str, n_chunks: int) -> None:
    """Mark an index ready and record its chunk count, in one transaction."""
    raise NotImplementedError


def open_index(conn: sqlite3.Connection, index_id: str, data_dir: Path, store_name: str) -> IndexHandle:
    """Open a ready index with the named vector store.

    Raises:
        IndexNotReadyError: If the index does not exist or is not ready.
    """
    raise NotImplementedError


def index_dir(data_dir: Path, index_id: str) -> Path:
    """Return the directory holding an index's vector files: ``<data_dir>/indexes/<index_id>``."""
    raise NotImplementedError
