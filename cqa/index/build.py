"""Building and opening indexes.

An index is immutable and content-addressed: its id hashes the repository,
commit, stage versions, and index configuration, so building the same inputs
twice reuses the first index. Chunking is file-local, files record their git
blob SHA, and every query store is derived from the written chunks, which
keeps incremental re-indexing possible without changing the data model.
Chunk ids are stable within an index and meaningless across indexes.
"""

from __future__ import annotations

import functools
import hashlib
import shutil
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from cqa.chunking import enrich
from cqa.chunking.base import split_lines
from cqa.chunking.enrich import finalize
from cqa.config import IndexConfig, canonical_json
from cqa.embed.base import batched
from cqa.errors import ConfigError, IndexBuildError, IndexNotReadyError, NotFoundError
from cqa.index.vector_store import make_store, read_vectors, write_vectors
from cqa.ingest.walker import SkippedFile, read_blobs, resolve_commit, walk
from cqa.types import Chunk, Chunker, Embedder, SourceFile, VectorStore

OnStage = Callable[[str, float], None]
"""Progress callback: (stage name, fraction complete)."""

EMBED_BATCH = 512
"""Chunks embedded per call while building, so memory stays flat on large repositories."""

AVAILABLE_STORES = frozenset({"vectors"})
"""Stores this version can build; the others raise ``ConfigError``."""

_QUERY_BATCH = 500


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
        """Load chunks by id in one query per batch of ids.

        Ids that are not chunks of this index are left out of the result.
        """
        found: dict[int, Chunk] = {}
        for batch in batched(list(dict.fromkeys(ids)), _QUERY_BATCH):
            marks = ",".join("?" * len(batch))
            rows = self.conn.execute(
                "SELECT c.*, f.path FROM chunks c JOIN files f ON f.id = c.file_id "
                f"WHERE c.index_id = ? AND c.id IN ({marks})",
                [self.index_id, *batch],
            )
            for r in rows:
                found[r["id"]] = Chunk(
                    id=r["id"],
                    path=r["path"],
                    start_line=r["start_line"],
                    end_line=r["end_line"],
                    citable=(r["citable_start"], r["citable_end"]),
                    kind=r["kind"],
                    symbol=r["qualified_symbol"],
                    raw_text=r["raw_text"],
                    embed_text=r["embed_text"],
                    lexical_text=r["lexical_text"],
                    token_count=r["token_count"],
                )
        return found

    def file_lines(self, path: str, start: int, end: int) -> list[str]:
        """Return lines ``start`` through ``end`` of a file in this index, at the indexed commit.

        Lines are numbered as the chunkers number them; a range past the end
        of the file is cut at the last line.

        Raises:
            NotFoundError: If the path is not an indexed file; skipped files are
                not served. Paths are looked up in the index, never resolved
                against the filesystem, which rules out path traversal.
            ValueError: If ``start`` is below 1 or ``end`` is below ``start``.
        """
        if start < 1 or end < start:
            raise ValueError(f"invalid line range {start}-{end}")
        row = self.conn.execute(
            "SELECT content_hash FROM files WHERE index_id = ? AND path = ? AND skipped_reason IS NULL",
            (self.index_id, path),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"{path} is not an indexed file of index {self.index_id}")
        return list(_blob_lines(str(self.checkout), row["content_hash"])[start - 1 : end])


@functools.lru_cache(maxsize=256)
def _blob_lines(checkout: str, sha: str) -> tuple[str, ...]:
    """A file's lines read from git by blob SHA, decoded exactly as the walk decoded them."""
    data = read_blobs(Path(checkout), [sha])[sha]
    return tuple(split_lines(data.decode("utf-8", errors="replace")))


def stage_versions(chunker: Chunker, cfg: IndexConfig) -> str:
    """Join the version of every stage whose output the index stores.

    Always the chunker and enrichment; a stage behind an optional store counts
    only when that store is built, so adding a stage never renames existing
    indexes that do not use it.

    Raises:
        ConfigError: If ``cfg.stores`` names a store this version cannot build.
    """
    unavailable = sorted(set(cfg.stores) - AVAILABLE_STORES)
    if unavailable:
        raise ConfigError(f"index.stores: {', '.join(unavailable)} not available yet")
    return f"{chunker.version}+enrich-{enrich.VERSION}"


def resolve_repo(repo: str, commit: str | None) -> tuple[str, str]:
    """Return ``(repository URL, full commit SHA)`` for a local path or a URL.

    A local repository is named by its absolute path, so one repository
    always has one identity, and ``commit`` defaults to its ``HEAD``. A URL
    needs an explicit commit.

    Raises:
        ValueError: If a URL has no commit.
        IndexBuildError: If a local revision does not resolve to a commit.
    """
    local = Path(repo).expanduser()
    if local.is_dir():
        return str(local.resolve()), resolve_commit(local.resolve(), commit or "HEAD")
    if commit is None:
        raise ValueError(f"--commit is required for a remote repository: {repo}")
    return repo, commit


def checkout_dir(data_dir: Path, repo_url: str, commit: str) -> Path:
    """Return where a repository's commit is read from.

    A local repository is read in place. A remote one is fetched into
    ``<data_dir>/repos/<url hash>/<commit>``; the hash is known before the
    repository has a database row. See docs/decisions/D51-checkout-dir.md.
    """
    local = Path(repo_url).expanduser()
    if local.is_dir():
        return local
    key = hashlib.sha256(repo_url.encode()).hexdigest()[:12]
    return data_dir / "repos" / key / commit


def index_identity(repo_url: str, commit: str, cfg: IndexConfig, stage_versions: str) -> str:
    """Return the index id: a short SHA-256 over the repository, commit, stage versions, and index config.

    ``stage_versions`` joins the version of every stage whose output the index
    stores. ``cfg.store`` is excluded because every vector store reads the
    same vectors.
    """
    payload = {
        "repo": repo_url,
        "commit": commit,
        "stages": stage_versions,
        "index": cfg.model_dump(mode="json", exclude={"store"}),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]


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
        ConfigError: If a configured store is not available, or the embedder
            is not the one ``cfg`` names; nothing is written then.
        IndexBuildError: If any stage fails. The index version is marked
            failed; rebuilding reuses cached embeddings.
    """
    if (embedder.model_id, embedder.dims) != (cfg.embedder, cfg.dims):
        raise ConfigError(
            f"embedder {embedder.model_id} at {embedder.dims} dims does not match "
            f"index.embedder {cfg.embedder} at {cfg.dims}"
        )
    versions = stage_versions(chunker, cfg)
    index_id = index_identity(repo_url, commit, cfg, versions)
    row = conn.execute("SELECT status FROM index_versions WHERE id = ?", (index_id,)).fetchone()
    if row is not None and row["status"] == "ready":
        return index_id
    if row is not None:
        _discard(conn, index_id, data_dir)

    with conn:
        conn.execute(
            "INSERT INTO repos (url, name) VALUES (?, ?) ON CONFLICT (url) DO NOTHING",
            (repo_url, _repo_name(repo_url)),
        )
        repo_id = conn.execute("SELECT id FROM repos WHERE url = ?", (repo_url,)).fetchone()["id"]
        conn.execute(
            "INSERT INTO index_versions (id, repo_id, commit_sha, chunker_version, chunker_params, "
            "embedder_id, dims, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'building', ?)",
            (
                index_id,
                repo_id,
                commit,
                versions,
                canonical_json(cfg.model_dump(mode="json", exclude={"store"})),
                cfg.embedder,
                cfg.dims,
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )

    stage = "walk"
    try:
        _report(on_stage, stage, 0.0)
        kept, skipped = walk(checkout, commit)
        stage = "chunk"
        _report(on_stage, stage, 0.2)
        file_ids = write_files(conn, index_id, kept, skipped)
        chunks = [finalize(c, f, cfg) for f in kept for c in chunker.chunk(f)]
        chunks = write_chunks(conn, index_id, file_ids, chunks)
        stage = "stores"
        _report(on_stage, stage, 0.4)
        build_stores(conn, index_id, cfg, data_dir, embedder, chunks, kept)
        stage = "publish"
        _report(on_stage, stage, 0.95)
        publish(conn, index_id, len(chunks))
    except Exception as e:
        with conn:
            conn.execute("UPDATE index_versions SET status = 'failed' WHERE id = ?", (index_id,))
        raise IndexBuildError(f"building index {index_id} failed at {stage}: {e}") from e
    _report(on_stage, "ready", 1.0)
    return index_id


def _report(on_stage: OnStage | None, stage: str, fraction: float) -> None:
    if on_stage is not None:
        on_stage(stage, fraction)


def _repo_name(repo_url: str) -> str:
    """The last path component of a URL or path, without ``.git``."""
    return repo_url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git") or repo_url


def _discard(conn: sqlite3.Connection, index_id: str, data_dir: Path) -> None:
    """Delete everything a failed or abandoned build of ``index_id`` left behind."""
    with conn:
        for table in ("refs", "symbols", "chunks", "files", "index_versions"):
            column = "id" if table == "index_versions" else "index_id"
            conn.execute(f"DELETE FROM {table} WHERE {column} = ?", (index_id,))
    shutil.rmtree(index_dir(data_dir, index_id), ignore_errors=True)


def write_files(
    conn: sqlite3.Connection, index_id: str, kept: list[SourceFile], skipped: list[SkippedFile]
) -> dict[str, int]:
    """Insert a row per file, skipped files included with their reason.

    Kept files record their line count; skipped files are never read, so
    theirs is 0.

    Returns:
        File ids keyed by path.
    """
    rows: list[tuple[str, str, str | None, str, int, str | None]] = [
        (index_id, f.path, f.language, f.content_hash, len(split_lines(f.text)), None) for f in kept
    ]
    rows += [(index_id, s.path, None, s.content_hash, 0, s.reason) for s in skipped]
    with conn:
        conn.executemany(
            "INSERT INTO files (index_id, path, language, content_hash, n_lines, skipped_reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    rows_back = conn.execute("SELECT id, path FROM files WHERE index_id = ?", (index_id,))
    return {r["path"]: r["id"] for r in rows_back}


def write_chunks(
    conn: sqlite3.Connection, index_id: str, file_ids: dict[str, int], chunks: list[Chunk]
) -> list[Chunk]:
    """Insert chunks and return them with their assigned ids, in the same order."""
    written = []
    with conn:
        for c in chunks:
            cursor = conn.execute(
                "INSERT INTO chunks (index_id, file_id, start_line, end_line, citable_start, citable_end, "
                "kind, symbol, qualified_symbol, is_synthetic, raw_text, embed_text, lexical_text, "
                "token_count, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    index_id,
                    file_ids[c.path],
                    c.start_line,
                    c.end_line,
                    *c.citable,
                    c.kind,
                    c.symbol.rsplit(".", 1)[-1] if c.symbol else None,
                    c.symbol,
                    int(c.kind == "class_skeleton"),
                    c.raw_text,
                    c.embed_text,
                    c.lexical_text,
                    c.token_count,
                    hashlib.sha256(c.raw_text.encode()).hexdigest(),
                ),
            )
            if cursor.lastrowid is None:  # pragma: no cover - INSERT always sets it
                raise IndexBuildError("chunk insert returned no id")
            written.append(replace(c, id=cursor.lastrowid))
    return written


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

    Chunks are embedded ``EMBED_BATCH`` at a time.

    Raises:
        ConfigError: If ``cfg.stores`` names a store this version cannot build.
        IndexBuildError: If the embedder rejects a chunk's ``embed_text`` as
            longer than its input limit.
    """
    unavailable = sorted(set(cfg.stores) - AVAILABLE_STORES)
    if unavailable:
        raise ConfigError(f"index.stores: {', '.join(unavailable)} not available yet")
    if "vectors" in cfg.stores:
        parts = [np.zeros((0, cfg.dims), dtype=np.float32)]
        for batch in batched(chunks, EMBED_BATCH):
            try:
                parts.append(embedder.embed([c.embed_text for c in batch], "document"))
            except ValueError as e:
                raise IndexBuildError(f"embedding failed: {e}") from e
        target = index_dir(data_dir, index_id)
        target.mkdir(parents=True, exist_ok=True)
        write_vectors(target, np.concatenate(parts), np.array([c.id for c in chunks], dtype=np.uint64))


def publish(conn: sqlite3.Connection, index_id: str, n_chunks: int) -> None:
    """Mark an index ready and record its chunk count, in one transaction."""
    with conn:
        conn.execute(
            "UPDATE index_versions SET status = 'ready', n_chunks = ? WHERE id = ?", (n_chunks, index_id)
        )


def open_index(conn: sqlite3.Connection, index_id: str, data_dir: Path, store_name: str) -> IndexHandle:
    """Open a ready index with the named vector store.

    Raises:
        IndexNotReadyError: If the index does not exist or is not ready.
    """
    row = conn.execute(
        "SELECT v.status, v.commit_sha, v.dims, r.url, r.name FROM index_versions v "
        "JOIN repos r ON r.id = v.repo_id WHERE v.id = ?",
        (index_id,),
    ).fetchone()
    if row is None or row["status"] != "ready":
        state = "does not exist" if row is None else f"is {row['status']}"
        raise IndexNotReadyError(f"index {index_id} {state}")
    store = make_store(store_name)
    store.build(*read_vectors(index_dir(data_dir, index_id), row["dims"]))
    return IndexHandle(
        index_id=index_id,
        repo_name=row["name"],
        commit=row["commit_sha"],
        checkout=checkout_dir(data_dir, row["url"], row["commit_sha"]),
        conn=conn,
        store=store,
    )


def index_dir(data_dir: Path, index_id: str) -> Path:
    """Return the directory holding an index's vector files: ``<data_dir>/indexes/<index_id>``."""
    return data_dir / "indexes" / index_id
