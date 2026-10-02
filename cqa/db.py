"""SQLite connections and schema.

Two databases live under the data directory: ``cqa.sqlite`` holds indexes,
jobs, queries, and evaluation runs; ``caches.sqlite`` holds the embedding and
response caches and is safe to delete. Both use write-ahead logging so the
API can read while the index worker writes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

SCHEMAS = {"main": "schema.sql", "caches": "schema_caches.sql"}
DB_FILES = {"main": "cqa.sqlite", "caches": "caches.sqlite"}
BUSY_TIMEOUT_MS = 5000


def data_dir() -> Path:
    """Return ``$CQA_DATA_DIR``, or ``./data`` when unset, creating it if needed."""
    raise NotImplementedError


def connect(path: Path) -> sqlite3.Connection:
    """Open a database connection.

    The connection uses write-ahead logging, enforces foreign keys, returns
    rows as ``sqlite3.Row``, and waits up to ``BUSY_TIMEOUT_MS`` for locks.
    It may be used from several threads (retrievers run concurrently);
    callers serialize their own writes.
    """
    raise NotImplementedError


def init_schema(conn: sqlite3.Connection, which: Literal["main", "caches"]) -> None:
    """Create any missing tables and indexes. Safe to call on an initialized database.

    Per-index full-text tables are created by ``cqa.index.lexical``, not here.
    """
    raise NotImplementedError
