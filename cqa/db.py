"""SQLite connections and schema.

Two databases live under the data directory: ``cqa.sqlite`` holds indexes,
jobs, queries, and evaluation runs; ``caches.sqlite`` holds the embedding and
response caches and is safe to delete. Both use write-ahead logging so the
API can read while the index worker writes.
"""

from __future__ import annotations

import os
import sqlite3
from importlib import resources
from pathlib import Path
from typing import Literal

SCHEMAS = {"main": "schema.sql", "caches": "schema_caches.sql"}
DB_FILES = {"main": "cqa.sqlite", "caches": "caches.sqlite"}
BUSY_TIMEOUT_MS = 5000


def data_dir() -> Path:
    """Return ``$CQA_DATA_DIR``, or ``./data`` when unset, creating it if needed."""
    path = Path(os.environ.get("CQA_DATA_DIR") or "data")
    path.mkdir(parents=True, exist_ok=True)
    return path


def connect(path: Path) -> sqlite3.Connection:
    """Open a database connection.

    The connection uses write-ahead logging, enforces foreign keys, returns
    rows as ``sqlite3.Row``, and waits up to ``BUSY_TIMEOUT_MS`` for locks.
    It may be used from several threads (retrievers run concurrently);
    callers serialize their own writes.
    """
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection, which: Literal["main", "caches"]) -> None:
    """Create any missing tables and indexes. Safe to call on an initialized database.

    Per-index full-text tables are created by ``cqa.index.lexical``, not here.
    """
    conn.executescript(resources.files("cqa").joinpath(SCHEMAS[which]).read_text())
