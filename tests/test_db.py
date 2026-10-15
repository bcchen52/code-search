"""SQLite connections, the schema, and the data directory."""

import sqlite3

import pytest

from cqa.db import connect, data_dir, init_schema


def table_names(conn: sqlite3.Connection) -> set[str]:
    return {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def test_connect_sets_wal_foreign_keys_and_rows(tmp_path):
    conn = connect(tmp_path / "cqa.sqlite")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.row_factory is sqlite3.Row


def test_init_schema_is_safe_to_repeat(main_db, caches_db):
    init_schema(main_db, "main")
    init_schema(caches_db, "caches")
    assert {"repos", "index_versions", "files", "chunks"} <= table_names(main_db)
    assert table_names(caches_db) == {"embedding_cache", "llm_cache"}


def test_foreign_keys_are_enforced(main_db):
    with pytest.raises(sqlite3.IntegrityError):
        main_db.execute(
            "INSERT INTO files (index_id, path, content_hash, n_lines) VALUES (?, ?, ?, ?)",
            ("no-such-index", "a.py", "x", 1),
        )


def test_data_dir_honors_the_environment_and_creates_it(monkeypatch, tmp_path):
    monkeypatch.setenv("CQA_DATA_DIR", str(tmp_path / "store"))
    assert data_dir() == tmp_path / "store"
    assert (tmp_path / "store").is_dir()


def test_an_older_database_gains_the_new_columns(tmp_path):
    conn = connect(tmp_path / "old.sqlite")
    conn.execute(
        "CREATE TABLE queries (id INTEGER PRIMARY KEY, index_id TEXT NOT NULL, question TEXT NOT NULL, "
        "config_hash TEXT NOT NULL, answer TEXT, citations TEXT, timings_ms TEXT, tokens_in INTEGER, "
        "tokens_out INTEGER, cost_usd REAL, created_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO queries (index_id, question, config_hash, created_at) VALUES ('i', 'q', 'h', 't')"
    )
    init_schema(conn, "main")
    init_schema(conn, "main")
    columns = {r[1] for r in conn.execute("PRAGMA table_info(queries)")}
    assert {"trace", "model", "stop_reason"} <= columns
    assert conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 1
