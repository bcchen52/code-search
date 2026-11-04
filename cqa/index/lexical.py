"""BM25 over chunks with SQLite FTS5.

Each index gets its own contentless FTS5 table, ``fts_<index_id>``, with
columns for symbol, path, and body tokens. Two details matter:

- The table is per index because ``bm25()`` computes document frequencies
  and average lengths over the whole table; a shared table would let other
  repositories shift this index's scores (docs/decisions/D37-per-index-fts-tables.md).
- The tokenizer is ``unicode61 tokenchars '_'`` because the default treats
  ``_`` as a separator and would split whole-identifier tokens such as
  ``max_retry_count`` (docs/decisions/D38-fts-tokenchars.md).

``bm25()`` returns lower-is-better values, so scores are negated.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable

from cqa.chunking.subtokenize import tokenize_code, tokenize_path
from cqa.types import Chunk

VERSION = 1
"""Increment when the table layout or tokenization changes; part of the index identity."""

COLUMNS = ("symbol_tokens", "path_tokens", "body_tokens")
TOKENIZER = "unicode61 tokenchars '_'"


class LexicalIndex:
    """The full-text table of one index."""

    def __init__(
        self, conn: sqlite3.Connection, index_id: str, weights: tuple[float, float, float] = (4.0, 2.0, 1.0)
    ) -> None:
        self.conn = conn
        self.index_id = index_id
        self.weights = weights

    @property
    def table(self) -> str:
        """The table name, ``fts_<index_id>``.

        Table names cannot be bound as SQL parameters, so the index id is
        validated as hexadecimal before it is formatted into SQL.

        Raises:
            ValueError: If the index id is not hexadecimal.
        """
        if not re.fullmatch(r"[0-9a-f]+", self.index_id):
            raise ValueError(f"index id must be hexadecimal, got {self.index_id!r}")
        return f"fts_{self.index_id}"

    def create(self) -> None:
        """Create the table if it does not exist."""
        with self.conn:
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.table} USING fts5("
                f"{', '.join(COLUMNS)}, content = '', tokenize = \"{TOKENIZER}\")"
            )

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Insert one row per chunk, keyed by chunk id, with its symbol, path, and body tokens."""
        rows = (
            (c.id, " ".join(tokenize_code(c.symbol or "")), " ".join(tokenize_path(c.path)), c.lexical_text)
            for c in chunks
        )
        with self.conn:
            self.conn.executemany(
                f"INSERT INTO {self.table} (rowid, {', '.join(COLUMNS)}) VALUES (?, ?, ?, ?)", rows
            )

    def drop(self) -> None:
        """Delete the table, if it exists."""
        with self.conn:
            self.conn.execute(f"DROP TABLE IF EXISTS {self.table}")

    def search(self, match: str, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs for an FTS5 MATCH expression, best first.

        Scores are ``-bm25(table, *weights)``; equal scores are ordered by chunk id.
        """
        if k <= 0:
            return []
        weights = ", ".join(str(float(w)) for w in self.weights)
        rows = self.conn.execute(
            f"SELECT rowid, bm25({self.table}, {weights}) AS s FROM {self.table} "
            f"WHERE {self.table} MATCH ? ORDER BY s, rowid LIMIT ?",
            (match, k),
        )
        return [(int(r[0]), -float(r[1])) for r in rows]
