"""Completing chunker output for embedding and BM25.

``raw_text`` is never changed: prompts, the UI, and citation line numbers
depend on it. ``embed_text`` gains a header with the naming context a bare
code body lacks::

    # file: src/auth/session.py (python)
    # symbol: SessionManager.validate_token  [method, lines 88-121]
    # in class: SessionManager
    # imports: .errors.AuthError, .models.User
"""

from __future__ import annotations

from cqa.config import IndexConfig
from cqa.types import Chunk, SourceFile

VERSION = 1
"""Increment when the header format changes; part of the index identity."""


def file_imports(f: SourceFile) -> list[str]:
    """Return the names a file imports, in file order, as dotted paths such as ``datetime.timedelta``.

    Relative imports keep their leading dots (``.models.User``), and
    ``__future__`` imports are left out. A file that fails to parse has no imports.
    """
    raise NotImplementedError


def header(chunk: Chunk, language: str, imports: list[str]) -> str:
    """Return the enrichment header for a chunk, one ``# key: value`` line per field.

    The ``imports`` line lists the file's imports whose final name occurs as
    an identifier in the chunk, in file order. Lines that do not apply are
    omitted: no ``in class`` line for module-level functions, no ``symbol``
    line for windows, and no ``imports`` line when the chunk uses none.
    """
    raise NotImplementedError


def finalize(chunk: Chunk, f: SourceFile, cfg: IndexConfig, imports: list[str] | None = None) -> Chunk:
    """Return the chunk with ``embed_text`` and ``lexical_text`` filled in.

    ``embed_text`` is the header followed by the chunk's current
    ``embed_text`` when ``cfg.enrich_header`` is set, and unchanged otherwise.
    ``lexical_text`` is the subtokenized ``raw_text`` when the ``lexical``
    store is built, and empty otherwise.

    Args:
        chunk: A chunk of ``f``.
        f: The chunk's source file.
        cfg: The index configuration.
        imports: ``file_imports(f)``, when the caller has computed it once for
            all of the file's chunks; computed here when omitted.
    """
    raise NotImplementedError
