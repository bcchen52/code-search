"""Code-aware tokenization for BM25.

Identifiers are split on case changes and underscores, the whole identifier
is kept as well, and everything is lowercased. Digits stay attached to the
letters before them, as in ``utf8``, ``v2``, and ``int64``. Questions are
tokenized the same way as code.

====================  ======================================
Input                 Tokens
====================  ======================================
``parseHTTPHeader``   parse http header parsehttpheader
``MAX_RETRY_COUNT``   max retry count max_retry_count
``src/auth/session``  src auth session (paths drop the extension)
``"retry-after"``     retry after
====================  ======================================
"""

from __future__ import annotations

import re

VERSION = 1
"""Increment when output changes; part of the index identity."""

_WORD = re.compile(r"[A-Za-z0-9_]+")
_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+[0-9]*|[A-Z]+[0-9]*|[0-9]+")
"""An acronym before a capitalized word, a word with trailing digits, an acronym, or a number."""


def split_identifier(word: str) -> list[str]:
    """Return an identifier's lowercase parts, then the whole identifier if it has more than one part.

    Example:
        ``split_identifier("getUTF8String")`` returns
        ``["get", "utf8", "string", "getutf8string"]``;
        ``split_identifier("session")`` returns ``["session"]``.
    """
    parts = [m.lower() for piece in word.split("_") for m in _PART.findall(piece)]
    return [*parts, word.lower()] if len(parts) > 1 else parts


def tokenize_code(text: str) -> list[str]:
    """Tokenize code or a question.

    Each run of letters, digits, and underscores is expanded with
    ``split_identifier``; everything else separates tokens.
    """
    return [token for word in _WORD.findall(text) for token in split_identifier(word)]


def tokenize_path(path: str) -> list[str]:
    """Tokenize a file path: each directory and the file name without its extension, split as identifiers.

    Example:
        ``tokenize_path("src/auth/session.py")`` returns ``["src", "auth", "session"]``.
    """
    *folders, name = path.split("/")
    stem = name.rsplit(".", 1)[0] if "." in name.lstrip(".") else name
    return tokenize_code(" ".join([*folders, stem]))


def lexical_text(text: str) -> str:
    """Return ``tokenize_code(text)`` joined with spaces, for the full-text body column."""
    return " ".join(tokenize_code(text))
