"""Lexical retrieval with BM25 over code subtokens."""

from __future__ import annotations

from cqa.chunking.subtokenize import tokenize_code
from cqa.index.lexical import LexicalIndex
from cqa.types import Scored


class Bm25Retriever:
    """Ranks chunks by weighted BM25 over symbol, path, and body tokens."""

    def __init__(self, lexical: LexicalIndex) -> None:
        self.lexical = lexical

    def retrieve(self, question: str, k: int) -> list[Scored]:
        """Subtokenize the question like code and return the top ``k`` chunks with ``source="bm25"``.

        A question with no tokens returns an empty list.
        """
        tokens = list(dict.fromkeys(tokenize_code(question)))
        if not tokens or k <= 0:
            return []
        hits = self.lexical.search(to_match(tokens), k)
        return [Scored(cid, score, rank=i + 1, source="bm25") for i, (cid, score) in enumerate(hits)]


def to_match(tokens: list[str]) -> str:
    """Build an FTS5 MATCH expression that ORs the tokens.

    Each token is double-quoted, with embedded quotes doubled, so FTS5 syntax
    in user text cannot change the query's meaning.

    Example:
        ``to_match(["validate", "token"])`` returns ``'"validate" OR "token"'``.
    """
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)
