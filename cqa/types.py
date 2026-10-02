"""Core data types and the interfaces implemented by each pipeline stage.

Dataclasses describe the data passed between stages. Protocols define the
stages; configuration selects an implementation for each one. Line numbers
are 1-based and inclusive throughout.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

UNASSIGNED = -1
"""Value of ``Chunk.id`` until the chunk is written to an index."""


@dataclass(frozen=True)
class SourceFile:
    """A file from a repository at a specific commit.

    Attributes:
        path: Repository-relative path with forward slashes.
        language: Language or format name, such as ``python`` or ``markdown``.
        text: Decoded file contents.
        content_hash: Git blob SHA-1 of the file contents.
    """

    path: str
    language: str
    text: str
    content_hash: str


@dataclass(frozen=True)
class Chunk:
    """A contiguous line range of one file, indexed and retrieved as a unit.

    Attributes:
        id: Assigned when the chunk is written; unique and stable within one index only.
        path: Repository-relative path of the source file.
        start_line: First line covered.
        end_line: Last line covered.
        citable: Lines a citation may reference. Equals ``(start_line, end_line)``
            except for class skeletons, which may only be cited by their header lines.
        kind: One of ``function``, ``method``, ``function_part``, ``class``,
            ``class_skeleton``, ``module_preamble``, ``doc_section``,
            ``config_block``, or ``window``; context assembly adds ``merged``,
            and evaluation adds ``gold`` for labeled evidence spans.
        symbol: Qualified name when the chunk is a single definition,
            such as ``SessionManager.validate_token``.
        raw_text: Source text as shown to the model and the user.
        embed_text: Text sent to the embedder: ``raw_text`` plus a context header.
        lexical_text: Subtokenized body for BM25; empty when no lexical index is built.
        token_count: Number of tokens in ``raw_text``.
    """

    id: int
    path: str
    start_line: int
    end_line: int
    citable: tuple[int, int]
    kind: str
    symbol: str | None
    raw_text: str
    embed_text: str
    lexical_text: str
    token_count: int


@dataclass(frozen=True)
class Scored:
    """One entry in a ranked list.

    Attributes:
        chunk_id: The ranked chunk.
        score: Source-specific score; higher is better in every list.
        rank: Position in the list, starting at 1.
        source: ``dense``, ``bm25``, ``symbol``, ``fused``, or ``rerank``.
    """

    chunk_id: int
    score: float
    rank: int
    source: str


@dataclass
class Context:
    """The chunks shown to the generator, in prompt order.

    Attributes:
        chunks: Chunks in prompt order; label ``Cn`` refers to ``chunks[n - 1]``.
        token_count: Sum of the token counts of the chunks selected, before any merging.
        low_confidence: True when retrieval confidence fell below the gate threshold.
    """

    chunks: list[Chunk]
    token_count: int
    low_confidence: bool


@dataclass(frozen=True)
class Usage:
    """Token accounting for one model call.

    Attributes:
        tokens_in: Input tokens billed at the normal rate.
        tokens_out: Output tokens.
        cache_read_tokens: Input tokens read from the provider's prompt cache.
        cache_write_tokens: Input tokens written to the provider's prompt cache.
        cached: True when the response came from the local response cache,
            so the call cost nothing.
    """

    tokens_in: int
    tokens_out: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cached: bool = False


class Chunker(Protocol):
    """Splits one file into chunks.

    Output must depend only on the file and the chunker's parameters, never on
    other files. ``version`` must change whenever output changes for the same input.
    """

    version: str

    def chunk(self, f: SourceFile) -> list[Chunk]: ...


class Embedder(Protocol):
    """Maps texts to L2-normalized float32 vectors of shape ``(len(texts), dims)``."""

    model_id: str
    dims: int

    def embed(self, texts: Sequence[str], mode: Literal["query", "document"]) -> np.ndarray: ...


class VectorStore(Protocol):
    """Top-k search over L2-normalized vectors; scores are cosine similarities."""

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> None: ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]: ...


class Retriever(Protocol):
    """Ranks chunks for a question from one source."""

    def retrieve(self, question: str, k: int) -> list[Scored]: ...


class Reranker(Protocol):
    """Rescores candidates jointly with the question and keeps the best ``keep``."""

    def rerank(self, question: str, chunks: list[Chunk], keep: int) -> list[Scored]: ...


class Generation(Protocol):
    """One streaming model call.

    Iterating yields text deltas. ``prompt_hash`` identifies the rendered
    prompt from the start; ``usage`` is set once iteration completes. Each call
    has its own ``Generation``, so concurrent calls never share state.
    """

    prompt_hash: str
    usage: Usage | None

    def __iter__(self) -> Iterator[str]: ...


class Generator(Protocol):
    """Starts streaming model calls for a question and its context."""

    def stream(self, question: str, ctx: Context) -> Generation: ...
