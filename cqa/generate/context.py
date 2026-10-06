"""Context assembly: turning reranked chunks into the excerpts the model reads.

Chunks are selected in rank order until the token budget is spent, and only
then merged and arranged. The ordering option therefore changes where
evidence appears, never which evidence is included. Whole chunks are dropped
at the budget; none is truncated. See docs/decisions/D42-select-then-order.md.
"""

from __future__ import annotations

from collections.abc import Callable

from cqa.config import ContextConfig
from cqa.types import Chunk, Context, Scored

LineReader = Callable[[str, int, int], list[str]]
"""Reads lines ``start`` through ``end`` of a file: ``(path, start, end) -> lines``."""


def assemble(
    ranked: list[Scored],
    chunks: dict[int, Chunk],
    cfg: ContextConfig,
    low_confidence: bool,
    read_lines: LineReader | None = None,
) -> Context:
    """Build the context for one question.

    Selects chunks with ``fit_budget``, then arranges them by ``cfg.order``:

    - ``rank``: selection order, unmerged.
    - ``by_file_best_first``: overlapping chunks merged, then ``order_by_file``.
    - ``ends``: overlapping chunks merged, then ``order_ends``.

    Args:
        ranked: Reranked candidates, best first.
        chunks: The candidate chunks by id.
        cfg: Token budget and ordering.
        low_confidence: Whether the confidence gate fired; copied to the context.
        read_lines: Source reader; required by the orders that merge.

    Returns:
        The context. ``token_count`` is the sum of the selected chunks' token counts.

    Raises:
        ValueError: If a merging order is configured without ``read_lines``.
    """
    if cfg.order != "rank" and read_lines is None:
        raise ValueError(f"context order {cfg.order!r} merges chunks, so it needs read_lines")
    selected = fit_budget([chunks[s.chunk_id] for s in ranked], cfg.budget_tokens)
    ordered = selected
    if cfg.order != "rank" and read_lines is not None:
        merged = merge_overlapping(selected, read_lines)
        ordered = order_by_file(merged) if cfg.order == "by_file_best_first" else order_ends(merged)
    return Context(
        chunks=ordered, token_count=sum(c.token_count for c in selected), low_confidence=low_confidence
    )


def fit_budget(chunks_in_rank_order: list[Chunk], budget: int) -> list[Chunk]:
    """Return the longest prefix whose token counts sum to at most ``budget``.

    Evaluation measures recall at a token budget with the same rule, so the
    metric scores exactly the evidence the generator would see. Selection
    stops at the first chunk that does not fit; a smaller chunk after it is
    not considered, and no chunk is ever truncated.
    """
    selected: list[Chunk] = []
    total = 0
    for c in chunks_in_rank_order:
        if total + c.token_count > budget:
            break
        selected.append(c)
        total += c.token_count
    return selected


def merge_overlapping(chunks: list[Chunk], read_lines: LineReader) -> list[Chunk]:
    """Merge chunks of the same file whose citable spans overlap or touch.

    A merged chunk spans its members, keeps the id of its best-ranked member,
    has kind ``merged``, and takes its text from ``read_lines``. Class
    skeletons are never merged, because their text is not contiguous source.

    Args:
        chunks: Selected chunks in rank order.
        read_lines: Source reader for merged spans.

    Returns:
        The chunks after merging, in the rank order of each group's best member.
    """
    raise NotImplementedError


def order_by_file(chunks_in_rank_order: list[Chunk]) -> list[Chunk]:
    """Group chunks by file, order files by their best-ranked chunk, and order each file's chunks by line."""
    raise NotImplementedError


def order_ends(chunks_in_rank_order: list[Chunk]) -> list[Chunk]:
    """Place the strongest chunks at both ends and the weakest in the middle.

    Models use the middle of a long context least. For six chunks the result
    is ranks 1, 3, 5, 6, 4, 2.
    """
    raise NotImplementedError
