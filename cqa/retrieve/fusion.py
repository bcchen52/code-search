"""Reciprocal Rank Fusion.

A chunk's fused score is the sum, over the lists it appears in, of
``weight / (k + rank)``. Only ranks are used, so lists whose scores live on
unrelated scales (cosine, BM25, exact symbol matches) combine without
calibration.
"""

from __future__ import annotations

from cqa.types import Scored


def rrf(
    lists: dict[str, list[Scored]],
    k: int = 60,
    weights: dict[str, float] | None = None,
    limit: int = 40,
) -> list[Scored]:
    """Fuse ranked lists into one.

    Args:
        lists: Ranked lists keyed by source name.
        k: Rank offset; larger values reduce the advantage of top ranks.
        weights: Weight per source. Missing sources weigh 1.0; a weight of 0
            removes a source's contribution.
        limit: Maximum number of results.

    Returns:
        Up to ``limit`` results, best first, with ``source="fused"`` and ranks
        from 1. A chunk contributes only from lists it appears in. Ties break
        by the best rank the chunk holds in any list, then by chunk id.

    Raises:
        ValueError: If ``k`` or a weight is negative.

    Example:
        With ``k=60``, dense ``[A, B]`` and bm25 ``[B]``: B scores
        ``1/62 + 1/61`` and A ``1/61``, so the result is ``[B, A]``.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    weights = weights or {}
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for source, ranked in lists.items():
        weight = weights.get(source, 1.0)
        if weight < 0:
            raise ValueError(f"weight for {source!r} must be non-negative, got {weight}")
        if weight == 0:
            continue
        for s in ranked:
            scores[s.chunk_id] = scores.get(s.chunk_id, 0.0) + weight / (k + s.rank)
            best_rank[s.chunk_id] = min(best_rank.get(s.chunk_id, s.rank), s.rank)
    order = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], cid))[:limit]
    return [Scored(cid, scores[cid], rank=i + 1, source="fused") for i, cid in enumerate(order)]
