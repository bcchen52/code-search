"""The confidence gate.

When the best rerank score falls below a threshold, the prompt tells the
model the evidence is weak and asks it to answer narrowly or abstain.
Generation still runs, because low-scoring contexts sometimes contain the
answer.
"""

from __future__ import annotations


def is_low_confidence(top_score: float | None, tau: float | None) -> bool:
    """Return True when the gate is enabled, a score exists, and ``top_score < tau``."""
    raise NotImplementedError


def tune_tau(answerable_scores: list[float], unanswerable_scores: list[float]) -> tuple[float, float]:
    """Choose the threshold that best separates unanswerable from answerable questions.

    Treats ``top_score < tau`` as a prediction of "unanswerable" and picks the
    candidate threshold, among midpoints between consecutive sorted scores,
    with the highest F1 for that prediction.

    Returns:
        ``(tau, f1)``.
    """
    raise NotImplementedError
