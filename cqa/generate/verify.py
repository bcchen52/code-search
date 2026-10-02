"""Mechanical citation checks, run on every answer.

Three checks need no model: each cited label exists, each narrowed range
falls inside the cited chunk's citable span, and each sentence that names an
identifier or a path carries a citation. Whether the cited lines actually
support a sentence takes judgment; that is measured during evaluation
(``cqa.eval.judge``), because judging every live answer would roughly double
cost and latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cqa.generate.citations import Citation
from cqa.types import Chunk


@dataclass(frozen=True)
class CitationCheck:
    """The verification result for one citation.

    Attributes:
        citation: The parsed citation.
        status: ``valid``, ``fabricated`` (the label does not exist), or
            ``out_of_range`` (the lines fall outside the citable span).
        path: The cited file, or None when fabricated.
        lines: The resolved line range, or None when fabricated.
    """

    citation: Citation
    status: Literal["valid", "fabricated", "out_of_range"]
    path: str | None
    lines: tuple[int, int] | None


@dataclass
class Verification:
    """The verification result for one answer."""

    checks: list[CitationCheck]
    uncited_sentences: list[int]

    @property
    def validity(self) -> float | None:
        """Share of citations that are valid, or None when the answer has no citations."""
        raise NotImplementedError


def verify(answer: str, citations: list[Citation], labels: dict[str, Chunk]) -> Verification:
    """Check an answer's citations against the excerpts the model was shown.

    A whole-excerpt citation such as ``[C3]`` is valid when the label exists,
    and resolves to the chunk's citable span. A narrowed citation such as
    ``[C3:La-b]`` also requires ``citable_start <= a <= b <= citable_end``.

    Args:
        answer: The answer text.
        citations: Citations parsed from the answer.
        labels: The chunk behind each label shown to the model.
    """
    raise NotImplementedError


def names_code(sentence: str) -> bool:
    """Return True when a sentence names an identifier or a path.

    The test is ``cqa.retrieve.query.extract_identifiers``, the same one that
    decides whether a question triggers symbol lookup.
    """
    raise NotImplementedError
