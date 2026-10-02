"""Parsing citations out of answers.

The model cites excerpts as ``[C3]`` or, narrowed to lines, ``[C3:L120-128]``.
The agent cites the evidence it read the same way with an ``E`` prefix, so
one parser serves both. Each citation belongs to the sentence it ends.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CITATION = re.compile(r"\[([CE])(\d+)(?::L(\d+)-(\d+))?\]")
"""A well-formed citation: groups are prefix, number, and the optional start and end lines."""

CITATION_ATTEMPT = re.compile(r"\[[CcEe]\d+[^\]\n]{0,20}\]|\[[CcEe]\]")
"""Anything that looks like an attempted citation, well-formed or not."""


@dataclass(frozen=True)
class Citation:
    """One citation in an answer.

    Attributes:
        label: The cited excerpt, such as ``C3``.
        lines: The narrowed line range, or None when the whole excerpt is cited.
        sentence: Index into ``split_sentences(answer)``.
        start: Character offset of the citation token in the answer.
        end: Character offset just past the token.
    """

    label: str
    lines: tuple[int, int] | None
    sentence: int
    start: int
    end: int


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return the character spans of the sentences in ``text``.

    A sentence ends at ``.``, ``!``, or ``?`` together with any citation
    tokens that follow it, when the next non-space character is uppercase, a
    backtick, or the end of the text. Text inside backticks never splits, a
    dot followed directly by a letter or digit (``settings.py``,
    ``auth.session.validate``) never ends a sentence, and neither does the
    final dot of the abbreviations ``e.g.`` and ``i.e.``.
    """
    raise NotImplementedError


def parse_citations(answer: str) -> list[Citation]:
    """Return every well-formed citation in order of appearance, each attached to its sentence."""
    raise NotImplementedError


def find_malformed(answer: str) -> list[str]:
    """Return citation attempts that break the grammar.

    Examples of malformed attempts: ``[C]``, ``[c2]``, ``[C3:95-103]``, and ``[C4:L10]``.
    """
    raise NotImplementedError
