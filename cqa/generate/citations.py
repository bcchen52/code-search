"""Parsing citations out of answers.

The model cites excerpts as ``[C3]`` or, narrowed to lines, ``[C3:L120-128]``.
The agent cites the evidence it read the same way with an ``E`` prefix, so
one parser serves both. Each citation belongs to the sentence it ends.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

CITATION = re.compile(r"\[([CE])(\d+)(?::L(\d+)-(\d+))?\]")
"""A well-formed citation: groups are prefix, number, and the optional start and end lines."""

CITATION_ATTEMPT = re.compile(r"\[[CcEe]\d+[^\]\n]{0,20}\]|\[[CcEe]\]")
"""Anything that looks like an attempted citation, well-formed or not."""

_TRAILING_CITATION = re.compile(r"\s*" + CITATION.pattern)
_ABBREVIATIONS = ("e.g.", "i.e.")


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

    Text after the last sentence end that is not only whitespace forms a final
    sentence of its own. A span starts at its first non-space character, except
    the first, which starts at 0.

    Example:
        ``split_sentences("It is read [C1]. Nothing else [C2].")`` returns
        ``[(0, 16), (17, 35)]``.
    """
    spans: list[tuple[int, int]] = []
    start, i, n, in_code = 0, 0, len(text), False
    while i < n:
        ch = text[i]
        if ch == "`":
            in_code = not in_code
        is_end = not in_code and ch in ".!?" and (i + 1 == n or text[i + 1].isspace())
        if is_end and ch == "." and text[max(0, i - 3) : i + 1].lower() in _ABBREVIATIONS:
            is_end = False
        if not is_end:
            i += 1
            continue
        end = i + 1
        while m := _TRAILING_CITATION.match(text, end):
            end = m.end()
        nxt = end
        while nxt < n and text[nxt].isspace():
            nxt += 1
        if nxt == n or text[nxt].isupper() or text[nxt] == "`":
            spans.append((start, end))
            start = i = nxt
        else:
            i += 1
    if text[start:].strip():
        spans.append((start, n))
    return spans


def parse_citations(answer: str) -> list[Citation]:
    """Return every well-formed citation in order of appearance, each attached to its sentence.

    Labels are normalized (``[C03]`` is ``C3``), so they match context labels.
    """
    starts = [s for s, _ in split_sentences(answer)]
    citations = []
    for m in CITATION.finditer(answer):
        lines = (int(m.group(3)), int(m.group(4))) if m.group(3) else None
        citations.append(
            Citation(
                label=f"{m.group(1)}{int(m.group(2))}",
                lines=lines,
                sentence=max(bisect.bisect_right(starts, m.start()) - 1, 0),
                start=m.start(),
                end=m.end(),
            )
        )
    return citations


def find_malformed(answer: str) -> list[str]:
    """Return citation attempts that break the grammar.

    Examples of malformed attempts: ``[C]``, ``[c2]``, ``[C3:95-103]``, and ``[C4:L10]``.
    """
    return [m.group(0) for m in CITATION_ATTEMPT.finditer(answer) if not CITATION.fullmatch(m.group(0))]
