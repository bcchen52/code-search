"""Line helpers shared by the chunkers."""

from __future__ import annotations

from cqa.tokens import count_tokens
from cqa.types import UNASSIGNED, Chunk, SourceFile


def split_lines(text: str) -> list[str]:
    """Split text into lines without newlines; line ``i`` is ``result[i - 1]``.

    A trailing newline does not produce an empty final line. Only ``\\n``
    separates lines, as in git and editors: unlike ``str.splitlines``, form
    feeds and Unicode line separators stay inside their line, and a CRLF file
    keeps each ``\\r``.

    Example:
        ``split_lines("a\\nb\\n")`` returns ``["a", "b"]``; ``split_lines("")`` returns ``[]``.
    """
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def line_text(lines: list[str], start: int, end: int) -> str:
    """Return lines ``start`` through ``end`` (1-based, inclusive) joined with newlines."""
    return "\n".join(lines[start - 1 : end])


def make_chunk(
    f: SourceFile,
    lines: list[str],
    start: int,
    end: int,
    kind: str,
    symbol: str | None = None,
    citable: tuple[int, int] | None = None,
    raw_text: str | None = None,
) -> Chunk:
    """Create an unassigned chunk with its token count.

    Args:
        f: The source file.
        lines: The file's lines, from ``split_lines``.
        start: First line of the chunk.
        end: Last line of the chunk.
        kind: The chunk kind.
        symbol: Qualified symbol name, when the chunk is one definition.
        citable: Citable line range; defaults to ``(start, end)``.
        raw_text: Chunk text; defaults to lines ``start`` through ``end``.
            Class skeletons pass their synthetic text.

    Returns:
        A chunk with ``id=UNASSIGNED``, ``embed_text`` equal to ``raw_text``, and
        empty ``lexical_text``; ``cqa.chunking.enrich.finalize`` completes both.
    """
    text = line_text(lines, start, end) if raw_text is None else raw_text
    return Chunk(
        id=UNASSIGNED,
        path=f.path,
        start_line=start,
        end_line=end,
        citable=citable or (start, end),
        kind=kind,
        symbol=symbol,
        raw_text=text,
        embed_text=text,
        lexical_text="",
        token_count=count_tokens(text),
    )
