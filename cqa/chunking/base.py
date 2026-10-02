"""Line helpers shared by the chunkers."""

from __future__ import annotations

from cqa.types import Chunk, SourceFile


def split_lines(text: str) -> list[str]:
    """Split text into lines without newlines; line ``i`` is ``result[i - 1]``.

    A trailing newline does not produce an empty final line.

    Example:
        ``split_lines("a\\nb\\n")`` returns ``["a", "b"]``; ``split_lines("")`` returns ``[]``.
    """
    raise NotImplementedError


def line_text(lines: list[str], start: int, end: int) -> str:
    """Return lines ``start`` through ``end`` (1-based, inclusive) joined with newlines."""
    raise NotImplementedError


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
    raise NotImplementedError
