"""Fixed-size line windows, used as a baseline and for files that cannot be parsed."""

from __future__ import annotations

from cqa.chunking.base import make_chunk, split_lines
from cqa.types import Chunk, SourceFile


class FixedWindowChunker:
    """Splits a file into overlapping windows of a fixed number of lines."""

    version = "fixed-1"

    def __init__(self, window_lines: int = 40, overlap_lines: int = 10) -> None:
        """Raises ValueError unless ``0 <= overlap_lines < window_lines``, so each window advances."""
        if not 0 <= overlap_lines < window_lines:
            got = f"overlap_lines={overlap_lines}, window_lines={window_lines}"
            raise ValueError(f"need 0 <= overlap_lines < window_lines, got {got}")
        self.window_lines = window_lines
        self.overlap_lines = overlap_lines

    def chunk(self, f: SourceFile) -> list[Chunk]:
        """Return windows of ``window_lines`` lines, each ``window_lines - overlap_lines`` after the previous.

        Windows have kind ``window`` and are citable in full. Windowing stops
        once a window reaches the last line, so no window lies entirely
        inside the previous one.

        Example:
            With the defaults, a 160-line file yields (1, 40), (31, 70),
            (61, 100), (91, 130), (121, 160); a 12-line file yields (1, 12);
            an empty file yields no chunks.
        """
        lines = split_lines(f.text)
        n, step = len(lines), self.window_lines - self.overlap_lines
        chunks: list[Chunk] = []
        start = 1
        while start <= n:
            end = min(start + self.window_lines - 1, n)
            chunks.append(make_chunk(f, lines, start, end, kind="window"))
            if end == n:
                break
            start += step
        return chunks
