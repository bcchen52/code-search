"""Fixed line windows, used as a baseline and when a file fails to parse."""

from cqa.chunking.base import split_lines
from cqa.chunking.fixed import FixedWindowChunker
from cqa.types import SourceFile


def numbered(n: int) -> SourceFile:
    text = "".join(f"line {i}\n" for i in range(1, n + 1))
    return SourceFile(path="x.py", language="python", text=text, content_hash="0" * 40)


def spans(n: int) -> list[tuple[int, int]]:
    return [(c.start_line, c.end_line) for c in FixedWindowChunker(40, 10).chunk(numbered(n))]


def test_split_lines_drops_only_the_final_newline():
    assert split_lines("a\nb\n") == ["a", "b"]
    assert split_lines("a\n\nb") == ["a", "", "b"]
    assert split_lines("") == []


def test_160_lines_make_five_windows():
    assert spans(160) == [(1, 40), (31, 70), (61, 100), (91, 130), (121, 160)]


def test_the_last_window_ends_at_the_last_line():
    assert spans(110) == [(1, 40), (31, 70), (61, 100), (91, 110)]
    assert spans(100) == [(1, 40), (31, 70), (61, 100)]


def test_short_and_empty_files():
    assert spans(12) == [(1, 12)]
    assert spans(0) == []


def test_windows_are_verbatim_and_fully_citable():
    chunks = FixedWindowChunker(40, 10).chunk(numbered(50))
    assert all(c.kind == "window" and c.symbol is None for c in chunks)
    assert all(c.citable == (c.start_line, c.end_line) for c in chunks)
    assert chunks[1].raw_text.splitlines()[0] == "line 31"
    assert all(c.token_count > 0 for c in chunks)


def test_chunking_is_deterministic():
    f = numbered(90)
    assert FixedWindowChunker(40, 10).chunk(f) == FixedWindowChunker(40, 10).chunk(f)
