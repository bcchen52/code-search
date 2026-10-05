"""Fixed line windows, used as a baseline and when a file fails to parse."""

import pytest

from cqa.chunking import make_chunker
from cqa.chunking.base import line_text, make_chunk, split_lines
from cqa.chunking.enrich import finalize
from cqa.chunking.fixed import FixedWindowChunker
from cqa.config import IndexConfig, load_config
from cqa.errors import ConfigError
from cqa.tokens import count_tokens
from cqa.types import UNASSIGNED, SourceFile

from helpers import ROOT


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


def source(text: str) -> SourceFile:
    return SourceFile(path="x.py", language="python", text=text, content_hash="0" * 40)


def test_only_newlines_separate_lines():
    assert split_lines("a\r\nb\r\n") == ["a\r", "b\r"]
    assert split_lines("a\x0cb\u2028c\nd") == ["a\x0cb\u2028c", "d"]


def test_line_text_is_one_based_and_inclusive():
    lines = ["one", "two", "three"]
    assert line_text(lines, 1, 1) == "one"
    assert line_text(lines, 2, 3) == "two\nthree"


def test_make_chunk_defaults():
    f = source("a = 1\nb = 2\nc = 3\n")
    c = make_chunk(f, split_lines(f.text), 2, 3, kind="window")
    assert (c.id, c.citable, c.symbol) == (UNASSIGNED, (2, 3), None)
    assert c.raw_text == c.embed_text == "b = 2\nc = 3"
    assert c.lexical_text == ""
    assert c.token_count == count_tokens("b = 2\nc = 3")


def test_make_chunk_keeps_synthetic_text_and_a_narrow_citable_span():
    f = source("class A:\n    def f(self):\n        pass\n")
    c = make_chunk(
        f, split_lines(f.text), 1, 3, kind="class_skeleton", citable=(1, 1), raw_text="class A: ..."
    )
    assert c.raw_text == "class A: ..." and c.citable == (1, 1)


def test_line_numbers_do_not_drift_on_form_feeds():
    f = source("a\x0cb\nline 2\nline 3\n")
    (c,) = FixedWindowChunker(40, 10).chunk(f)
    assert (c.start_line, c.end_line) == (1, 3)
    assert c.raw_text.split("\n")[1] == "line 2"


@pytest.mark.parametrize("window, overlap", [(10, 10), (10, 12), (10, -1), (0, 0)])
def test_a_window_must_advance(window, overlap):
    with pytest.raises(ValueError):
        FixedWindowChunker(window, overlap)


def index_cfg(**changes) -> IndexConfig:
    return load_config(ROOT / "configs/base.yaml").index.model_copy(update=changes)


def test_finalize_leaves_a_baseline_chunk_unchanged():
    f = numbered(50)
    c = FixedWindowChunker(40, 10).chunk(f)[0]
    assert finalize(c, f, index_cfg()) == c


def test_finalize_refuses_the_lexical_store_until_it_exists():
    f = numbered(5)
    c = FixedWindowChunker(40, 10).chunk(f)[0]
    with pytest.raises(ConfigError, match="lexical"):
        finalize(c, f, index_cfg(stores=["vectors", "lexical"]))


def test_make_chunker_uses_the_configured_windows():
    chunker = make_chunker(index_cfg(fallback_window_lines=20, fallback_overlap_lines=5))
    assert isinstance(chunker, FixedWindowChunker)
    assert (chunker.window_lines, chunker.overlap_lines) == (20, 5)


def test_make_chunker_refuses_a_chunker_that_does_not_exist_yet():
    with pytest.raises(ConfigError, match="ast"):
        make_chunker(index_cfg(chunker="ast"))
