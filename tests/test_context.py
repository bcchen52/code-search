"""Context assembly: selection under the budget, then merging and ordering."""

import pytest

from cqa.config import ContextConfig
from cqa.generate.context import assemble, fit_budget, merge_overlapping, order_ends

from helpers import chunk, scored

RANK = ContextConfig(budget_tokens=12000, order="rank")
BY_FILE = ContextConfig(budget_tokens=12000, order="by_file_best_first")


def read_lines(path, start, end):
    return [f"line {i}" for i in range(start, end + 1)]


def test_rank_order_labels_follow_rank():
    chunks = {1: chunk(1, "a.py", 50, 60), 2: chunk(2, "b.py", 10, 20), 3: chunk(3, "a.py", 5, 15)}
    ctx = assemble(scored([1, 2, 3]), chunks, RANK, low_confidence=False)
    assert [c.id for c in ctx.chunks] == [1, 2, 3]
    assert ctx.token_count == 300 and ctx.low_confidence is False


def test_the_budget_is_a_prefix_rule():
    # 400 fits; 400 + 700 would exceed 1000, so selection stops there, even
    # though the 300-token chunk after it would still fit.
    chunks = {
        1: chunk(1, "a.py", 1, 40, tokens=400),
        2: chunk(2, "b.py", 1, 70, tokens=700),
        3: chunk(3, "c.py", 1, 30, tokens=300),
    }
    cfg = ContextConfig(budget_tokens=1000, order="rank")
    ctx = assemble(scored([1, 2, 3]), chunks, cfg, low_confidence=False)
    assert [c.id for c in ctx.chunks] == [1] and ctx.token_count == 400


def test_fit_budget_exact_fit_and_empty():
    two = [chunk(1, "a.py", 1, 5, tokens=500), chunk(2, "b.py", 1, 5, tokens=500)]
    assert [c.id for c in fit_budget(two, 1000)] == [1, 2]
    assert fit_budget([chunk(1, "a.py", 1, 5, tokens=2000)], 1000) == []


def test_chunks_are_never_truncated():
    c = chunk(1, "a.py", 1, 3, text="a\nb\nc")
    assert assemble(scored([1]), {1: c}, RANK, low_confidence=True).chunks[0].raw_text == "a\nb\nc"


def test_low_confidence_passes_through():
    assert assemble(scored([1]), {1: chunk(1, "a.py", 1, 3)}, RANK, low_confidence=True).low_confidence


def test_merging_orders_need_a_line_reader():
    with pytest.raises(ValueError):
        assemble(scored([1]), {1: chunk(1, "a.py", 1, 3)}, BY_FILE, low_confidence=False)


def test_by_file_best_first():
    chunks = {1: chunk(1, "a.py", 50, 60), 2: chunk(2, "b.py", 10, 20), 3: chunk(3, "a.py", 5, 15)}
    ctx = assemble(scored([1, 2, 3]), chunks, BY_FILE, low_confidence=False, read_lines=read_lines)
    assert [(c.path, c.start_line) for c in ctx.chunks] == [("a.py", 5), ("a.py", 50), ("b.py", 10)]


def test_selection_happens_before_ordering():
    # a.py holds ranks 1 and 3 and b.py rank 2. With room for two chunks,
    # b.py's chunk must be selected whatever the display order.
    chunks = {
        1: chunk(1, "a.py", 50, 60, tokens=500),
        2: chunk(2, "b.py", 10, 20, tokens=500),
        3: chunk(3, "a.py", 5, 15, tokens=500),
    }
    cfg = ContextConfig(budget_tokens=1000, order="by_file_best_first")
    ctx = assemble(scored([1, 2, 3]), chunks, cfg, low_confidence=False, read_lines=read_lines)
    assert sorted(c.id for c in ctx.chunks) == [1, 2]


def test_overlapping_chunks_merge():
    merged = merge_overlapping([chunk(1, "a.py", 10, 30), chunk(2, "a.py", 25, 40)], read_lines)
    assert [(c.start_line, c.end_line, c.kind) for c in merged] == [(10, 40, "merged")]
    assert merged[0].raw_text.splitlines()[0] == "line 10" and merged[0].id == 1


def test_skeletons_never_merge():
    skeleton = chunk(5, "a.py", 27, 136, kind="class_skeleton", citable=(27, 30))
    method = chunk(6, "a.py", 28, 35)
    assert sorted(c.id for c in merge_overlapping([skeleton, method], read_lines)) == [5, 6]


def test_order_ends_puts_the_best_at_both_ends():
    ranked = [chunk(i, f"f{i}.py", 1, 2) for i in range(1, 7)]
    assert [c.id for c in order_ends(ranked)] == [1, 3, 5, 6, 4, 2]
