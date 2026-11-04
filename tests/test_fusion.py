"""Reciprocal Rank Fusion on a worked example."""

import pytest

from cqa.retrieve.fusion import rrf

from helpers import scored

A, B, C, D, E, F, G, H = range(1, 9)
LISTS = {"dense": scored([A, B, C, D, E], "dense"), "bm25": scored([C, F, G, H, B], "bm25")}


def test_worked_example():
    fused = rrf(LISTS, k=60)
    assert [s.chunk_id for s in fused[:3]] == [C, B, A]
    assert fused[0].score == pytest.approx(1 / 63 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 62 + 1 / 65)
    assert fused[2].score == pytest.approx(1 / 61)


def test_output_is_a_ranked_list():
    fused = rrf(LISTS, k=60)
    assert [s.rank for s in fused] == list(range(1, len(fused) + 1))
    assert {s.source for s in fused} == {"fused"}
    assert len({s.chunk_id for s in fused}) == len(fused) == 8


def test_ties_break_by_best_rank_then_chunk_id():
    # D (rank 4 in dense) and H (rank 4 in bm25) both score 1/64.
    assert [s.chunk_id for s in rrf(LISTS, k=60)] == [C, B, A, F, G, D, H, E]


def test_a_chunk_absent_from_a_list_gets_nothing_from_it():
    f = next(s for s in rrf(LISTS, k=60) if s.chunk_id == F)
    assert f.score == pytest.approx(1 / 62)


def test_weights_scale_a_list():
    lists = {"dense": scored([A, B], "dense"), "symbol": scored([C], "symbol")}
    fused = rrf(lists, k=60, weights={"symbol": 2})
    assert fused[0].chunk_id == C and fused[0].score == pytest.approx(2 / 61)


def test_weight_zero_removes_a_list():
    fused = rrf(LISTS, k=60, weights={"bm25": 0})
    assert [s.chunk_id for s in fused[:5]] == [A, B, C, D, E]


def test_limit():
    assert len(rrf(LISTS, k=60, limit=3)) == 3


def test_a_single_list_keeps_its_order():
    assert [s.chunk_id for s in rrf({"dense": scored([E, A, C], "dense")}, k=60)] == [E, A, C]
