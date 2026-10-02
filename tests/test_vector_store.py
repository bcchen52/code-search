"""Vector stores and the vector files. Flat search is exact."""

import numpy as np
import pytest

from cqa.index.flat import FlatNumpyStore
from cqa.index.vector_store import make_store, read_vectors, write_vectors

from helpers import unit_rows


def test_flat_hand_example():
    store = FlatNumpyStore()
    vectors = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    store.build(vectors, np.array([10, 20, 30], dtype=np.uint64))
    got = store.search(np.array([1, 0], dtype=np.float32), k=3)
    assert [cid for cid, _ in got] == [10, 30, 20]
    assert [round(score, 6) for _, score in got] == [1.0, 0.6, 0.0]


def test_flat_matches_brute_force():
    vectors, query = unit_rows(1000, 64, seed=0), unit_rows(1, 64, seed=1)[0]
    store = FlatNumpyStore()
    store.build(vectors, np.arange(1000, dtype=np.uint64))
    assert [cid for cid, _ in store.search(query, 10)] == list(np.argsort(-(vectors @ query))[:10])


def test_k_larger_than_the_corpus():
    store = FlatNumpyStore()
    store.build(unit_rows(3, 8, seed=0), np.arange(3, dtype=np.uint64))
    assert len(store.search(unit_rows(1, 8, seed=1)[0], 10)) == 3


def test_vector_files_round_trip(tmp_path):
    vectors, ids = unit_rows(50, 8, seed=0), np.arange(100, 150, dtype=np.uint64)
    write_vectors(tmp_path, vectors, ids)
    back, back_ids = read_vectors(tmp_path, dims=8)
    np.testing.assert_array_equal(vectors, back)
    np.testing.assert_array_equal(ids, back_ids)
    assert (tmp_path / "vectors.f32").stat().st_size == 50 * 8 * 4


def test_unknown_store_is_an_error():
    with pytest.raises(ValueError):
        make_store("faiss")
