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


def test_empty_vector_files_round_trip(tmp_path):
    write_vectors(tmp_path, np.zeros((0, 8), dtype=np.float32), np.zeros(0, dtype=np.uint64))
    vectors, ids = read_vectors(tmp_path, dims=8)
    assert vectors.shape == (0, 8) and ids.shape == (0,)


def test_reading_at_the_wrong_dims_is_an_error(tmp_path):
    write_vectors(tmp_path, unit_rows(10, 8, seed=0), np.arange(10, dtype=np.uint64))
    with pytest.raises(ValueError):
        read_vectors(tmp_path, dims=4)
    with pytest.raises(ValueError):
        read_vectors(tmp_path, dims=3)


def test_vector_and_id_files_must_agree(tmp_path):
    write_vectors(tmp_path, unit_rows(10, 8, seed=0), np.arange(10, dtype=np.uint64))
    np.arange(9, dtype="<u8").tofile(tmp_path / "chunk_ids.u64")
    with pytest.raises(ValueError):
        read_vectors(tmp_path, dims=8)


def test_writing_needs_one_id_per_row(tmp_path):
    with pytest.raises(ValueError):
        write_vectors(tmp_path, unit_rows(3, 8, seed=0), np.arange(2, dtype=np.uint64))


def test_make_store():
    assert isinstance(make_store("flat"), FlatNumpyStore)
    for name in ("hnswlib", "cpp"):
        with pytest.raises(ValueError, match="not available yet"):
            make_store(name)


def test_build_does_not_copy_a_memory_mapped_matrix(tmp_path):
    write_vectors(tmp_path, unit_rows(20, 8, seed=0), np.arange(20, dtype=np.uint64))
    vectors, ids = read_vectors(tmp_path, dims=8)
    store = FlatNumpyStore()
    store.build(vectors, ids)
    assert store.vectors is vectors and isinstance(store.vectors, np.memmap)
    assert len(store.search(unit_rows(1, 8, seed=1)[0], 5)) == 5


def test_nothing_to_return():
    assert FlatNumpyStore().search(np.ones(8, dtype=np.float32), 5) == []
    store = FlatNumpyStore()
    store.build(np.zeros((0, 8), dtype=np.float32), np.zeros(0, dtype=np.uint64))
    assert store.search(np.ones(8, dtype=np.float32), 5) == []
    store.build(unit_rows(3, 8, seed=0), np.arange(3, dtype=np.uint64))
    assert store.search(unit_rows(1, 8, seed=1)[0], 0) == []


def test_a_query_of_the_wrong_dims_is_an_error():
    store = FlatNumpyStore()
    store.build(unit_rows(3, 8, seed=0), np.arange(3, dtype=np.uint64))
    with pytest.raises(ValueError, match="shape"):
        store.search(np.ones(4, dtype=np.float32), 2)


def test_equal_scores_are_ordered_by_chunk_id():
    duplicate = np.array([0.6, 0.8], dtype=np.float32)
    vectors = np.stack([duplicate, [1, 0], duplicate, duplicate, [0, 1]]).astype(np.float32)
    store = FlatNumpyStore()
    store.build(vectors, np.array([42, 7, 5, 19, 3], dtype=np.uint64))
    query = np.array([0.6, 0.8], dtype=np.float32)
    assert [cid for cid, _ in store.search(query, 2)] == [5, 19]
    assert [cid for cid, _ in store.search(query, 4)] == [5, 19, 42, 3]


def test_results_are_plain_python_numbers():
    store = FlatNumpyStore()
    store.build(unit_rows(3, 8, seed=0), np.arange(3, dtype=np.uint64))
    cid, score = store.search(unit_rows(1, 8, seed=1)[0], 1)[0]
    assert type(cid) is int and type(score) is float
