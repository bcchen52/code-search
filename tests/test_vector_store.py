"""Vector stores and the vector files. Flat search is exact; every other store must agree with it."""

import numpy as np
import pytest

from cqa.config import StoreParams
from cqa.index.cpp_store import EXTENSION_MODULE
from cqa.index.flat import FlatNumpyStore
from cqa.index.vector_store import make_store, open_store, read_vectors, write_vectors

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


@pytest.mark.parametrize("name, module", [("hnswlib", "hnswlib"), ("cpp", EXTENSION_MODULE)])
def test_approximate_stores_agree_with_flat(name, module):
    pytest.importorskip(module)
    vectors, ids = unit_rows(2000, 64, seed=1), np.arange(2000, dtype=np.uint64)
    queries = unit_rows(50, 64, seed=2)
    flat, other = FlatNumpyStore(), make_store(name)
    flat.build(vectors, ids)
    other.build(vectors, ids)
    overlaps = [
        len({c for c, _ in flat.search(q, 10)} & {c for c, _ in other.search(q, 10)}) / 10 for q in queries
    ]
    assert np.mean(overlaps) >= 0.95


def test_make_store():
    assert isinstance(make_store("flat"), FlatNumpyStore)
    with pytest.raises(ValueError, match="not available yet"):
        make_store("cpp")


def hnsw(**params):
    from cqa.index.hnswlib_store import HnswlibStore

    return HnswlibStore(threads=1, **params)


def test_make_store_passes_the_configured_parameters():
    pytest.importorskip("hnswlib")
    store = make_store("hnswlib", StoreParams(M=8, ef_construction=50, ef_search=20))
    assert (store.M, store.ef_construction, store.ef_search) == (8, 50, 20)
    assert store.cache_name == "hnswlib-M8-efc50"


def test_hnswlib_scores_and_order_match_flat_on_a_small_store():
    pytest.importorskip("hnswlib")
    vectors, ids = unit_rows(200, 16, seed=3), np.arange(500, 700, dtype=np.uint64)
    query = unit_rows(1, 16, seed=4)[0]
    flat, store = FlatNumpyStore(), hnsw(ef_search=200)
    flat.build(vectors, ids)
    store.build(vectors, ids)
    got, want = store.search(query, 5), flat.search(query, 5)
    assert [c for c, _ in got] == [c for c, _ in want]
    assert [s for _, s in got] == pytest.approx([s for _, s in want], abs=1e-5)
    cid, score = got[0]
    assert type(cid) is int and type(score) is float


def test_hnswlib_edge_cases():
    pytest.importorskip("hnswlib")
    store = hnsw()
    assert store.search(np.ones(8, dtype=np.float32), 5) == []  # never built
    store.build(np.zeros((0, 8), dtype=np.float32), np.zeros(0, dtype=np.uint64))
    assert store.search(np.ones(8, dtype=np.float32), 5) == []
    store.build(unit_rows(3, 8, seed=0), np.arange(3, dtype=np.uint64))
    assert len(store.search(unit_rows(1, 8, seed=1)[0], 10)) == 3  # k beyond the store
    assert store.search(unit_rows(1, 8, seed=1)[0], 0) == []
    with pytest.raises(ValueError, match="shape"):
        store.search(np.ones(4, dtype=np.float32), 2)
    with pytest.raises(ValueError):
        store.build(unit_rows(3, 8, seed=0), np.arange(2, dtype=np.uint64))


def test_hnswlib_breaks_score_ties_by_chunk_id():
    pytest.importorskip("hnswlib")
    duplicate = np.array([0.6, 0.8], dtype=np.float32)
    vectors = np.stack([duplicate, [1, 0], duplicate, duplicate, [0, 1]]).astype(np.float32)
    store = hnsw()
    store.build(vectors, np.array([42, 7, 5, 19, 3], dtype=np.uint64))
    assert [cid for cid, _ in store.search(duplicate, 3)] == [5, 19, 42]


def test_a_saved_graph_loads_with_identical_results(tmp_path):
    pytest.importorskip("hnswlib")
    vectors, ids = unit_rows(500, 32, seed=5), np.arange(500, dtype=np.uint64)
    queries = unit_rows(10, 32, seed=6)
    store = hnsw()
    store.build(vectors, ids)
    store.save(tmp_path / "graph")
    loaded = hnsw()
    loaded.load(tmp_path / "graph", dims=32)
    assert loaded.count == 500
    assert [loaded.search(q, 10) for q in queries] == [store.search(q, 10) for q in queries]
    assert [p.name for p in (tmp_path / "graph").iterdir()] == ["graph.bin"]
    with pytest.raises(FileNotFoundError):
        hnsw().load(tmp_path / "missing", dims=32)


def test_open_store_builds_a_graph_once_then_loads_it(tmp_path, monkeypatch):
    pytest.importorskip("hnswlib")
    from cqa.index.hnswlib_store import HnswlibStore

    write_vectors(tmp_path, unit_rows(100, 16, seed=7), np.arange(100, dtype=np.uint64))
    params = StoreParams(M=8, ef_construction=40, ef_search=32)
    builds = []
    original = HnswlibStore.build

    def counted(self, vectors, ids):
        builds.append(len(ids))
        original(self, vectors, ids)

    monkeypatch.setattr(HnswlibStore, "build", counted)
    first = open_store("hnswlib", params, tmp_path, dims=16)
    graph = tmp_path / "ann" / "hnswlib-M8-efc40" / "graph.bin"
    assert builds == [100] and graph.is_file()
    second = open_store("hnswlib", params, tmp_path, dims=16)
    query = unit_rows(1, 16, seed=8)[0]
    assert builds == [100] and second.search(query, 5) == first.search(query, 5)
    open_store("hnswlib", params.model_copy(update={"ef_construction": 80}), tmp_path, dims=16)
    assert builds == [100, 100]  # a different graph shape is built, not loaded
    assert isinstance(open_store("flat", None, tmp_path, dims=16), FlatNumpyStore)
    assert not (tmp_path / "ann" / "flat").exists()


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
