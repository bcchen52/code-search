"""The content-addressed embedding cache."""

import hashlib

import numpy as np
import pytest

from cqa.embed.cache import CachedEmbedder

from helpers import FakeEmbedder


def test_key_format():
    assert CachedEmbedder.key("m", 384, "query", "x") == hashlib.sha256(b"m|384|query|x").hexdigest()


def test_a_repeat_call_is_served_from_the_cache(caches_db):
    inner = FakeEmbedder()
    emb = CachedEmbedder(inner, caches_db)
    first = emb.embed(["a", "b"], "document")
    second = emb.embed(["a", "b"], "document")
    assert inner.calls == [["a", "b"]]
    np.testing.assert_array_equal(first, second)
    assert (emb.hits, emb.misses) == (2, 2)


def test_only_misses_reach_the_model_and_order_is_kept(caches_db):
    inner = FakeEmbedder()
    emb = CachedEmbedder(inner, caches_db)
    emb.embed(["a", "b"], "document")
    out = emb.embed(["b", "c", "a"], "document")
    assert inner.calls[-1] == ["c"]
    np.testing.assert_array_equal(out, inner.embed(["b", "c", "a"], "document"))


def test_mode_is_part_of_the_key(caches_db):
    inner = FakeEmbedder()
    emb = CachedEmbedder(inner, caches_db)
    emb.embed(["validate token"], "query")
    emb.embed(["validate token"], "document")
    assert len(inner.calls) == 2


def test_rows_are_float32_with_the_right_shape(caches_db):
    out = CachedEmbedder(FakeEmbedder(dims=16), caches_db).embed(["a", "b", "c"], "document")
    assert out.shape == (3, 16) and out.dtype == np.float32


def test_the_cache_outlives_the_embedder_object(caches_db):
    first = CachedEmbedder(FakeEmbedder(), caches_db)
    first.embed(["a"], "document")
    inner = FakeEmbedder()
    CachedEmbedder(inner, caches_db).embed(["a"], "document")
    assert inner.calls == []


def test_duplicates_in_one_call_reach_the_model_once(caches_db):
    inner = FakeEmbedder()
    emb = CachedEmbedder(inner, caches_db)
    out = emb.embed(["a", "b", "a"], "document")
    assert inner.calls == [["a", "b"]]
    np.testing.assert_array_equal(out[0], out[2])
    assert (emb.hits, emb.misses) == (0, 3)


def test_lookups_beyond_one_query_batch(caches_db):
    texts = [f"t{i}" for i in range(1201)]
    inner = FakeEmbedder()
    emb = CachedEmbedder(inner, caches_db)
    first = emb.embed(texts, "document")
    second = emb.embed(texts, "document")
    assert len(inner.calls) == 1 and emb.hits == 1201
    np.testing.assert_array_equal(first, second)


def test_a_wrong_shape_from_the_model_is_an_error_and_nothing_is_cached(caches_db):
    class Short(FakeEmbedder):
        def embed(self, texts, mode):
            return super().embed(texts, mode)[:-1]

    with pytest.raises(ValueError, match="expected"):
        CachedEmbedder(Short(), caches_db).embed(["a", "b"], "document")
    assert caches_db.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0] == 0


def test_no_texts(caches_db):
    inner = FakeEmbedder(dims=16)
    out = CachedEmbedder(inner, caches_db).embed([], "query")
    assert out.shape == (0, 16) and inner.calls == []
