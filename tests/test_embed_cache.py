"""The content-addressed embedding cache."""

import hashlib

import numpy as np

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
