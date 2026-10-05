"""Vector helpers, the local embedder, and the embedder registry."""

import importlib.util
import os
import sys
import types

import numpy as np
import pytest

from cqa.config import load_config
from cqa.embed import MODELS, make_embedder
from cqa.embed.base import batched, l2_normalize, truncate
from cqa.embed.cache import CachedEmbedder
from cqa.embed.local import LocalEmbedder
from cqa.errors import ConfigError

from helpers import ROOT


class FakeTokenizer:
    """One token per whitespace-separated word, plus two special tokens."""

    def __call__(self, texts):
        return {"input_ids": [[0, *range(len(t.split())), 0] for t in texts]}


class FakeModel:
    max_seq_length = 8

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.encoded = []

    def encode(self, texts, batch_size, convert_to_numpy, show_progress_bar):
        self.encoded.append(list(texts))
        return np.array([[len(t), 1.0, 0.0, 2.0] for t in texts], dtype=np.float32)


def local(dims=4, model=None):
    emb = LocalEmbedder("fake", "org/fake", dims, query_prefix="q: ", document_prefix="d: ")
    emb._model = model or FakeModel()
    return emb


def test_l2_normalize_makes_unit_rows_and_leaves_zero_rows_at_zero():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])
    assert out.dtype == np.float32


def test_truncate_keeps_a_prefix_and_renormalizes():
    out = truncate(np.array([[3.0, 4.0, 12.0]], dtype=np.float32), 2)
    np.testing.assert_allclose(out, [[0.6, 0.8]])


def test_batched():
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(batched([], 3)) == []


def test_each_mode_gets_its_prefix():
    model = FakeModel()
    emb = local(model=model)
    emb.embed(["a b"], "query")
    emb.embed(["a b"], "document")
    assert model.encoded == [["q: a b"], ["d: a b"]]


def test_vectors_are_truncated_and_normalized():
    out = local(dims=2).embed(["x", "y z"], "document")
    assert out.shape == (2, 2) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)


def test_a_text_over_the_model_limit_is_an_error_never_a_truncation():
    model = FakeModel()
    with pytest.raises(ValueError, match="text 1 has 9 tokens"):
        local(model=model).embed(["short", "one two three four five six"], "document")
    assert model.encoded == []


def test_no_texts_means_no_model_load():
    emb = LocalEmbedder("fake", "org/fake", 4)
    assert emb.embed([], "query").shape == (0, 4)
    assert emb._model is None


def test_the_model_loads_once_at_its_pinned_revision(monkeypatch):
    loads = []

    def fake_sentence_transformer(name, revision, trust_remote_code):
        loads.append((name, revision, trust_remote_code))
        return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=fake_sentence_transformer),
    )
    emb = LocalEmbedder("fake", "org/fake", 4, revision="abc123")
    emb.embed(["a"], "query")
    emb.embed(["b"], "query")
    assert loads == [("org/fake", "abc123", False)]


def index_cfg(**changes):
    return load_config(ROOT / "configs/base.yaml").index.model_copy(update=changes)


def test_every_local_model_is_pinned_and_runs_no_remote_code():
    for name, spec in MODELS.items():
        if spec.backend == "local":
            assert spec.revision and len(spec.revision) == 40, name
            assert not spec.trust_remote_code, name


def test_make_embedder_uses_the_registry_name_and_the_cache(caches_db):
    emb = make_embedder(index_cfg(), caches_db)
    assert isinstance(emb, CachedEmbedder) and isinstance(emb.inner, LocalEmbedder)
    assert emb.model_id == "modernbert-embed-base" and emb.dims == 768
    assert emb.inner.revision == MODELS["modernbert-embed-base"].revision
    assert isinstance(make_embedder(index_cfg(), None), LocalEmbedder)


def test_make_embedder_truncates_to_fewer_dims():
    assert make_embedder(index_cfg(dims=256), None).dims == 256


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"embedder": "no-such-model"}, "choose one of"),
        ({"dims": 769}, "1 to 768"),
        ({"dims": 0}, "1 to 768"),
        ({"embedder": "voyage-4", "dims": 1024}, "not available yet"),
    ],
)
def test_make_embedder_rejects_bad_configs(changes, message):
    with pytest.raises(ConfigError, match=message):
        make_embedder(index_cfg(**changes), None)


@pytest.mark.skipif(
    not os.environ.get("CQA_SLOW_TESTS") or importlib.util.find_spec("sentence_transformers") is None,
    reason="downloads the real model: set CQA_SLOW_TESTS=1 with the [local] extra installed",
)
def test_the_real_baseline_model_ranks_the_right_function_first():
    emb = make_embedder(index_cfg(), None)
    docs = emb.embed(
        [
            "def validate_token(self, token):\n    session = self._decode(token)\n    if session.revoked:\n"
            "        raise AuthError",
            "def to_dict(self):\n    return {'id': self.id, 'name': self.name}",
        ],
        "document",
    )
    query = emb.embed(["where is the session token validated?"], "query")[0]
    assert docs[0] @ query > docs[1] @ query + 0.1
