"""Dense retrieval, reranking, identifier extraction, and the confidence gate."""

import numpy as np
import pytest

from cqa.index.flat import FlatNumpyStore
from cqa.retrieve.dense import DenseRetriever
from cqa.retrieve.gate import is_low_confidence, tune_tau
from cqa.retrieve.query import extract_identifiers
from cqa.retrieve.rerank import NoopReranker

from helpers import FakeEmbedder, chunk


def test_dense_retriever_ranks_by_cosine():
    embedder = FakeEmbedder()
    texts = ["validate the session token", "parse the config file", "revoke a session"]
    store = FlatNumpyStore()
    store.build(embedder.embed(texts, "document"), np.array([1, 2, 3], dtype=np.uint64))
    got = DenseRetriever(embedder, store).retrieve("parse the config file", k=2)
    assert got[0].chunk_id == 2 and got[0].score == pytest.approx(1.0)
    assert [s.rank for s in got] == [1, 2] and {s.source for s in got} == {"dense"}


def test_noop_reranker_keeps_the_order_it_was_given():
    candidates = [chunk(5, "a.py", 1, 2), chunk(3, "a.py", 3, 4), chunk(9, "b.py", 1, 2)]
    out = NoopReranker().rerank("q", candidates, keep=2)
    assert [s.chunk_id for s in out] == [5, 3]
    assert [s.rank for s in out] == [1, 2] and {s.source for s in out} == {"rerank"}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("What calls `validate_token`?", ["validate_token"]),
        ("Where is MAX_CONN read in server.c?", ["MAX_CONN", "server.c"]),
        ("What breaks if emitLabel gains a parameter?", ["emitLabel"]),
        ("Trace auth.session.validate to its callers", ["auth.session.validate"]),
        ("Where is the session token validated?", []),
        ("Is OAuth supported?", []),
        ("Where is `HTTPServer` created?", ["HTTPServer"]),
    ],
)
def test_extract_identifiers(text, expected):
    assert extract_identifiers(text) == expected


def test_gate():
    assert is_low_confidence(0.2, tau=0.5)
    assert not is_low_confidence(0.7, tau=0.5)
    assert not is_low_confidence(0.2, tau=None)
    assert not is_low_confidence(None, tau=0.5)


def test_tune_tau_separates_a_clean_split():
    tau, f1 = tune_tau(answerable_scores=[2.0, 3.1, 1.5], unanswerable_scores=[-1.0, 0.2])
    assert 0.2 < tau < 1.5 and f1 == 1.0
