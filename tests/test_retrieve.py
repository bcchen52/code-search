"""Dense retrieval, reranking, identifier extraction, and the confidence gate."""

import numpy as np
import pytest

from cqa.chunking.fixed import FixedWindowChunker
from cqa.config import load_config
from cqa.embed.cache import CachedEmbedder
from cqa.errors import ConfigError
from cqa.generate.context import assemble
from cqa.index.build import build_index, open_index
from cqa.index.flat import FlatNumpyStore
from cqa.retrieve.dense import DenseRetriever
from cqa.retrieve.gate import is_low_confidence, tune_tau
from cqa.retrieve.pipeline import RetrievalPipeline
from cqa.retrieve.query import extract_identifiers
from cqa.retrieve.rerank import NoopReranker, make_reranker

from helpers import ROOT, FakeEmbedder, chunk


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


def test_dense_retriever_edge_cases():
    embedder, store = FakeEmbedder(), FlatNumpyStore()
    store.build(np.zeros((0, 16), dtype=np.float32), np.zeros(0, dtype=np.uint64))
    assert DenseRetriever(embedder, store).retrieve("anything", k=5) == []
    assert DenseRetriever(embedder, store).retrieve("anything", k=0) == []
    assert embedder.calls == [["anything"]]


def test_noop_reranker_keeps_fewer_than_keep_and_nothing():
    assert [s.chunk_id for s in NoopReranker().rerank("q", [chunk(5, "a.py", 1, 2)], keep=8)] == [5]
    assert NoopReranker().rerank("q", [], keep=8) == []


def rerank_cfg(**changes):
    return load_config(ROOT / "configs/base.yaml").rerank.model_copy(update=changes)


def test_make_reranker():
    assert isinstance(make_reranker(rerank_cfg()), NoopReranker)
    with pytest.raises(ConfigError, match="needs a model id"):
        make_reranker(rerank_cfg(model="cross-encoder"))
    with pytest.raises(ConfigError, match="not available yet"):
        make_reranker(rerank_cfg(model="cross-encoder", model_id="some/cross-encoder"))


@pytest.fixture
def toy_index(toyrepo, main_db, caches_db, tmp_path):
    repo, sha = toyrepo
    embedder = CachedEmbedder(FakeEmbedder(), caches_db)
    base = load_config(ROOT / "configs/base.yaml")
    cfg = base.model_copy(update={"index": base.index.model_copy(update={"embedder": "fake", "dims": 16})})
    index_id = build_index(
        str(repo), sha, repo, cfg.index, main_db, tmp_path, FixedWindowChunker(40, 10), embedder
    )
    return cfg, open_index(main_db, index_id, tmp_path, "flat"), embedder


def with_changes(cfg, **sections):
    return cfg.model_copy(
        update={name: getattr(cfg, name).model_copy(update=v) for name, v in sections.items()}
    )


def test_the_pipeline_finds_a_chunk_by_its_own_text(toy_index):
    cfg, handle, embedder = toy_index
    target = next(c for c in handle.chunks(range(1, 200)).values() if c.path == "src/auth/session.py")
    result = RetrievalPipeline.from_config(cfg, handle, embedder).run(target.embed_text)
    assert result.lists["dense"][0].chunk_id == target.id
    assert result.fused[0].chunk_id == target.id and result.reranked[0].chunk_id == target.id
    assert {s.chunk_id for s in result.lists["dense"]} <= set(result.chunks)
    assert result.top_score is None and result.low_confidence is False
    assert {"dense", "load", "rerank", "total"} <= set(result.timings_ms)


def test_the_candidate_stage_is_cut_to_the_configured_depth(toy_index):
    cfg, handle, embedder = toy_index
    cfg = with_changes(cfg, retrieve={"dense_k": 10, "candidates": 4}, rerank={"keep": 2})
    result = RetrievalPipeline.from_config(cfg, handle, embedder).run("where is the session token validated?")
    assert len(result.lists["dense"]) == 10
    assert [s.chunk_id for s in result.fused] == [s.chunk_id for s in result.lists["dense"][:4]]
    assert [s.chunk_id for s in result.reranked] == [s.chunk_id for s in result.fused[:2]]
    assert [s.rank for s in result.reranked] == [1, 2]


def test_the_gate_stays_off_without_a_rerank_score(toy_index):
    cfg, handle, embedder = toy_index
    cfg = with_changes(cfg, rerank={"gate_tau": 0.5})
    result = RetrievalPipeline.from_config(cfg, handle, embedder).run("anything at all")
    assert result.top_score is None and result.low_confidence is False


def test_the_pipeline_feeds_context_assembly(toy_index):
    cfg, handle, embedder = toy_index
    result = RetrievalPipeline.from_config(cfg, handle, embedder).run("how are sessions revoked?")
    ctx = assemble(result.reranked, result.chunks, cfg.context, result.low_confidence)
    assert [c.id for c in ctx.chunks] == [s.chunk_id for s in result.reranked]
    assert ctx.token_count == sum(c.token_count for c in ctx.chunks) <= cfg.context.budget_tokens


@pytest.mark.parametrize(
    "sections, message",
    [
        ({"retrieve": {"lists": ["dense", "bm25"]}}, "bm25 not available yet"),
        ({"retrieve": {"lists": ["symbol"]}}, "symbol not available yet"),
        ({"rerank": {"model": "llm", "model_id": "some-model"}}, "not available yet"),
    ],
)
def test_unavailable_parts_are_config_errors(toy_index, sections, message):
    cfg, handle, embedder = toy_index
    with pytest.raises(ConfigError, match=message):
        RetrievalPipeline.from_config(with_changes(cfg, **sections), handle, embedder)
