"""The single answer path: event order, verification, outcomes, failures, and the command line."""

import json

import pytest
from typer.testing import CliRunner

from cqa.chunking.fixed import FixedWindowChunker
from cqa.cli import app
from cqa.config import config_hash, load_config
from cqa.embed.cache import CachedEmbedder
from cqa.errors import CqaError
from cqa.generate.answer import Answerer, build_answerer, sources_payload
from cqa.generate.llm import AnthropicGenerator, LlmCache
from cqa.index.build import build_index, open_index
from cqa.retrieve.pipeline import RetrievalPipeline
from cqa.trace import Trace
from cqa.types import Context, Usage

from helpers import ROOT, FakeEmbedder, chunk, git, scored

QUESTION = "Where is the session token validated?"


class FakeGeneration:
    def __init__(self, deltas, usage, fail=False):
        self.deltas, self.final, self.fail = deltas, usage, fail
        self.prompt_hash = "p" * 64
        self.usage = None

    def __iter__(self):
        yield from self.deltas
        if self.fail:
            raise RuntimeError("model unavailable")
        self.usage = self.final


class FakeGenerator:
    def __init__(
        self, deltas=("Tokens are checked in `validate_token` [C1]. ", "It raises on expiry [C9]."), **usage
    ):
        self.deltas = list(deltas)
        self.usage = Usage(
            **{
                "tokens_in": 900,
                "tokens_out": 40,
                "model": "claude-sonnet-5-5",
                "stop_reason": "end_turn",
                **usage,
            }
        )
        self.fail = False
        self.calls = []

    def stream(self, question, ctx):
        self.calls.append((question, ctx))
        return FakeGeneration(self.deltas, self.usage, self.fail)


@pytest.fixture
def toy(toyrepo, main_db, caches_db, tmp_path):
    repo, sha = toyrepo
    embedder = CachedEmbedder(FakeEmbedder(), caches_db)
    base = load_config(ROOT / "configs/base.yaml")
    cfg = base.model_copy(update={"index": base.index.model_copy(update={"embedder": "fake", "dims": 16})})
    index_id = build_index(
        str(repo), sha, repo, cfg.index, main_db, tmp_path, FixedWindowChunker(40, 10), embedder
    )
    handle = open_index(main_db, index_id, tmp_path, "flat")
    pipeline = RetrievalPipeline.from_config(cfg, handle, embedder)

    def make(generator=None):
        generator = generator or FakeGenerator()
        return Answerer(cfg, pipeline, generator, "toy", sha, config_hash(cfg)), generator

    return make


def test_events_arrive_in_order_and_sources_precede_the_model(toy):
    answerer, generator = toy()
    stream = answerer.stream(QUESTION)
    first = next(stream)
    assert first.type == "sources" and generator.calls == []
    rest = [e.type for e in stream]
    assert rest == ["token", "token", "citations", "done"]


def test_citations_are_verified_against_the_context(toy):
    answerer, _ = toy()
    events = {e.type: e for e in answerer.stream(QUESTION)}
    checks = events["citations"].data["citations"]
    assert [(c["label"], c["status"]) for c in checks] == [("C1", "valid"), ("C9", "fabricated")]
    assert events["citations"].data["uncited_sentences"] == []


def test_the_done_trace_is_complete_and_round_trips(toy):
    answerer, _ = toy()
    trace = answerer.run(QUESTION)
    assert trace.answer.startswith("Tokens are checked") and trace.stop_reason == "end_turn"
    assert trace.model == "claude-sonnet-5-5" and trace.prompt_hash == "p" * 64
    assert [e.label for e in trace.context] == [f"C{i}" for i in range(1, len(trace.context) + 1)]
    assert trace.lists["dense"] and trace.fused and trace.reranked
    assert {"dense", "load", "rerank", "retrieve", "context", "ttft", "generate", "verify", "total"} <= set(
        trace.timings_ms
    )
    assert (trace.tokens_in, trace.tokens_out, trace.cost_usd) == (900, 40, 0.0)
    assert Trace.from_json(trace.to_json()) == trace


def test_a_refusal_discards_the_streamed_text(toy):
    answerer, _ = toy(FakeGenerator(deltas=["Partial [C1]"], stop_reason="refusal"))
    events = list(answerer.stream(QUESTION))
    assert [e.data["text"] for e in events if e.type == "token"] == ["Partial [C1]"]
    trace = events[-1].data["trace"]
    assert trace.answer == "" and trace.citations == [] and trace.stop_reason == "refusal"
    assert events[-1].data["stop_reason"] == "refusal"


def test_a_cut_off_answer_is_kept_and_marked(toy):
    answerer, _ = toy(
        FakeGenerator(deltas=["It is checked in `validate_token` [C1:L8"], stop_reason="max_tokens")
    )
    trace = answerer.run(QUESTION)
    assert trace.stop_reason == "max_tokens" and trace.answer.endswith("[C1:L8")
    assert trace.uncited_sentences == [0] and trace.malformed == []


def test_malformed_citations_are_recorded(toy):
    answerer, _ = toy(FakeGenerator(deltas=["It is in `validate_token` [C1:95-100]."]))
    events = {e.type: e for e in answerer.stream(QUESTION)}
    assert events["citations"].data["malformed"] == ["[C1:95-100]"]
    assert events["done"].data["trace"].malformed == ["[C1:95-100]"]


def test_a_failing_model_is_an_error_event_and_run_raises(toy):
    generator = FakeGenerator()
    generator.fail = True
    answerer, _ = toy(generator)
    events = list(answerer.stream(QUESTION))
    assert [e.type for e in events][-1] == "error" and "done" not in [e.type for e in events]
    assert events[-1].data == {"stage": "generate", "message": "model unavailable"}
    with pytest.raises(CqaError, match="generate: model unavailable"):
        answerer.run(QUESTION)


def test_a_retrieval_failure_names_its_stage(toy, monkeypatch):
    answerer, _ = toy()
    monkeypatch.setattr(answerer.pipeline, "run", lambda q: (_ for _ in ()).throw(RuntimeError("store gone")))
    (event,) = list(answerer.stream(QUESTION))
    assert event.type == "error" and event.data == {"stage": "retrieve", "message": "store gone"}


def test_generate_answers_from_a_given_context_without_retrieval(toy):
    answerer, _ = toy()
    answerer.pipeline = None
    with pytest.raises(ValueError):
        list(answerer.stream(QUESTION))
    gold = chunk(1, "src/auth/session.py", 88, 121, kind="gold")
    ctx = Context(chunks=[gold], token_count=300, low_confidence=False)
    trace = Trace(question=QUESTION, index_id="x", config_hash="y", arm="oracle")
    events = list(answerer.generate(QUESTION, ctx, trace))
    assert events[0].data["sources"][0]["rerank_score"] is None
    assert events[-1].data["trace"].arm == "oracle"


def test_payloads_match_the_client_contract(toy):
    answerer, _ = toy()
    events = {e.type: e.data for e in answerer.stream(QUESTION)}
    assert set(events["sources"]) == {"sources", "low_confidence"}
    assert set(events["sources"]["sources"][0]) == {"label", "path", "start_line", "end_line", "rerank_score"}
    assert set(events["citations"]) == {"citations", "uncited_sentences", "malformed"}
    assert set(events["citations"]["citations"][0]) == {"label", "lines", "sentence", "status"}
    assert set(events["done"]) == {
        "trace",
        "timings_ms",
        "cost_usd",
        "config_hash",
        "cached",
        "model",
        "stop_reason",
    }
    json.dumps({k: v for k, v in events["done"].items() if k != "trace"})


def test_sources_payload():
    ctx = Context(
        chunks=[chunk(7, "a.py", 1, 40), chunk(9, "b.py", 5, 9)], token_count=200, low_confidence=True
    )
    payload = sources_payload(ctx, scored([7]))
    assert payload["low_confidence"] is True
    assert [(s["label"], s["path"], s["rerank_score"]) for s in payload["sources"]] == [
        ("C1", "a.py", 1.0),
        ("C2", "b.py", None),
    ]


def fake_index_embedder(monkeypatch):
    monkeypatch.setattr(
        "cqa.generate.answer.make_embedder",
        lambda cfg, caches: CachedEmbedder(FakeEmbedder(dims=cfg.dims, model_id=cfg.embedder), caches),
    )


def test_build_answerer_wires_everything_and_defaults_to_head(toyrepo, tmp_path, monkeypatch):
    repo, sha = toyrepo
    fake_index_embedder(monkeypatch)
    cfg = load_config(ROOT / "configs/base.yaml")
    answerer = build_answerer(cfg, str(repo), None, tmp_path)
    assert answerer.commit == sha and answerer.repo == repo.name
    assert answerer.config_hash == config_hash(cfg) and answerer.prices is None
    assert isinstance(answerer.generator, AnthropicGenerator)
    assert isinstance(answerer.generator.cache, LlmCache)
    assert answerer.pipeline.index.store is not None


def test_the_ask_command_prints_the_answer_and_its_citations(toyrepo, tmp_path, monkeypatch):
    repo, _ = toyrepo
    monkeypatch.setenv("CQA_DATA_DIR", str(tmp_path / "data"))
    fake_index_embedder(monkeypatch)
    real_build = build_answerer

    def build_with_fake_model(cfg, repo_arg, commit, data_dir):
        answerer = real_build(cfg, repo_arg, commit, data_dir)
        answerer.generator = FakeGenerator()
        return answerer

    monkeypatch.setattr("cqa.cli.build_answerer", build_with_fake_model)
    args = ["ask", QUESTION, "--repo", str(repo), "--config", str(ROOT / "configs/base.yaml"), "--trace"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Tokens are checked in `validate_token` [C1]." in result.stdout
    assert "[C1]  src/" in result.stdout and "valid" in result.stdout
    assert "[C9]  (no such excerpt)  fabricated" in result.stdout
    assert "timings (ms):" in result.stdout and "dense:" in result.stdout


def test_the_ask_command_reports_a_declined_answer(toyrepo, tmp_path, monkeypatch):
    repo, _ = toyrepo
    monkeypatch.setenv("CQA_DATA_DIR", str(tmp_path / "data"))
    fake_index_embedder(monkeypatch)
    real_build = build_answerer

    def build_with_refusal(cfg, repo_arg, commit, data_dir):
        answerer = real_build(cfg, repo_arg, commit, data_dir)
        answerer.generator = FakeGenerator(deltas=["I"], stop_reason="refusal")
        return answerer

    monkeypatch.setattr("cqa.cli.build_answerer", build_with_refusal)
    result = CliRunner().invoke(app, ["ask", QUESTION, "--repo", str(repo)])
    assert result.exit_code == 0 and "declined to answer" in result.stderr


def test_the_ask_command_reports_a_bad_repository(tmp_path, monkeypatch):
    monkeypatch.setenv("CQA_DATA_DIR", str(tmp_path / "data"))
    result = CliRunner().invoke(app, ["ask", QUESTION, "--repo", "https://example.com/r.git"])
    assert result.exit_code == 1 and "--commit is required" in result.stderr


def test_a_local_revision_resolves_to_its_full_sha(toyrepo):
    from cqa.index.build import resolve_repo

    repo, sha = toyrepo
    git("branch", "feature", cwd=repo)
    assert resolve_repo(str(repo), "feature") == (str(repo.resolve()), sha)
    assert resolve_repo(str(repo), sha[:8])[1] == sha
