"""The HTTP API: registration, guarded streaming queries, stored traces, files, and metrics."""

import json
import shutil
from dataclasses import replace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cqa.chunking.fixed import FixedWindowChunker  # noqa: E402
from cqa.config import load_config  # noqa: E402
from cqa.db import connect, init_schema  # noqa: E402
from cqa.embed.cache import CachedEmbedder  # noqa: E402
from cqa.generate.llm import load_prices  # noqa: E402
from cqa.index.build import build_index, checkout_dir  # noqa: E402
from cqa.serve import api  # noqa: E402
from cqa.serve.guardrails import RateLimiter, SpendCap  # noqa: E402
from cqa.serve.settings import ServerSettings  # noqa: E402

from helpers import ROOT, FakeEmbedder, FakeGenerator  # noqa: E402

URL = "https://example.com/acme/toy"
OTHER = "https://example.com/acme/other"
QUESTION = "Where is the session token validated?"


def frames(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@pytest.fixture
def server(toyrepo, tmp_path, monkeypatch):
    """A server over the toy repository, indexed as an https URL, with a fake model."""
    repo, sha = toyrepo
    data = tmp_path / "data"
    data.mkdir()
    base = load_config(ROOT / "configs/base.yaml")
    cfg = base.model_copy(update={"index": base.index.model_copy(update={"embedder": "fake", "dims": 16})})
    conn, caches = connect(data / "cqa.sqlite"), connect(data / "caches.sqlite")
    init_schema(conn, "main")
    init_schema(caches, "caches")
    embedder = CachedEmbedder(FakeEmbedder(), caches)
    shutil.copytree(repo, checkout_dir(data, URL, sha))
    index_id = build_index(URL, sha, repo, cfg.index, conn, data, FixedWindowChunker(40, 10), embedder)
    settings = ServerSettings(
        data_dir=data,
        config_path=ROOT / "configs/base.yaml",
        rate_limit_per_min=100,
        daily_spend_cap_usd=1.0,
        repo_allowlist=frozenset({URL, OTHER}),
        expose_traces=True,
    )
    state = api.ServerState(
        settings=settings,
        conn=conn,
        rate_limiter=RateLimiter(100),
        spend_cap=SpendCap(1.0),
        cfg=cfg,
        caches=caches,
        prices=load_prices(ROOT / "configs/prices.yaml"),
    )
    state._embedders[("fake", 16)] = embedder
    generator = FakeGenerator()
    monkeypatch.setattr(api, "AnthropicGenerator", lambda *args, **kwargs: generator)
    repo_id = conn.execute("SELECT id FROM repos WHERE url = ?", (URL,)).fetchone()["id"]
    client = TestClient(api.create_app(state))
    return client, state, repo_id, sha, index_id, generator


def ask(client, repo_id, question=QUESTION):
    return client.post(f"/repos/{repo_id}/query", json={"question": question})


def test_a_query_streams_events_and_stores_its_trace(server):
    client, _, repo_id, _, index_id, _ = server
    response = ask(client, repo_id)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache" and response.headers["x-accel-buffering"] == "no"
    events = frames(response.text)
    assert [e for e, _ in events] == ["sources", "token", "token", "citations", "done"]
    done = events[-1][1]
    stored = client.get(f"/queries/{done['query_id']}").json()
    assert (
        stored["trace"]["answer"].startswith("Tokens are checked") and stored["trace"]["index_id"] == index_id
    )
    assert stored["trace"]["lists"]["dense"] and stored["model"] == "claude-sonnet-5-5"
    assert stored["stop_reason"] == "end_turn" and stored["cost_usd"] == pytest.approx(0.0022)


def test_a_refusal_is_stored_with_an_empty_answer(server):
    client, _, repo_id, _, _, generator = server
    generator.deltas, generator.usage = (
        ["Partial"],
        generator.usage.__class__(
            tokens_in=900, tokens_out=1, model="claude-sonnet-5-5", stop_reason="refusal"
        ),
    )
    done = frames(ask(client, repo_id).text)[-1][1]
    stored = client.get(f"/queries/{done['query_id']}").json()
    assert stored["trace"]["answer"] == "" and stored["stop_reason"] == "refusal"


def test_a_model_failure_ends_the_stream_with_an_error_frame(server):
    client, _, repo_id, _, _, generator = server
    generator.fail = True
    events = frames(ask(client, repo_id).text)
    assert events[-1] == ("error", {"stage": "generate", "message": "model unavailable"})


def test_an_exception_that_escapes_the_answerer_still_ends_cleanly(server, monkeypatch):
    client, state, repo_id, _, _, _ = server
    ask(client, repo_id)
    (answerer,) = state._answerers.values()

    def broken(question):
        yield api.Event("sources", {"sources": [], "low_confidence": False})
        raise RuntimeError("disk vanished")

    monkeypatch.setattr(answerer, "stream", broken)
    events = frames(ask(client, repo_id).text)
    assert events == [
        ("sources", {"sources": [], "low_confidence": False}),
        ("error", {"stage": "stream", "message": "disk vanished"}),
    ]


def test_the_rate_limit_refuses_before_the_stream_starts(server):
    client, state, repo_id, _, _, _ = server
    state.rate_limiter = RateLimiter(per_minute=1)
    assert ask(client, repo_id).status_code == 200
    second = ask(client, repo_id)
    assert second.status_code == 429 and second.json()["error"] == "RateLimitedError"


def test_the_spend_cap_refuses_before_the_stream_starts(server):
    client, state, repo_id, _, _, _ = server
    state.spend_cap = SpendCap(daily_usd=0.0)
    response = ask(client, repo_id)
    assert response.status_code == 503 and "budget" in response.json()["message"]


def test_bad_questions_and_unknown_repositories(server):
    client, _, repo_id, _, _, _ = server
    assert ask(client, repo_id, question="   ").status_code == 400
    assert ask(client, repo_id, question="x" * 2001).status_code == 400
    assert ask(client, 999).status_code == 404
    assert client.post(f"/repos/{repo_id}/query", json={"question": "q", "extra": 1}).status_code == 422


def test_only_an_index_built_with_the_current_configuration_is_served(server):
    client, state, repo_id, _, _, _ = server
    state.cfg = state.cfg.model_copy(
        update={"index": state.cfg.index.model_copy(update={"fallback_window_lines": 30})}
    )
    response = ask(client, repo_id)
    assert response.status_code == 409 and "current configuration" in response.json()["message"]
    (listed,) = [r for r in client.get("/repos").json() if r["repo_id"] == repo_id]
    assert listed["servable"] is False


def test_registering_queues_a_job_and_lists_the_repository(server):
    client, _, _, sha, _, _ = server
    first = client.post("/repos", json={"url": "https://Example.com/acme/other.git", "commit": sha})
    assert first.status_code == 202
    again = client.post("/repos", json={"url": OTHER, "commit": sha}).json()
    assert again["job_id"] == first.json()["job_id"]
    job = client.get(f"/jobs/{again['job_id']}").json()
    assert (job["state"], job["repo_url"], job["commit"]) == ("queued", OTHER, sha)
    (listed,) = [r for r in client.get("/repos").json() if r["url"] == OTHER]
    assert listed["servable"] is False and listed["job"]["state"] == "queued"


def test_registering_without_a_commit_pins_the_default_branch(server, monkeypatch):
    client, _, _, sha, _, _ = server
    monkeypatch.setattr(api, "remote_head", lambda url: sha)
    assert client.post("/repos", json={"url": OTHER}).json()["commit"] == sha

    def unreachable(url):
        raise api.CqaError("resolve: could not read from remote")

    monkeypatch.setattr(api, "remote_head", unreachable)
    response = client.post("/repos", json={"url": OTHER})
    assert response.status_code == 400 and "default branch" in response.json()["message"]


@pytest.mark.parametrize(
    "body, status",
    [
        ({"url": "/srv/repos/toy"}, 403),
        ({"url": "https://github.com/acme/not-listed"}, 403),
        ({"url": "ext::sh -c id"}, 403),
        ({"url": OTHER, "commit": "abc123"}, 400),
    ],
)
def test_registering_refuses_bad_requests(server, body, status):
    client, *_ = server
    assert client.post("/repos", json=body).status_code == status


def test_file_lines_are_numbered_and_only_from_indexed_files(server):
    client, _, repo_id, _, _, _ = server
    response = client.get(
        f"/repos/{repo_id}/file", params={"path": "src/auth/session.py", "start": 88, "end": 89}
    )
    assert response.status_code == 200
    assert response.text.splitlines()[0].startswith("88 | ") and "validate_token" in response.text
    for path in ("vendor/leftpad.py", "../../etc/passwd"):
        assert client.get(f"/repos/{repo_id}/file", params={"path": path}).status_code == 404
    bad = client.get(f"/repos/{repo_id}/file", params={"path": "src/auth/session.py", "start": 0, "end": 5})
    assert bad.status_code == 400


def test_metrics_summary(server):
    client, _, repo_id, _, _, _ = server
    for _ in range(3):
        ask(client, repo_id)
    summary = client.get("/metrics/summary").json()
    assert summary["queries"] == 3 and summary["spend_cap_usd"] == 1.0
    assert {"ttft", "retrieve", "total"} <= set(summary["stages"])
    assert summary["stages"]["total"]["p50"] <= summary["stages"]["total"]["p95"]
    assert summary["cost_today_usd"] == pytest.approx(3 * 0.0022)


def test_unknown_jobs_queries_and_eval_runs_are_404_and_runs_start_empty(server):
    client, *_ = server
    assert client.get("/jobs/999").status_code == 404
    assert client.get("/queries/999").status_code == 404
    assert client.get("/eval/runs").json() == []
    assert client.get("/eval/runs/nope").status_code == 404


def test_a_stored_query_names_every_chunk_its_trace_mentions(server):
    client, _, repo_id, _, _, _ = server
    done = frames(ask(client, repo_id).text)[-1][1]
    stored = client.get(f"/queries/{done['query_id']}").json()
    trace = stored["trace"]
    mentioned = {e["chunk_id"] for e in trace["lists"]["dense"] + trace["reranked"] + trace["context"]}
    assert len(trace["lists"]["dense"]) > len(trace["context"])  # some candidates were cut
    assert {int(k) for k in stored["chunks"]} == mentioned and stored["repo_id"] == repo_id
    first = trace["context"][0]
    assert stored["chunks"][str(first["chunk_id"])] == {
        "path": first["path"],
        "start_line": first["start_line"],
        "end_line": first["end_line"],
        "kind": "window",
        "symbol": None,
    }


def test_recent_queries_are_newest_first_and_per_repository(server):
    client, state, repo_id, _, _, _ = server
    first = frames(ask(client, repo_id, "What does the toy repository do?").text)[-1][1]["query_id"]
    second = frames(ask(client, repo_id).text)[-1][1]["query_id"]
    recent = client.get(f"/repos/{repo_id}/queries").json()
    assert [q["query_id"] for q in recent] == [second, first]
    assert recent[0]["question"] == QUESTION and recent[0]["low_confidence"] is False
    assert [q["query_id"] for q in client.get(f"/repos/{repo_id}/queries?limit=1").json()] == [second]
    client.post("/repos", json={"url": OTHER, "commit": "a" * 40})
    other = state.conn.execute("SELECT id FROM repos WHERE url = ?", (OTHER,)).fetchone()["id"]
    assert client.get(f"/repos/{other}/queries").json() == []
    assert client.get("/repos/999/queries").status_code == 404
    assert client.get(f"/repos/{repo_id}/queries?limit=0").status_code == 400


def test_stored_queries_are_not_served_unless_exposed(server):
    client, state, repo_id, _, _, _ = server
    query_id = frames(ask(client, repo_id).text)[-1][1]["query_id"]
    hidden = TestClient(api.create_app(replace(state, settings=replace(state.settings, expose_traces=False))))
    for path in (f"/queries/{query_id}", f"/repos/{repo_id}/queries"):
        response = hidden.get(path)
        assert response.status_code == 404 and "CQA_EXPOSE_TRACES" in response.json()["message"]


def test_answerers_and_embedders_are_built_once(server):
    client, state, repo_id, _, _, _ = server
    ask(client, repo_id)
    ask(client, repo_id)
    assert len(state._answerers) == 1 and len(state._embedders) == 1


def test_create_app_reads_the_environment_and_fails_closed(tmp_path, monkeypatch):
    for name in (
        "CQA_REPO_ALLOWLIST",
        "CQA_ALLOW_ANY_REPO",
        "CQA_RATE_LIMIT_PER_MIN",
        "CQA_DAILY_SPEND_CAP_USD",
        "CQA_EXPOSE_TRACES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CQA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CQA_CONFIG", str(ROOT / "configs/base.yaml"))
    with pytest.raises(api.CqaError, match="CQA_REPO_ALLOWLIST"):
        api.create_app()
    monkeypatch.setenv("CQA_REPO_ALLOWLIST", URL)
    client = TestClient(api.create_app())
    assert client.get("/repos").json() == []
