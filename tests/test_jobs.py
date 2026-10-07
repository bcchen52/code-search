"""Index jobs: atomic claims, recovery from stale heartbeats, and claim-guarded updates."""

import threading
import time
from datetime import datetime

import pytest
import yaml

from cqa.chunking.fixed import FixedWindowChunker
from cqa.config import IndexConfig
from cqa.embed.cache import CachedEmbedder
from cqa.index.build import index_identity, stage_versions
from cqa.serve import worker
from cqa.serve.jobs import advance, claim, enqueue, fail, heartbeat, now_iso, requeue_stale, retry

from helpers import ROOT, FakeEmbedder

T0 = "2026-10-01T12:00:00.000000Z"
T30 = "2026-10-01T12:00:30.000000Z"
T61 = "2026-10-01T12:01:01.000000Z"
T90 = "2026-10-01T12:01:30.000000Z"

INDEX_CONFIG = IndexConfig.model_validate(yaml.safe_load((ROOT / "configs/base.yaml").read_text())["index"])
URL = "https://github.com/acme/toy"
SHA = "0123456789abcdef0123456789abcdef01234567"


def queue(conn, index_id, now=T0):
    return enqueue(conn, index_id, URL, SHA, INDEX_CONFIG, now)


def test_a_job_is_claimed_exactly_once(main_db):
    job = queue(main_db, "idx1")
    assert claim(main_db, "w1", T0) == job
    assert claim(main_db, "w2", T0) is None


def test_jobs_are_claimed_oldest_first(main_db):
    first, second = queue(main_db, "idx1"), queue(main_db, "idx2")
    assert [claim(main_db, "w1", T0), claim(main_db, "w1", T0)] == [first, second]


def test_one_active_job_per_index(main_db):
    job = queue(main_db, "idx1")
    assert queue(main_db, "idx1") == job
    claim(main_db, "w1", T0)
    assert queue(main_db, "idx1") == job


def test_a_job_keeps_what_the_worker_needs(main_db):
    job = queue(main_db, "idx1")
    row = main_db.execute(
        "SELECT repo_url, commit_sha, index_config FROM index_jobs WHERE id = ?", (job,)
    ).fetchone()
    assert (row[0], row[1]) == (URL, SHA)
    assert IndexConfig.model_validate_json(row[2]) == INDEX_CONFIG


def test_a_stale_job_returns_to_the_queue(main_db):
    job = queue(main_db, "idx1")
    claim(main_db, "w1", T0)
    assert requeue_stale(main_db, T30) == 0
    assert requeue_stale(main_db, T61) == 1
    assert claim(main_db, "w2", T61) == job


def test_a_heartbeat_keeps_a_job_alive(main_db):
    queue(main_db, "idx1")
    job = claim(main_db, "w1", T0)
    heartbeat(main_db, job, "w1", T30)
    assert requeue_stale(main_db, T61) == 0


def test_a_released_worker_cannot_touch_the_job(main_db):
    job = queue(main_db, "idx1")
    claim(main_db, "w1", T0)
    requeue_stale(main_db, T61)
    claim(main_db, "w2", T61)
    heartbeat(main_db, job, "w1", T90)
    advance(main_db, job, "w1", "embedding", 0.5)
    fail(main_db, job, "w1", "parsing: out of memory")
    row = main_db.execute(
        "SELECT state, heartbeat_at, claimed_by FROM index_jobs WHERE id = ?", (job,)
    ).fetchone()
    assert tuple(row) == ("cloning", T61, "w2")


def test_states_only_move_forward(main_db):
    job = queue(main_db, "idx1")
    claim(main_db, "w1", T0)
    advance(main_db, job, "w1", "embedding", 0.5)
    with pytest.raises(ValueError):
        advance(main_db, job, "w1", "parsing", 0.6)


def state_of(conn, job):
    return tuple(
        conn.execute("SELECT state, progress, error FROM index_jobs WHERE id = ?", (job,)).fetchone()
    )


def test_now_iso_is_fixed_width_utc():
    stamp = now_iso()
    assert len(stamp) == len(T0) and stamp.endswith("Z")
    assert datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%fZ")


def test_progress_within_a_state_and_on_to_ready(main_db):
    job = queue(main_db, "idx1")
    claim(main_db, "w1", T0)
    advance(main_db, job, "w1", "parsing", 0.1)
    advance(main_db, job, "w1", "parsing", 0.2)
    advance(main_db, job, "w1", "ready", 1.0)
    assert state_of(main_db, job)[:2] == ("ready", 1.0)
    with pytest.raises(ValueError):
        advance(main_db, job, "w1", "failed", 1.0)


def test_a_finished_job_is_neither_stale_nor_active(main_db):
    job = queue(main_db, "idx1")
    claim(main_db, "w1", T0)
    advance(main_db, job, "w1", "ready", 1.0)
    assert requeue_stale(main_db, T90) == 0
    assert queue(main_db, "idx1") != job


def test_fail_then_retry(main_db):
    job = queue(main_db, "idx1")
    claim(main_db, "w1", T0)
    fail(main_db, job, "w1", "embedding: model crashed")
    assert state_of(main_db, job) == ("failed", 0.0, "embedding: model crashed")
    retry(main_db, job)
    assert state_of(main_db, job) == ("queued", 0.0, None)
    assert claim(main_db, "w2", T30) == job


def test_retry_refuses_a_job_that_is_not_failed_or_has_a_replacement(main_db):
    job = queue(main_db, "idx1")
    with pytest.raises(ValueError, match="not failed"):
        retry(main_db, job)
    claim(main_db, "w1", T0)
    fail(main_db, job, "w1", "cloning: timeout")
    replacement = queue(main_db, "idx1")
    assert replacement != job
    with pytest.raises(ValueError, match="already active"):
        retry(main_db, job)


def test_two_workers_racing_never_share_a_job(tmp_path):
    from cqa.db import connect, init_schema

    path = tmp_path / "cqa.sqlite"
    setup = connect(path)
    init_schema(setup, "main")
    jobs = {queue(setup, f"idx{i}") for i in range(40)}
    claimed = {"a": [], "b": []}

    def work(name):
        conn = connect(path)
        while (job := claim(conn, name, T0)) is not None:
            claimed[name].append(job)

    threads = [threading.Thread(target=work, args=(n,)) for n in claimed]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(claimed["a"] + claimed["b"]) == sorted(jobs)


@pytest.fixture
def fake_embedder(monkeypatch):
    monkeypatch.setattr(
        "cqa.serve.worker.make_embedder",
        lambda cfg, caches: CachedEmbedder(FakeEmbedder(dims=cfg.dims, model_id=cfg.embedder), caches),
    )


def toy_job(conn, repo, sha, data_dir):
    url = str(repo.resolve())
    index_id = index_identity(
        url, sha, INDEX_CONFIG, stage_versions(FixedWindowChunker(40, 10), INDEX_CONFIG)
    )
    return enqueue(conn, index_id, url, sha, INDEX_CONFIG, now_iso()), index_id


def test_run_job_builds_the_index_and_reports_each_state(toyrepo, tmp_path, fake_embedder, monkeypatch):
    from cqa.db import connect, init_schema

    conn = connect(tmp_path / "cqa.sqlite")
    init_schema(conn, "main")
    repo, sha = toyrepo
    job, index_id = toy_job(conn, repo, sha, tmp_path)
    seen = []
    real_advance = worker.advance
    monkeypatch.setattr(
        worker, "advance", lambda c, j, w, s, p: seen.append(s) or real_advance(c, j, w, s, p)
    )
    claim(conn, "w1", now_iso())
    worker.run_job(conn, job, "w1", tmp_path)
    assert state_of(conn, job)[:2] == ("ready", 1.0)
    assert seen == ["parsing", "parsing", "embedding", "writing", "ready"]
    status = conn.execute("SELECT status FROM index_versions WHERE id = ?", (index_id,)).fetchone()[0]
    assert status == "ready"


def test_run_job_failure_names_the_stage(toyrepo, tmp_path, monkeypatch):
    from cqa.db import connect, init_schema

    class Broken(FakeEmbedder):
        def embed(self, texts, mode):
            raise RuntimeError("model crashed")

    monkeypatch.setattr("cqa.serve.worker.make_embedder", lambda cfg, caches: Broken(cfg.dims, cfg.embedder))
    conn = connect(tmp_path / "cqa.sqlite")
    init_schema(conn, "main")
    repo, sha = toyrepo
    job, _ = toy_job(conn, repo, sha, tmp_path)
    claim(conn, "w1", now_iso())
    worker.run_job(conn, job, "w1", tmp_path)
    state, _, error = state_of(conn, job)
    assert state == "failed" and error.startswith("embedding: ") and "model crashed" in error


def test_run_job_refuses_to_build_a_different_index(toyrepo, tmp_path, fake_embedder):
    from cqa.db import connect, init_schema

    conn = connect(tmp_path / "cqa.sqlite")
    init_schema(conn, "main")
    repo, sha = toyrepo
    job = enqueue(conn, "not-this-index", str(repo.resolve()), sha, INDEX_CONFIG, now_iso())
    claim(conn, "w1", now_iso())
    worker.run_job(conn, job, "w1", tmp_path)
    state, _, error = state_of(conn, job)
    assert state == "failed" and "the job names not-this-index" in error


def test_the_worker_loop_runs_a_job_with_heartbeats_then_stops(toyrepo, tmp_path, fake_embedder, monkeypatch):
    from cqa.db import connect, init_schema

    conn = connect(tmp_path / "cqa.sqlite")
    init_schema(conn, "main")
    repo, sha = toyrepo
    job, _ = toy_job(conn, repo, sha, tmp_path)
    real_run_job = worker.run_job

    def slow_run_job(c, j, w, d):
        time.sleep(0.3)
        real_run_job(c, j, w, d)

    monkeypatch.setattr(worker, "run_job", slow_run_job)
    worker.run_worker(
        tmp_path, "w1", poll_s=0.01, heartbeat_s=0.05, stop=lambda: state_of(conn, job)[0] == "ready"
    )
    row = conn.execute(
        "SELECT claimed_by, heartbeat_at, created_at FROM index_jobs WHERE id = ?", (job,)
    ).fetchone()
    assert row["claimed_by"] == "w1" and row["heartbeat_at"] > row["created_at"]
