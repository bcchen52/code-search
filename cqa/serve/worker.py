"""The index worker: claims jobs, builds indexes, and keeps each claimed job's heartbeat fresh."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

from cqa.chunking import make_chunker
from cqa.config import IndexConfig
from cqa.db import DB_FILES, connect, init_schema
from cqa.embed import make_embedder
from cqa.errors import IndexBuildError
from cqa.index.build import build_index, checkout_dir
from cqa.ingest.walker import clone_at
from cqa.serve.jobs import advance, claim, fail, heartbeat, now_iso, requeue_stale

STAGE_STATES = {"walk": "parsing", "chunk": "parsing", "stores": "embedding", "publish": "writing"}
"""The job state for each stage ``build_index`` reports."""


def run_worker(
    data_dir: Path,
    worker_id: str,
    poll_s: float = 1.0,
    heartbeat_s: float = 10.0,
    stop: Callable[[], bool] = lambda: False,
) -> None:
    """Process jobs until ``stop()`` returns True.

    Each cycle requeues stale jobs, claims one, and runs it while a
    background thread sends heartbeats every ``heartbeat_s`` seconds. When
    the queue is empty, the worker sleeps for ``poll_s`` seconds. Several
    workers may run at once; claiming is atomic.
    """
    conn = connect(data_dir / DB_FILES["main"])
    init_schema(conn, "main")
    while not stop():
        requeue_stale(conn, now_iso())
        job_id = claim(conn, worker_id, now_iso())
        if job_id is None:
            time.sleep(poll_s)
            continue
        finished = threading.Event()
        beats = threading.Thread(
            target=_beat, args=(data_dir, job_id, worker_id, heartbeat_s, finished), daemon=True
        )
        beats.start()
        try:
            run_job(conn, job_id, worker_id, data_dir)
        finally:
            finished.set()
            beats.join()


def _beat(data_dir: Path, job_id: int, worker_id: str, every_s: float, finished: threading.Event) -> None:
    """Refresh the job's heartbeat until it finishes, on a connection of its own."""
    conn = connect(data_dir / DB_FILES["main"])
    try:
        while not finished.wait(every_s):
            heartbeat(conn, job_id, worker_id, now_iso())
    finally:
        conn.close()


def run_job(conn: sqlite3.Connection, job_id: int, worker_id: str, data_dir: Path) -> None:
    """Build the index a claimed job describes, reporting each stage to the job.

    Fetches the job's repository at its commit and builds with the job's
    stored index configuration. Any exception marks the job failed with the
    stage where it occurred.

    The index built must be the one the job names. If this worker computes a
    different identity, for example because it runs different code than the
    server that queued the job, the job fails instead of producing an index
    nobody asked for.
    """
    row = conn.execute(
        "SELECT index_id, repo_url, commit_sha, index_config FROM index_jobs WHERE id = ?", (job_id,)
    ).fetchone()
    stage = "cloning"
    try:
        cfg = IndexConfig.model_validate_json(row["index_config"])
        url, commit = row["repo_url"], row["commit_sha"]
        checkout = clone_at(url, commit, checkout_dir(data_dir, url, commit))
        caches = connect(data_dir / DB_FILES["caches"])
        init_schema(caches, "caches")

        def on_stage(name: str, fraction: float) -> None:
            nonlocal stage
            if name in STAGE_STATES:
                stage = STAGE_STATES[name]
                advance(conn, job_id, worker_id, stage, fraction)

        built = build_index(
            url,
            commit,
            checkout,
            cfg,
            conn,
            data_dir,
            make_chunker(cfg),
            make_embedder(cfg, caches),
            on_stage,
        )
        if built != row["index_id"]:
            raise IndexBuildError(f"built index {built}, but the job names {row['index_id']}")
        advance(conn, job_id, worker_id, "ready", 1.0)
    except Exception as e:
        fail(conn, job_id, worker_id, f"{stage}: {e}")
