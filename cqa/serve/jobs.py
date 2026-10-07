"""Index jobs, queued as rows in SQLite.

::

    queued -> cloning -> parsing -> embedding -> writing -> ready
    any running state -> failed
    failed -> queued, on retry

A worker claims a job with one atomic ``UPDATE``. It refreshes the job's
heartbeat while working; a running job whose heartbeat goes stale returns to
the queue, which recovers from a crashed worker. Every update a worker makes
requires that it still holds the claim, so a worker presumed dead cannot
touch a job another worker has taken over. At most one job per index is
queued or running at a time. Timestamps are UTC ISO-8601 in one fixed-width
format, so string comparison in SQL is time comparison.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from cqa.config import IndexConfig

STATES = ("queued", "cloning", "parsing", "embedding", "writing", "ready", "failed")
RUNNING = ("cloning", "parsing", "embedding", "writing")
ACTIVE = ("queued", *RUNNING)

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
_RUNNING_SQL = ", ".join(f"'{s}'" for s in RUNNING)
_ACTIVE_SQL = ", ".join(f"'{s}'" for s in ACTIVE)

CLAIM_SQL = """
UPDATE index_jobs
SET state = 'cloning', claimed_by = :worker, heartbeat_at = :now
WHERE id = (SELECT id FROM index_jobs WHERE state = 'queued' ORDER BY id LIMIT 1)
RETURNING id;
"""
"""Claims the oldest queued job in a single statement. ``RETURNING`` requires SQLite 3.35 or later."""


def now_iso() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SS.ffffffZ``."""
    return datetime.now(UTC).strftime(TIME_FORMAT)


def enqueue(
    conn: sqlite3.Connection, index_id: str, repo_url: str, commit: str, index_config: IndexConfig, now: str
) -> int:
    """Queue a job to build an index, and return its id.

    The job stores the repository, commit, and index configuration, so the
    worker can rebuild exactly the index that ``index_id`` names. If a job for
    the same index is already queued or running, its id is returned instead.
    The check and the insert share one write transaction, so two requests for
    the same index cannot both queue a job.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            f"SELECT id FROM index_jobs WHERE index_id = ? AND state IN ({_ACTIVE_SQL}) ORDER BY id LIMIT 1",
            (index_id,),
        ).fetchone()
        if row is not None:
            conn.commit()
            return int(row["id"])
        cursor = conn.execute(
            "INSERT INTO index_jobs (index_id, repo_url, commit_sha, index_config, state, created_at) "
            "VALUES (?, ?, ?, ?, 'queued', ?)",
            (index_id, repo_url, commit, index_config.model_dump_json(), now),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    if cursor.lastrowid is None:  # pragma: no cover - INSERT always sets it
        raise RuntimeError("job insert returned no id")
    return cursor.lastrowid


def claim(conn: sqlite3.Connection, worker_id: str, now: str) -> int | None:
    """Claim the oldest queued job with ``CLAIM_SQL``; return its id, or None when the queue is empty.

    Two workers never claim the same job.
    """
    with conn:
        row = conn.execute(CLAIM_SQL, {"worker": worker_id, "now": now}).fetchone()
    return None if row is None else int(row["id"])


def heartbeat(conn: sqlite3.Connection, job_id: int, worker_id: str, now: str) -> None:
    """Refresh a job's heartbeat, only while ``worker_id`` still holds the job."""
    with conn:
        conn.execute(
            f"UPDATE index_jobs SET heartbeat_at = ? "
            f"WHERE id = ? AND claimed_by = ? AND state IN ({_RUNNING_SQL})",
            (now, job_id, worker_id),
        )


def advance(conn: sqlite3.Connection, job_id: int, worker_id: str, state: str, progress: float) -> None:
    """Move a running job to a later state and record its progress, only while ``worker_id`` holds it.

    The same state with new progress is accepted, since one state can span
    several stages. A worker that no longer holds the job changes nothing.

    Raises:
        ValueError: If ``state`` comes before the job's current state, or is
            ``queued`` or ``failed`` (use ``retry`` and ``fail``).
    """
    if state not in STATES or state in ("queued", "failed"):
        raise ValueError(f"cannot advance a job to {state!r}")
    with conn:
        row = conn.execute(
            f"SELECT state FROM index_jobs WHERE id = ? AND claimed_by = ? AND state IN ({_RUNNING_SQL})",
            (job_id, worker_id),
        ).fetchone()
        if row is None:
            return
        if STATES.index(state) < STATES.index(row["state"]):
            raise ValueError(f"job {job_id} is {row['state']!r}; it cannot move back to {state!r}")
        conn.execute(
            "UPDATE index_jobs SET state = ?, progress = ? WHERE id = ? AND claimed_by = ? AND state = ?",
            (state, progress, job_id, worker_id, row["state"]),
        )


def fail(conn: sqlite3.Connection, job_id: int, worker_id: str, error: str) -> None:
    """Mark a job failed with an error message naming the stage, only while ``worker_id`` holds it."""
    with conn:
        conn.execute(
            f"UPDATE index_jobs SET state = 'failed', error = ? "
            f"WHERE id = ? AND claimed_by = ? AND state IN ({_RUNNING_SQL})",
            (error, job_id, worker_id),
        )


def requeue_stale(conn: sqlite3.Connection, now: str, stale_after_s: int = 60) -> int:
    """Return running jobs whose heartbeat is older than ``stale_after_s`` to the queue; return how many."""
    cutoff = (datetime.strptime(now, TIME_FORMAT) - timedelta(seconds=stale_after_s)).strftime(TIME_FORMAT)
    with conn:
        cursor = conn.execute(
            f"UPDATE index_jobs SET state = 'queued', claimed_by = NULL, progress = 0 "
            f"WHERE state IN ({_RUNNING_SQL}) AND heartbeat_at < ?",
            (cutoff,),
        )
    return cursor.rowcount


def retry(conn: sqlite3.Connection, job_id: int) -> None:
    """Queue a failed job again. Cached embeddings mean only uncached work is repeated.

    Raises:
        ValueError: If the job is not failed, or another job for the same
            index is already queued or running.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT index_id, state FROM index_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None or row["state"] != "failed":
            raise ValueError(f"job {job_id} is not failed")
        active = conn.execute(
            f"SELECT id FROM index_jobs WHERE index_id = ? AND state IN ({_ACTIVE_SQL})", (row["index_id"],)
        ).fetchone()
        if active is not None:
            raise ValueError(f"job {active['id']} is already active for index {row['index_id']}")
        conn.execute(
            "UPDATE index_jobs SET state = 'queued', error = NULL, claimed_by = NULL, progress = 0 "
            "WHERE id = ?",
            (job_id,),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
