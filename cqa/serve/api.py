"""The HTTP API.

Handlers are plain functions over a ``ServerState``, testable without HTTP;
``create_app`` binds them to FastAPI routes. FastAPI is imported inside
``create_app``, so importing this module needs no optional dependencies.

Routes:

- ``POST /repos``: register a repository and queue its index job.
- ``GET /repos``: repositories with their index status.
- ``GET /jobs/{id}``: an index job's state and progress.
- ``POST /repos/{id}/query``: answer a question as a server-sent event stream.
- ``GET /repos/{id}/file``: numbered source lines for the citation viewer.
- ``GET /repos/{id}/queries``: the repository's recent stored queries.
- ``GET /queries/{id}``: a stored query trace, with the chunks it mentions.

The two query routes are served only when ``CQA_EXPOSE_TRACES`` is set, since
they show every visitor's questions.
- ``GET /metrics/summary``: latency percentiles and today's cost.
- ``GET /eval/runs`` and ``GET /eval/runs/{id}``: evaluation results.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from cqa.chunking import make_chunker
from cqa.config import Config, config_hash, load_config
from cqa.db import DB_FILES, connect, init_schema
from cqa.embed import make_embedder
from cqa.errors import (
    CqaError,
    IndexNotReadyError,
    InvalidRequestError,
    NotFoundError,
    RateLimitedError,
    RepoNotAllowedError,
    SpendCapReachedError,
)
from cqa.generate.answer import DEFAULT_PRICES, Answerer
from cqa.generate.llm import AnthropicGenerator, LlmCache, load_prices
from cqa.index.build import IndexHandle, index_identity, open_index, stage_versions
from cqa.ingest.walker import remote_head
from cqa.retrieve.pipeline import RetrievalPipeline
from cqa.serve.guardrails import RateLimiter, SpendCap, normalize_repo_url, repo_allowed
from cqa.serve.jobs import enqueue, now_iso
from cqa.serve.settings import ServerSettings
from cqa.trace import Event, Trace
from cqa.types import Embedder

MAX_QUESTION_CHARS = 2000
MAX_FILE_LINES = 2000
METRIC_WINDOW = 1000
"""Recent queries the latency percentiles are computed over."""
MAX_RECENT_QUERIES = 100

_FULL_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass
class ServerState:
    """Shared state for request handlers.

    Besides the guardrails, it holds the pipeline configuration and prices the
    server answers with, and caches one ``Answerer`` per index and one
    embedder per model, so each model is loaded once per process. Handlers run
    on concurrent threads: the caches are guarded by a lock, and writes to the
    main database by ``write_lock``.
    """

    settings: ServerSettings
    conn: sqlite3.Connection
    rate_limiter: RateLimiter
    spend_cap: SpendCap
    cfg: Config | None = None
    caches: sqlite3.Connection | None = None
    prices: dict[str, Any] | None = None
    write_lock: threading.Lock = field(default_factory=threading.Lock)
    _answerers: dict[str, Answerer] = field(default_factory=dict)
    _embedders: dict[tuple[str, int], Embedder] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_settings(cls, settings: ServerSettings) -> ServerState:
        """Open the database and build the guardrails from settings.

        Also loads the pipeline configuration and the price table, so a
        broken configuration stops the server at startup.

        Raises:
            ConfigError: If the configuration or the price table is invalid.
        """
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        conn = connect(settings.data_dir / DB_FILES["main"])
        init_schema(conn, "main")
        caches = connect(settings.data_dir / DB_FILES["caches"])
        init_schema(caches, "caches")
        return cls(
            settings=settings,
            conn=conn,
            rate_limiter=RateLimiter(settings.rate_limit_per_min),
            spend_cap=SpendCap(settings.daily_spend_cap_usd),
            cfg=load_config(settings.config_path),
            caches=caches,
            prices=load_prices(DEFAULT_PRICES),
        )

    def config(self) -> Config:
        if self.cfg is None:
            raise CqaError("the server has no pipeline configuration")
        return self.cfg

    def stage_versions(self) -> str:
        cfg = self.config()
        return stage_versions(make_chunker(cfg.index), cfg.index)


def register_repo(state: ServerState, url: str, commit: str | None) -> dict[str, Any]:
    """Register a repository and queue its index job.

    Only https URLs are accepted: local paths and other transports are
    refused, so the API can never index files from the server's own disk.
    Without a commit, the head of the remote's default branch is resolved and
    pinned. A job already queued or running for the same index is reused.
    The repository is listed from this moment, before its index exists.

    Returns:
        ``{"job_id", "repo_id", "index_id", "commit"}``, sent with status 202.

    Raises:
        RepoNotAllowedError: If the URL is not https or not on the allowlist.
        InvalidRequestError: If the commit is not a full 40-character SHA, or
            the remote's default branch cannot be read.
    """
    normalized = normalize_repo_url(url)
    if not repo_allowed(normalized, state.settings.repo_allowlist):
        raise RepoNotAllowedError(f"{normalized} is not on this server's allowlist")
    if commit is None:
        try:
            commit = remote_head(normalized)
        except (CqaError, ValueError) as e:
            raise InvalidRequestError(f"cannot read the default branch of {normalized}: {e}") from e
    elif not _FULL_SHA.fullmatch(commit):
        raise InvalidRequestError(f"commit must be a full 40-character lowercase SHA, got {commit!r}")
    cfg = state.config()
    index_id = index_identity(normalized, commit, cfg.index, state.stage_versions())
    name = normalized.rsplit("/", 1)[-1]
    with state.write_lock, state.conn:
        state.conn.execute(
            "INSERT INTO repos (url, name) VALUES (?, ?) ON CONFLICT (url) DO NOTHING", (normalized, name)
        )
    repo_id = state.conn.execute("SELECT id FROM repos WHERE url = ?", (normalized,)).fetchone()["id"]
    with state.write_lock:
        job_id = enqueue(state.conn, index_id, normalized, commit, cfg.index, now_iso())
    return {"job_id": job_id, "repo_id": repo_id, "index_id": index_id, "commit": commit}


def get_job(state: ServerState, job_id: int) -> dict[str, Any]:
    """Return a job's state, progress, and error.

    Raises:
        NotFoundError: If the job does not exist.
    """
    row = state.conn.execute(
        "SELECT id, index_id, repo_url, commit_sha, state, progress, error, created_at "
        "FROM index_jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        raise NotFoundError(f"no job {job_id}")
    return {
        "job_id": row["id"],
        "index_id": row["index_id"],
        "repo_url": row["repo_url"],
        "commit": row["commit_sha"],
        "state": row["state"],
        "progress": row["progress"],
        "error": row["error"],
        "created_at": row["created_at"],
    }


def list_repos(state: ServerState) -> list[dict[str, Any]]:
    """Return every repository with the status of its latest index.

    ``servable`` is True when the repository has a ready index built with
    the server's current configuration; ``job`` is its latest index job.
    """
    repos = []
    for r in state.conn.execute("SELECT id, url, name FROM repos ORDER BY name, id"):
        serving = _serving_index(state, r["id"], r["url"])
        job = state.conn.execute(
            "SELECT id, state, progress, error FROM index_jobs WHERE repo_url = ? ORDER BY id DESC LIMIT 1",
            (r["url"],),
        ).fetchone()
        repos.append(
            {
                "repo_id": r["id"],
                "url": r["url"],
                "name": r["name"],
                "servable": serving is not None,
                "index_id": serving[0] if serving else None,
                "commit": serving[1] if serving else None,
                "job": dict(job) if job else None,
            }
        )
    return repos


def query(state: ServerState, repo_id: int, question: str, client_ip: str) -> Iterator[Event]:
    """Answer a question about a repository as an event stream, and store the trace when it completes.

    Guardrails and the index are checked before the stream is returned, so a
    refusal is an HTTP error status rather than an event inside a started stream.
    The ``done`` event gains the stored query's id, ``query_id``.

    Raises:
        InvalidRequestError: If the question is empty or too long.
        RateLimitedError: If the client is over its rate limit.
        SpendCapReachedError: If today's spend has reached the cap.
        NotFoundError: If the repository does not exist.
        IndexNotReadyError: If the repository has no ready index.
    """
    if not question.strip() or len(question) > MAX_QUESTION_CHARS:
        raise InvalidRequestError(f"a question must have 1 to {MAX_QUESTION_CHARS} characters")
    if not state.rate_limiter.allow(client_ip, time.monotonic()):
        raise RateLimitedError(f"at most {state.rate_limiter.per_minute} questions a minute")
    if state.spend_cap.exceeded(state.conn, datetime.now(UTC).date().isoformat()):
        raise SpendCapReachedError("today's answer budget is spent; try again tomorrow (UTC)")
    answerer = _answerer_for(state, repo_id)
    return _stream_and_store(state, answerer, question)


def _stream_and_store(state: ServerState, answerer: Answerer, question: str) -> Iterator[Event]:
    for event in answerer.stream(question):
        if event.type == "done":
            query_id = _store_query(state, event.data["trace"])
            event = Event("done", {**event.data, "query_id": query_id})
        yield event


def _store_query(state: ServerState, trace: Trace) -> int:
    with state.write_lock, state.conn:
        cursor = state.conn.execute(
            "INSERT INTO queries (index_id, question, config_hash, answer, citations, timings_ms, tokens_in, "
            "tokens_out, cost_usd, created_at, trace, model, stop_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trace.index_id,
                trace.question,
                trace.config_hash,
                trace.answer,
                json.dumps([asdict(c) for c in trace.citations]),
                json.dumps(trace.timings_ms),
                trace.tokens_in,
                trace.tokens_out,
                trace.cost_usd,
                now_iso(),
                trace.to_json(),
                trace.model,
                trace.stop_reason,
            ),
        )
    if cursor.lastrowid is None:  # pragma: no cover - INSERT always sets it
        raise CqaError("query insert returned no id")
    return cursor.lastrowid


def _serving_index(state: ServerState, repo_id: int, url: str) -> tuple[str, str] | None:
    """The newest ready index of a repository built with the server's current configuration.

    An index built with an earlier configuration is never served, since its
    vectors or chunks may not match what the pipeline expects; registering
    the repository again builds the current one.
    """
    cfg, versions = state.config(), state.stage_versions()
    rows = state.conn.execute(
        "SELECT id, commit_sha FROM index_versions WHERE repo_id = ? AND status = 'ready' "
        "ORDER BY created_at DESC",
        (repo_id,),
    )
    for r in rows:
        if index_identity(url, r["commit_sha"], cfg.index, versions) == r["id"]:
            return r["id"], r["commit_sha"]
    return None


def _index_for(state: ServerState, repo_id: int) -> IndexHandle:
    pipeline = _answerer_for(state, repo_id).pipeline
    if pipeline is None:  # pragma: no cover - every cached answerer has a pipeline
        raise CqaError(f"repository {repo_id} has no retrieval pipeline")
    return pipeline.index


def _answerer_for(state: ServerState, repo_id: int) -> Answerer:
    repo = state.conn.execute("SELECT url FROM repos WHERE id = ?", (repo_id,)).fetchone()
    if repo is None:
        raise NotFoundError(f"no repository {repo_id}")
    serving = _serving_index(state, repo_id, repo["url"])
    if serving is None:
        raise IndexNotReadyError(
            f"repository {repo_id} has no index built with the current configuration yet"
        )
    index_id, commit = serving
    with state._lock:
        answerer = state._answerers.get(index_id)
        if answerer is None:
            cfg = state.config()
            handle = open_index(state.conn, index_id, state.settings.data_dir, cfg.index.store)
            key = (cfg.index.embedder, cfg.index.dims)
            if key not in state._embedders:
                state._embedders[key] = make_embedder(cfg.index, state.caches)
            embedder = state._embedders[key]
            pipeline = RetrievalPipeline.from_config(cfg, handle, embedder)
            cache = LlmCache(state.caches) if state.caches is not None else None
            generator = AnthropicGenerator(cfg.generate, handle.repo_name, commit, cache=cache)
            answerer = Answerer(
                cfg, pipeline, generator, handle.repo_name, commit, config_hash(cfg), state.prices
            )
            state._answerers[index_id] = answerer
    return answerer


def file_lines(state: ServerState, repo_id: int, path: str, start: int, end: int) -> str:
    """Return numbered lines of an indexed file.

    Each line is ``"<number> | <text>"``; a range past the end of the file is
    cut at the last line.

    Raises:
        NotFoundError: If the path is not a file of the repository's index.
        InvalidRequestError: If the range is empty, starts below 1, or spans
            more than ``MAX_FILE_LINES`` lines.
    """
    if start < 1 or end < start or end - start + 1 > MAX_FILE_LINES:
        raise InvalidRequestError(
            f"a line range must start at 1 or later and span at most {MAX_FILE_LINES} lines"
        )
    lines = _index_for(state, repo_id).file_lines(path, start, end)
    width = len(str(start + len(lines) - 1)) if lines else 1
    return "\n".join(
        f"{n:>{width}} | {text}" if text else f"{n:>{width}} |" for n, text in enumerate(lines, start=start)
    )


def list_queries(state: ServerState, repo_id: int, limit: int = 20) -> list[dict[str, Any]]:
    """Return a repository's most recent stored queries, newest first.

    Queries against every index of the repository are listed, not only the
    one served now. ``low_confidence`` is the gate's verdict from the trace.

    Raises:
        NotFoundError: If traces are not exposed, or the repository does not exist.
        InvalidRequestError: If ``limit`` is not between 1 and ``MAX_RECENT_QUERIES``.
    """
    _require_traces(state)
    if not 1 <= limit <= MAX_RECENT_QUERIES:
        raise InvalidRequestError(f"limit must be between 1 and {MAX_RECENT_QUERIES}")
    if state.conn.execute("SELECT 1 FROM repos WHERE id = ?", (repo_id,)).fetchone() is None:
        raise NotFoundError(f"no repository {repo_id}")
    rows = state.conn.execute(
        "SELECT q.id, q.question, q.created_at, q.cost_usd, q.model, q.stop_reason, "
        "json_extract(q.trace, '$.low_confidence') AS low_confidence "
        "FROM queries q JOIN index_versions v ON v.id = q.index_id "
        "WHERE v.repo_id = ? ORDER BY q.id DESC LIMIT ?",
        (repo_id, limit),
    )
    return [
        {
            "query_id": r["id"],
            "question": r["question"],
            "created_at": r["created_at"],
            "cost_usd": r["cost_usd"],
            "model": r["model"],
            "stop_reason": r["stop_reason"],
            "low_confidence": bool(r["low_confidence"]),
        }
        for r in rows
    ]


def get_query(state: ServerState, query_id: int) -> dict[str, Any]:
    """Return a stored query trace, with the chunks it mentions.

    ``chunks`` maps every chunk id in the trace's ranked lists and context
    (as a string, since JSON keys are strings) to its ``path``, line range,
    ``kind``, and ``symbol``, looked up in the index the query was answered
    from. Chunks of an index that no longer exists are left out.

    Raises:
        NotFoundError: If traces are not exposed, or the query does not exist.
    """
    _require_traces(state)
    row = state.conn.execute(
        "SELECT q.id, q.question, q.created_at, q.cost_usd, q.model, q.stop_reason, q.trace, q.index_id, "
        "v.repo_id FROM queries q LEFT JOIN index_versions v ON v.id = q.index_id WHERE q.id = ?",
        (query_id,),
    ).fetchone()
    if row is None:
        raise NotFoundError(f"no query {query_id}")
    trace = json.loads(row["trace"]) if row["trace"] else None
    return {
        "query_id": row["id"],
        "repo_id": row["repo_id"],
        "question": row["question"],
        "created_at": row["created_at"],
        "cost_usd": row["cost_usd"],
        "model": row["model"],
        "stop_reason": row["stop_reason"],
        "trace": trace,
        "chunks": _trace_chunks(state, row["index_id"], trace) if trace else {},
    }


def _require_traces(state: ServerState) -> None:
    if not state.settings.expose_traces:
        raise NotFoundError("stored queries are not served here; set CQA_EXPOSE_TRACES=1 to serve them")


def _trace_chunks(state: ServerState, index_id: str, trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = [e for ranked in trace.get("lists", {}).values() for e in ranked]
    entries += trace.get("fused", []) + trace.get("reranked", []) + trace.get("context", [])
    ids = list(dict.fromkeys(int(e["chunk_id"]) for e in entries))
    if not ids:
        return {}
    rows = state.conn.execute(
        "SELECT c.id, f.path, c.start_line, c.end_line, c.kind, c.qualified_symbol "
        "FROM chunks c JOIN files f ON f.id = c.file_id "
        f"WHERE c.index_id = ? AND c.id IN ({','.join('?' * len(ids))})",
        [index_id, *ids],
    )
    return {
        str(r["id"]): {
            "path": r["path"],
            "start_line": r["start_line"],
            "end_line": r["end_line"],
            "kind": r["kind"],
            "symbol": r["qualified_symbol"],
        }
        for r in rows
    }


def metrics_summary(state: ServerState) -> dict[str, Any]:
    """Return median and 95th-percentile latency per stage, time to first token, and today's cost.

    Percentiles cover the latest ``METRIC_WINDOW`` queries, nearest-rank;
    ``ttft`` is one of the stages.
    """
    rows = state.conn.execute(
        "SELECT timings_ms FROM queries WHERE timings_ms IS NOT NULL ORDER BY id DESC LIMIT ?",
        (METRIC_WINDOW,),
    ).fetchall()
    by_stage: dict[str, list[float]] = {}
    for r in rows:
        for stage, ms in json.loads(r["timings_ms"]).items():
            by_stage.setdefault(stage, []).append(float(ms))
    today = datetime.now(UTC).date().isoformat()
    cost = state.conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM queries WHERE created_at >= ?", (today,)
    ).fetchone()[0]
    return {
        "queries": len(rows),
        "stages": {
            s: {"p50": _percentile(v, 50), "p95": _percentile(v, 95)} for s, v in sorted(by_stage.items())
        },
        "cost_today_usd": float(cost),
        "spend_cap_usd": state.spend_cap.daily_usd,
    }


def _percentile(values: list[float], p: int) -> float:
    ordered = sorted(values)
    rank = max(1, -(-p * len(ordered) // 100))
    return ordered[rank - 1]


def list_eval_runs(state: ServerState) -> list[dict[str, Any]]:
    """Return a summary of every evaluation run, newest first."""
    rows = state.conn.execute(
        "SELECT id, created_at, split, arms, hypothesis, outcome, config_hash, cost_usd FROM eval_runs "
        "ORDER BY created_at DESC"
    )
    return [{**dict(r), "arms": json.loads(r["arms"])} for r in rows]


def get_eval_run(state: ServerState, run_id: str) -> dict[str, Any]:
    """Return one run's metrics with intervals, bucket counts, and per-question results.

    Raises:
        NotFoundError: If the run does not exist.
    """
    row = state.conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise NotFoundError(f"no evaluation run {run_id!r}")
    run = dict(row)
    run["arms"] = json.loads(run["arms"])
    run["metrics"] = json.loads(run["metrics"]) if run["metrics"] else None
    return run


def create_app(state: ServerState | None = None) -> Any:
    """Return the FastAPI application, with state from the environment when none is given.

    Query responses stream ``cqa.serve.sse.to_sse`` output with media type
    ``text/event-stream``. Errors map to status codes: ``InvalidRequestError``
    400, ``RepoNotAllowedError`` 403, ``NotFoundError`` 404,
    ``IndexNotReadyError`` 409, ``RateLimitedError`` 429, and
    ``SpendCapReachedError`` 503. The routes live in ``cqa.serve.app``, which
    imports FastAPI; this module does not.
    """
    from cqa.serve.app import build_app

    return build_app(state if state is not None else ServerState.from_settings(ServerSettings.from_env()))
