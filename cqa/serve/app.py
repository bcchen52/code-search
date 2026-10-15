"""The FastAPI application: routes over the handlers in ``cqa.serve.api``.

This module imports FastAPI, so only ``cqa.serve.api.create_app`` imports it,
and only when an application is built. It has no ``from __future__ import
annotations``, because FastAPI reads route annotations at runtime.
"""

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from cqa.errors import (
    CqaError,
    IndexNotReadyError,
    InvalidRequestError,
    NotFoundError,
    RateLimitedError,
    RepoNotAllowedError,
    SpendCapReachedError,
)
from cqa.serve import api
from cqa.serve.sse import to_sse

ERROR_STATUS: list[tuple[type[CqaError], int]] = [
    (InvalidRequestError, 400),
    (RepoNotAllowedError, 403),
    (NotFoundError, 404),
    (IndexNotReadyError, 409),
    (RateLimitedError, 429),
    (SpendCapReachedError, 503),
]
"""How request errors map to status codes; any other exception is a bug and becomes a 500."""

STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
"""Keep proxies from caching or buffering the event stream."""


class RepoRequest(BaseModel):
    """The body of ``POST /repos``."""

    model_config = ConfigDict(extra="forbid")
    url: str
    commit: str | None = None


class QueryRequest(BaseModel):
    """The body of ``POST /repos/{id}/query``."""

    model_config = ConfigDict(extra="forbid")
    question: str


def build_app(state: api.ServerState) -> FastAPI:
    """Bind the API handlers to routes and map request errors to status codes."""
    app = FastAPI(title="cqa", summary="Question answering over a codebase, with line-level citations.")

    for error, status in ERROR_STATUS:
        app.add_exception_handler(error, _error_response(status))

    @app.post("/repos", status_code=202)
    def register(body: RepoRequest) -> dict[str, Any]:
        return api.register_repo(state, body.url, body.commit)

    @app.get("/repos")
    def repos() -> list[dict[str, Any]]:
        return api.list_repos(state)

    @app.get("/jobs/{job_id}")
    def job(job_id: int) -> dict[str, Any]:
        return api.get_job(state, job_id)

    @app.post("/repos/{repo_id}/query")
    def ask(repo_id: int, body: QueryRequest, request: Request) -> StreamingResponse:
        client = request.client.host if request.client else "unknown"
        events = api.query(state, repo_id, body.question, client)
        return StreamingResponse(to_sse(events), media_type="text/event-stream", headers=STREAM_HEADERS)

    @app.get("/repos/{repo_id}/file", response_class=PlainTextResponse)
    def file(repo_id: int, path: str, start: int = 1, end: int = 200) -> str:
        return api.file_lines(state, repo_id, path, start, end)

    @app.get("/queries/{query_id}")
    def stored_query(query_id: int) -> dict[str, Any]:
        return api.get_query(state, query_id)

    @app.get("/metrics/summary")
    def metrics() -> dict[str, Any]:
        return api.metrics_summary(state)

    @app.get("/eval/runs")
    def eval_runs() -> list[dict[str, Any]]:
        return api.list_eval_runs(state)

    @app.get("/eval/runs/{run_id}")
    def eval_run(run_id: str) -> dict[str, Any]:
        return api.get_eval_run(state, run_id)

    return app


def _error_response(status: int) -> Callable[[Request, Exception], JSONResponse]:
    def handle(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status, content={"error": type(error).__name__, "message": str(error)}
        )

    return handle
