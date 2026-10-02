"""The single path from a question to a verified, cited answer.

``cqa ask``, the HTTP API, and the evaluation harness all answer through
``Answerer``, so evaluation measures exactly what users receive. Evaluation
arms that supply their own context enter at ``Answerer.generate``.

An answer streams as events: ``sources``, then ``token`` repeatedly, then
``citations``, then ``done``, or ``error`` at any point with the failing
stage. Sources are sent before the first token, so a client can show which
files are in play while the model is still generating.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cqa.config import Config
from cqa.retrieve.pipeline import RetrievalPipeline
from cqa.trace import Event, Trace
from cqa.types import Context, Generator, Scored


class Answerer:
    """Answers questions about one indexed repository."""

    def __init__(
        self,
        cfg: Config,
        pipeline: RetrievalPipeline | None,
        generator: Generator,
        repo: str,
        commit: str,
        config_hash: str,
        prices: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self.generator = generator
        self.repo = repo
        self.commit = commit
        self.config_hash = config_hash
        self.prices = prices

    def stream(self, question: str) -> Iterator[Event]:
        """Retrieve, assemble the context, and continue with ``generate``.

        Raises:
            ValueError: If the answerer was built without a retrieval pipeline.
        """
        raise NotImplementedError

    def generate(self, question: str, ctx: Context, trace: Trace) -> Iterator[Event]:
        """Answer from a given context.

        Emits ``sources``, the generator's ``token`` deltas, ``citations`` with
        each citation's verification status, and ``done`` carrying the
        completed trace. Records each stage's duration in ``trace.timings_ms``,
        including time to first token, the prompt hash and token usage from
        the call's ``Generation``, and its cost when prices are set. An
        answerer holds no per-call state, so one instance can serve
        concurrent questions.
        """
        raise NotImplementedError

    def run(self, question: str) -> Trace:
        """Answer a question and return the trace from its ``done`` event.

        Raises:
            CqaError: If the stream ends with an ``error`` event.
        """
        raise NotImplementedError


def sources_payload(ctx: Context, reranked: list[Scored]) -> dict[str, Any]:
    """Build the ``sources`` event: each label's path, span, and rerank score, and the low-confidence flag."""
    raise NotImplementedError


def build_answerer(cfg: Config, repo: str, commit: str, data_dir: Path) -> Answerer:
    """Assemble an answerer from configuration.

    Builds or reuses the repository's index, then wires the embedder and its
    cache, the retrieval pipeline, and the generator and its cache.
    """
    raise NotImplementedError
