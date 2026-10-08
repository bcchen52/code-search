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

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cqa.chunking import make_chunker
from cqa.config import Config, config_hash
from cqa.db import DB_FILES, connect, init_schema
from cqa.embed import make_embedder
from cqa.errors import CqaError
from cqa.generate.citations import find_malformed, parse_citations
from cqa.generate.context import assemble
from cqa.generate.llm import AnthropicGenerator, LlmCache, cost_usd, load_prices
from cqa.generate.verify import verify
from cqa.index.build import build_index, checkout_dir, open_index, resolve_repo
from cqa.ingest.walker import clone_at
from cqa.retrieve.pipeline import RetrievalPipeline
from cqa.trace import CitationRecord, ContextEntry, Event, Trace
from cqa.types import Context, Generator, Scored


def _ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000


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

        Retrieval timings are recorded under their own names, with the whole
        retrieval as ``retrieve``; a failure emits ``error`` with stage
        ``retrieve`` or ``context``.

        Raises:
            ValueError: If the answerer was built without a retrieval pipeline.
        """
        if self.pipeline is None:
            raise ValueError("this answerer has no retrieval pipeline; call generate with a context")
        trace = Trace(question=question, index_id=self.pipeline.index.index_id, config_hash=self.config_hash)
        stage = "retrieve"
        try:
            r = self.pipeline.run(question)
            trace.lists, trace.fused, trace.reranked = r.lists, r.fused, r.reranked
            trace.top_score, trace.low_confidence = r.top_score, r.low_confidence
            trace.timings_ms.update({k: v for k, v in r.timings_ms.items() if k != "total"})
            trace.timings_ms["retrieve"] = r.timings_ms.get("total", 0.0)
            stage = "context"
            t = time.perf_counter()
            ctx = assemble(
                r.reranked, r.chunks, self.cfg.context, r.low_confidence, self.pipeline.index.file_lines
            )
            trace.timings_ms["context"] = _ms(t)
        except Exception as e:
            yield Event("error", {"stage": stage, "message": str(e)})
            return
        yield from self.generate(question, ctx, trace)

    def generate(self, question: str, ctx: Context, trace: Trace) -> Iterator[Event]:
        """Answer from a given context.

        Emits ``sources``, the generator's ``token`` deltas, ``citations`` with
        each citation's verification status, and ``done`` carrying the
        completed trace. Records each stage's duration in ``trace.timings_ms``,
        including time to first token, the prompt hash and token usage from
        the call's ``Generation``, and its cost when prices are set. An
        answerer holds no per-call state, so one instance can serve
        concurrent questions.

        On a refusal the streamed text is discarded: the trace's ``answer`` is
        empty and its ``stop_reason`` says why. A failure in generation or
        verification emits ``error`` with stage ``generate`` or ``verify``,
        and nothing after it.
        """
        started = time.perf_counter()
        labels = {f"C{i}": c for i, c in enumerate(ctx.chunks, start=1)}
        trace.context = [
            ContextEntry(label, c.id, c.path, c.start_line, c.end_line, c.citable)
            for label, c in labels.items()
        ]
        trace.context_tokens, trace.low_confidence = ctx.token_count, ctx.low_confidence
        trace.prompt_version = self.cfg.generate.prompt_version
        yield Event("sources", sources_payload(ctx, trace.reranked))

        stage = "generate"
        try:
            generation = self.generator.stream(question, ctx)
            trace.prompt_hash = generation.prompt_hash
            parts: list[str] = []
            t = time.perf_counter()
            for delta in generation:
                if not parts:
                    trace.timings_ms["ttft"] = _ms(t)
                parts.append(delta)
                yield Event("token", {"text": delta})
            trace.timings_ms["generate"] = _ms(t)
            usage = generation.usage
            if usage is None:
                raise CqaError("the model call finished without reporting usage")

            stage = "verify"
            t = time.perf_counter()
            answer = "" if usage.stop_reason == "refusal" else "".join(parts)
            citations = parse_citations(answer)
            verification = verify(answer, citations, labels)
            trace.answer = answer
            trace.citations = [
                CitationRecord(c.citation.label, c.citation.lines, c.citation.sentence, c.status)
                for c in verification.checks
            ]
            trace.uncited_sentences = verification.uncited_sentences
            trace.malformed = find_malformed(answer)
            trace.timings_ms["verify"] = _ms(t)
        except Exception as e:
            yield Event("error", {"stage": stage, "message": str(e)})
            return

        yield Event(
            "citations",
            {
                "citations": [
                    {"label": r.label, "lines": r.lines, "sentence": r.sentence, "status": r.status}
                    for r in trace.citations
                ],
                "uncited_sentences": trace.uncited_sentences,
                "malformed": trace.malformed,
            },
        )
        trace.tokens_in, trace.tokens_out, trace.cached = usage.tokens_in, usage.tokens_out, usage.cached
        trace.model, trace.stop_reason = usage.model, usage.stop_reason
        if self.prices is not None and not usage.cached:
            trace.cost_usd = cost_usd(usage.model, usage, self.prices)
        before = trace.timings_ms.get("retrieve", 0.0) + trace.timings_ms.get("context", 0.0)
        trace.timings_ms["total"] = before + _ms(started)
        yield Event(
            "done",
            {
                "trace": trace,
                "timings_ms": trace.timings_ms,
                "cost_usd": trace.cost_usd,
                "config_hash": trace.config_hash,
                "cached": trace.cached,
                "model": trace.model,
                "stop_reason": trace.stop_reason,
            },
        )

    def run(self, question: str) -> Trace:
        """Answer a question and return the trace from its ``done`` event.

        Raises:
            CqaError: If the stream ends with an ``error`` event, or without a ``done`` event.
        """
        for event in self.stream(question):
            if event.type == "error":
                raise CqaError(f"{event.data['stage']}: {event.data['message']}")
            if event.type == "done":
                trace: Trace = event.data["trace"]
                return trace
        raise CqaError("the answer stream ended without a result")


def sources_payload(ctx: Context, reranked: list[Scored]) -> dict[str, Any]:
    """Build the ``sources`` event: each label's path, span, and rerank score, and the low-confidence flag.

    A chunk that was not reranked, such as a gold excerpt in the oracle arm,
    has a ``rerank_score`` of None.
    """
    scores = {s.chunk_id: s.score for s in reranked}
    sources = [
        {
            "label": f"C{i}",
            "path": c.path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "rerank_score": scores.get(c.id),
        }
        for i, c in enumerate(ctx.chunks, start=1)
    ]
    return {"sources": sources, "low_confidence": ctx.low_confidence}


DEFAULT_PRICES = Path("configs/prices.yaml")


def build_answerer(
    cfg: Config, repo: str, commit: str | None, data_dir: Path, prices_path: Path = DEFAULT_PRICES
) -> Answerer:
    """Assemble an answerer from configuration.

    Builds or reuses the repository's index, then wires the embedder and its
    cache, the retrieval pipeline, and the generator and its cache, and
    loads the price table so every answer carries its cost.

    Args:
        cfg: The pipeline configuration.
        repo: A local repository path or an https URL.
        commit: A commit; None means ``HEAD`` of a local repository (see
            ``cqa.index.build.resolve_repo``).
        data_dir: Where the databases, checkouts, and indexes live.
        prices_path: The dated price table. A missing table is an error,
            never free answers.

    Raises:
        CqaError: If the configuration names something unavailable, the
            price table is missing or invalid, or the index cannot be built.
        ValueError: If the repository or commit is invalid.
    """
    prices = load_prices(prices_path)
    url, commit = resolve_repo(repo, commit)
    conn = connect(data_dir / DB_FILES["main"])
    init_schema(conn, "main")
    caches = connect(data_dir / DB_FILES["caches"])
    init_schema(caches, "caches")
    checkout = clone_at(url, commit, checkout_dir(data_dir, url, commit))
    embedder = make_embedder(cfg.index, caches)
    index_id = build_index(
        url, commit, checkout, cfg.index, conn, data_dir, make_chunker(cfg.index), embedder
    )
    handle = open_index(conn, index_id, data_dir, cfg.index.store)
    pipeline = RetrievalPipeline.from_config(cfg, handle, embedder)
    generator = AnthropicGenerator(cfg.generate, handle.repo_name, commit, cache=LlmCache(caches))
    return Answerer(cfg, pipeline, generator, handle.repo_name, commit, config_hash(cfg), prices)
