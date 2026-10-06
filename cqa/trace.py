"""Records of how an answer was produced.

A ``Trace`` captures one answer end to end: the ranked lists, the context,
the answer, its citations, timings, and cost. The API stores one per query
and the evaluation harness one per question, arm, and repeat, in the same
JSON form. ``Event`` is the incremental form streamed to clients; the final
``done`` event carries the complete trace.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

from cqa.types import Scored

EventType = Literal["sources", "token", "citations", "done", "error"]


@dataclass(frozen=True)
class Event:
    """One step of an answer stream.

    Attributes:
        type: ``sources`` (context labels and spans), ``token`` (a text delta),
            ``citations`` (verification results), ``done`` (the trace), or
            ``error`` (the failing stage and a message).
        data: The event payload.
    """

    type: EventType
    data: dict[str, Any]


@dataclass(frozen=True)
class ContextEntry:
    """One labeled excerpt as it was shown to the model."""

    label: str
    chunk_id: int
    path: str
    start_line: int
    end_line: int
    citable: tuple[int, int]


@dataclass(frozen=True)
class CitationRecord:
    """One parsed citation and its verification status.

    Attributes:
        label: The cited excerpt label, such as ``C3``.
        lines: The narrowed line range, or None for a whole-excerpt citation.
        sentence: Index of the sentence the citation belongs to.
        status: ``valid``, ``fabricated`` (unknown label), or ``out_of_range``.
    """

    label: str
    lines: tuple[int, int] | None
    sentence: int
    status: Literal["valid", "fabricated", "out_of_range"]


@dataclass
class Trace:
    """Everything recorded about one answer.

    ``model`` is the model that produced the answer, which differs from the
    configured one after a fallback; ``stop_reason`` is why generation
    stopped. On a refusal ``answer`` is empty: any partial text was
    discarded. ``malformed`` lists citation attempts that broke the grammar.
    """

    question: str
    index_id: str
    config_hash: str
    arm: str = "real"
    lists: dict[str, list[Scored]] = field(default_factory=dict)
    fused: list[Scored] = field(default_factory=list)
    reranked: list[Scored] = field(default_factory=list)
    top_score: float | None = None
    low_confidence: bool = False
    context: list[ContextEntry] = field(default_factory=list)
    context_tokens: int = 0
    prompt_version: str = ""
    prompt_hash: str = ""
    answer: str = ""
    citations: list[CitationRecord] = field(default_factory=list)
    uncited_sentences: list[int] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    model: str = ""
    stop_reason: str | None = None
    malformed: list[str] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> str:
        """Serialize to a single line of JSON; nested dataclasses become objects and tuples lists."""
        return json.dumps(asdict(self), separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Trace:
        """Parse a trace written by ``to_json``, restoring nested types.

        ``Trace.from_json(t.to_json()) == t`` holds for every trace. Keys this
        version does not know are ignored, and missing ones take their
        defaults, so stored traces stay readable as fields are added.
        """
        raw = json.loads(text)
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in raw.items() if k in known}
        d["lists"] = {name: [Scored(**s) for s in ranked] for name, ranked in d.get("lists", {}).items()}
        d["fused"] = [Scored(**s) for s in d.get("fused", [])]
        d["reranked"] = [Scored(**s) for s in d.get("reranked", [])]
        d["context"] = [ContextEntry(**{**e, "citable": tuple(e["citable"])}) for e in d.get("context", [])]
        d["citations"] = [
            CitationRecord(**{**c, "lines": tuple(c["lines"]) if c["lines"] is not None else None})
            for c in d.get("citations", [])
        ]
        return cls(**d)
