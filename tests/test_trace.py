"""Trace serialization: what the API stores, evaluation can load."""

from cqa.trace import CitationRecord, ContextEntry, Trace
from cqa.types import Scored


def test_round_trip_restores_nested_types():
    trace = Trace(
        question="Where is the session token validated?",
        index_id="ab12cd34",
        config_hash="c0ffee",
        lists={"dense": [Scored(7, 0.83, 1, "dense"), Scored(3, 0.79, 2, "dense")]},
        fused=[Scored(7, 0.0164, 1, "fused")],
        reranked=[Scored(7, 0.0, 1, "rerank")],
        context=[ContextEntry("C1", 7, "src/auth/session.py", 88, 121, (88, 121))],
        context_tokens=412,
        answer="Tokens are checked in `validate_token` [C1:L95-105].",
        citations=[CitationRecord("C1", (95, 105), 0, "valid")],
        timings_ms={"dense": 41.0, "generate": 1830.5},
        tokens_in=2100,
        tokens_out=64,
        cost_usd=0.004,
    )
    back = Trace.from_json(trace.to_json())
    assert back == trace
    assert isinstance(back.lists["dense"][0], Scored)
    assert isinstance(back.context[0].citable, tuple)
    assert back.citations[0].lines == (95, 105)


def test_json_is_one_line():
    assert "\n" not in Trace(question="two\nlines", index_id="x", config_hash="y").to_json()
