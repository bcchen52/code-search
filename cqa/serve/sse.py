"""Server-sent events.

Queries are POST requests, so browsers read the stream with ``fetch`` and a
stream reader rather than ``EventSource``, which only issues GET requests.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from typing import Any

from cqa.trace import Event

HEARTBEAT = ": ping\n\n"
"""An SSE comment line, sent periodically so idle proxies keep the connection open."""


def format_event(e: Event) -> str:
    """Format one event as ``"event: <type>\\ndata: <single-line JSON>\\n\\n"``.

    Dataclass values in the payload, such as the trace in a ``done`` event,
    are serialized as JSON objects. JSON escapes newlines inside strings, so
    one ``data:`` line always holds the whole payload.
    """
    data = json.dumps(e.data, default=_encode, separators=(",", ":"), ensure_ascii=False)
    return f"event: {e.type}\ndata: {data}\n\n"


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__} in an event")


def to_sse(events: Iterator[Event]) -> Iterator[str]:
    """Format each event. If the iterator raises, emit one ``error`` event naming the stage, then stop.

    The stage is ``stream``: the answerer reports failures in its own stages
    as ``error`` events, so an exception reaching here escaped all of them.
    """
    try:
        for event in events:
            yield format_event(event)
    except Exception as e:
        yield format_event(Event("error", {"stage": "stream", "message": str(e)}))
