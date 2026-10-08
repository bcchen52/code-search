"""Server-sent events.

Queries are POST requests, so browsers read the stream with ``fetch`` and a
stream reader rather than ``EventSource``, which only issues GET requests.
"""

from __future__ import annotations

from collections.abc import Iterator

from cqa.trace import Event

HEARTBEAT = ": ping\n\n"
"""An SSE comment line, sent periodically so idle proxies keep the connection open."""


def format_event(e: Event) -> str:
    """Format one event as ``"event: <type>\\ndata: <single-line JSON>\\n\\n"``.

    Dataclass values in the payload, such as the trace in a ``done`` event,
    are serialized as JSON objects.
    """
    raise NotImplementedError


def to_sse(events: Iterator[Event]) -> Iterator[str]:
    """Format each event. If the iterator raises, emit one ``error`` event naming the stage, then stop."""
    raise NotImplementedError
