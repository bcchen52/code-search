"""Guardrails for public deployments: a rate limit per client, a daily spend cap, and an allowlist."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Set
from datetime import date, timedelta
from urllib.parse import urlsplit

from cqa.errors import RepoNotAllowedError

WINDOW_S = 60.0


class RateLimiter:
    """A sliding one-minute window per client address."""

    def __init__(self, per_minute: int, max_clients: int = 1024) -> None:
        self.per_minute = per_minute
        self.max_clients = max_clients
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, ip: str, now: float) -> bool:
        """Return whether a request is within the limit, recording it only when allowed.

        The client's timestamps older than 60 seconds are discarded first, so
        refused requests never extend a client's lockout. Safe to call from
        concurrent request threads. Clients idle for a full window are
        forgotten once more than ``max_clients`` are tracked, so memory stays
        bounded.

        Args:
            ip: Client address.
            now: Current time in seconds.
        """
        with self._lock:
            if len(self._hits) > self.max_clients:
                self._hits = {k: v for k, v in self._hits.items() if v and v[-1] > now - WINDOW_S}
            recent = [t for t in self._hits.get(ip, ()) if t > now - WINDOW_S]
            allowed = len(recent) < self.per_minute
            if allowed:
                recent.append(now)
            if recent:
                self._hits[ip] = recent
            else:
                self._hits.pop(ip, None)
            return allowed


class SpendCap:
    """A daily limit on model spend, summed from ``queries.cost_usd``."""

    def __init__(self, daily_usd: float) -> None:
        self.daily_usd = daily_usd

    def exceeded(self, conn: sqlite3.Connection, today: str) -> bool:
        """Return True when spend on ``today`` (a UTC date, ``YYYY-MM-DD``) has reached the cap.

        Queries already running when the cap is reached still finish, so a
        day's spend can pass the cap by the cost of the answers in flight. A
        cap of 0 refuses every query.

        Raises:
            ValueError: If ``today`` is not a ``YYYY-MM-DD`` date.
        """
        tomorrow = (date.fromisoformat(today) + timedelta(days=1)).isoformat()
        spent = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM queries WHERE created_at >= ? AND created_at < ?",
            (today, tomorrow),
        ).fetchone()[0]
        return float(spent) >= self.daily_usd


def normalize_repo_url(url: str) -> str:
    """Return the canonical form of an https repository URL.

    The scheme and host are lowercased, and credentials, a trailing slash,
    and a ``.git`` suffix are removed, so equivalent spellings compare equal.

    Example:
        ``normalize_repo_url("https://user:token@GitHub.com/acme/toy.git/")``
        returns ``"https://github.com/acme/toy"``.

    Raises:
        RepoNotAllowedError: If the URL is not an https URL with a host and a
            path: local paths and other transports are never served. A query,
            a fragment, or an empty, ``.``, or ``..`` path segment is refused
            too, so the repository cannot change after the allowlist check.
    """
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError as e:
        raise RepoNotAllowedError(f"not a repository URL: {url!r}") from e
    if parts.scheme.lower() != "https" or not parts.hostname or parts.query or parts.fragment:
        raise RepoNotAllowedError(f"only plain https repository URLs are served, not {url!r}")
    segments = parts.path.rstrip("/").removesuffix(".git").split("/")[1:]
    if not segments or any(s in ("", ".", "..") for s in segments):
        raise RepoNotAllowedError(f"not a repository path: {url!r}")
    host = parts.hostname.lower() + (f":{port}" if port is not None else "")
    return f"https://{host}/{'/'.join(segments)}"


def repo_allowed(url: str, allowlist: Set[str] | None) -> bool:
    """Return True when a repository may be indexed and served.

    Only https URLs qualify. With no allowlist, every https repository is
    allowed; otherwise its normalized URL must be on the list. Server settings
    give no allowlist only when ``CQA_ALLOW_ANY_REPO`` asks for it.
    """
    try:
        normalized = normalize_repo_url(url)
    except RepoNotAllowedError:
        return False
    return allowlist is None or normalized in allowlist
