"""Server settings, read from the environment."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from cqa.errors import ConfigError, RepoNotAllowedError
from cqa.serve.guardrails import normalize_repo_url

_TRUE = ("1", "true", "yes")
_FALSE = ("", "0", "false", "no")
_N = TypeVar("_N", int, float)


@dataclass(frozen=True)
class ServerSettings:
    """Deployment settings for the API and the worker.

    Attributes:
        data_dir: Databases, indexes, and checkouts (``CQA_DATA_DIR``).
        config_path: Pipeline configuration used to answer queries (``CQA_CONFIG``).
        rate_limit_per_min: Queries allowed per client address per minute
            (``CQA_RATE_LIMIT_PER_MIN``).
        daily_spend_cap_usd: Model spend allowed per UTC day (``CQA_DAILY_SPEND_CAP_USD``).
        repo_allowlist: Normalized https URLs of the repositories that may be
            indexed and served (``CQA_REPO_ALLOWLIST``, comma-separated), or
            None to allow any https repository, which only an explicit
            ``CQA_ALLOW_ANY_REPO=1`` gives. See docs/decisions/D53-allowlist-fails-closed.md.
    """

    data_dir: Path
    config_path: Path
    rate_limit_per_min: int
    daily_spend_cap_usd: float
    repo_allowlist: frozenset[str] | None

    @classmethod
    def from_env(cls) -> ServerSettings:
        """Read settings from the environment, with defaults ``./data``, ``configs/base.yaml``, 10, and 1.0.

        There is no default allowlist: without ``CQA_REPO_ALLOWLIST``, the
        server refuses to start unless ``CQA_ALLOW_ANY_REPO`` is ``1``,
        ``true``, or ``yes``.

        Raises:
            ConfigError: If a numeric variable does not parse or is out of
                range, an allowlist entry is not an https repository URL, or
                the allowlist and ``CQA_ALLOW_ANY_REPO`` are both set, or
                neither is.
        """
        env = os.environ
        rate = _number("CQA_RATE_LIMIT_PER_MIN", "10", int)
        if rate < 1:
            raise ConfigError(f"CQA_RATE_LIMIT_PER_MIN must be at least 1, got {rate}")
        cap = _number("CQA_DAILY_SPEND_CAP_USD", "1.0", float)
        if not math.isfinite(cap) or cap < 0:
            raise ConfigError(f"CQA_DAILY_SPEND_CAP_USD must be a non-negative amount, got {cap}")

        any_repo = env.get("CQA_ALLOW_ANY_REPO", "").strip().lower()
        if any_repo not in _TRUE + _FALSE:
            raise ConfigError(f"CQA_ALLOW_ANY_REPO must be 1, true, yes, 0, false, or no, got {any_repo!r}")
        entries = [e.strip() for e in env.get("CQA_REPO_ALLOWLIST", "").split(",") if e.strip()]
        try:
            allowlist = frozenset(normalize_repo_url(e) for e in entries)
        except RepoNotAllowedError as e:
            raise ConfigError(f"CQA_REPO_ALLOWLIST: {e}") from e
        if allowlist and any_repo in _TRUE:
            raise ConfigError("set CQA_REPO_ALLOWLIST or CQA_ALLOW_ANY_REPO, not both")
        if not allowlist and any_repo not in _TRUE:
            raise ConfigError(
                "CQA_REPO_ALLOWLIST is empty: list the https repositories to serve, "
                "or set CQA_ALLOW_ANY_REPO=1 to serve any https repository"
            )
        return cls(
            data_dir=Path(env.get("CQA_DATA_DIR") or "data"),
            config_path=Path(env.get("CQA_CONFIG") or "configs/base.yaml"),
            rate_limit_per_min=rate,
            daily_spend_cap_usd=cap,
            repo_allowlist=allowlist or None,
        )


def _number(name: str, default: str, kind: type[_N]) -> _N:
    raw = os.environ.get(name, "").strip() or default
    try:
        return kind(raw)
    except ValueError as e:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from e
