"""Model clients and the response cache.

Responses are cached by model, parameters, prompt, and repeat index. The
repeat index matters: repeated runs exist to measure run-to-run variation,
and without it every repeat would return the first cached response.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Protocol

from cqa.config import GenerateConfig
from cqa.types import Context, Generation, Usage


class CompletionClient(Protocol):
    """Sends one prompt to a model and returns the complete reply.

    The evaluation judge talks to its provider through this interface, so the
    judge's model family can differ from the generator's.
    """

    def complete(self, model: str, prompt: str, temperature: float, max_tokens: int) -> tuple[str, Usage]: ...


class LlmCache:
    """Completed model responses, keyed by everything that determines them."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @staticmethod
    def key(model: str, params: dict[str, Any], prompt: str, repeat: int) -> str:
        """Return the cache key for one call.

        The key is a SHA-256 hex digest over the model, the canonical JSON of
        ``params``, the prompt, and the repeat index.
        """
        raise NotImplementedError

    def get(self, key: str) -> tuple[str, Usage] | None:
        """Return a cached response and the usage of the call that produced it, or None."""
        raise NotImplementedError

    def put(self, key: str, text: str, usage: Usage) -> None:
        """Store a completed response. Callers never store partial output from a failed stream."""
        raise NotImplementedError


class AnthropicGenerator:
    """Streams answers from an Anthropic model. Reads ``ANTHROPIC_API_KEY`` on first use."""

    def __init__(
        self,
        cfg: GenerateConfig,
        repo: str,
        sha: str,
        cache: LlmCache | None = None,
        repeat: int = 0,
    ) -> None:
        self.cfg = cfg
        self.repo = repo
        self.sha = sha
        self.cache = cache
        self.repeat = repeat
        self._client: Any = None

    def stream(self, question: str, ctx: Context) -> Generation:
        """Render ``cfg.prompt_version`` for the question and context, and start streaming the reply.

        The returned generation yields text deltas; its ``usage`` is set when
        the stream ends. A cache hit yields the cached text as a single delta,
        with ``usage.cached`` set. Only a completed stream is cached.

        Raises:
            PromptNotFoundError: If the configured prompt version does not exist.
        """
        raise NotImplementedError


def cost_usd(model: str, usage: Usage, prices: dict[str, Any]) -> float:
    """Return the dollar cost of one call from the dated price table in ``configs/prices.yaml``.

    Prompt-cache reads and writes are priced at the model's ``cached_input``
    and ``cache_write`` rates. A response from the local cache costs nothing.

    Raises:
        ConfigError: If the model has no price entry. An unpriced model is an
            error rather than a silent zero.
    """
    raise NotImplementedError
