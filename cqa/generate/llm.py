"""Model clients and the response cache.

Responses are cached by model, parameters, prompt, and repeat index. The
repeat index matters: repeated runs exist to measure run-to-run variation,
and without it every repeat would return the first cached response.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

from cqa.config import GenerateConfig, canonical_json
from cqa.errors import ConfigError
from cqa.generate.prompts import prompt_hash, render
from cqa.types import Attempt, Context, Generation, Usage

FALLBACK_BETA = "server-side-fallback-2026-07-01"
"""Beta header for server-side fallback; a refused request is retried on another model in the same call."""

_THINKING_OFF_MAX_EFFORT = ("low", "medium", "high")


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
        return hashlib.sha256(f"{model}|{canonical_json(params)}|{prompt}|{repeat}".encode()).hexdigest()

    def get(self, key: str) -> tuple[str, Usage] | None:
        """Return a cached response and the usage of the call that produced it, or None."""
        row = self.conn.execute(
            "SELECT response, tokens_in, tokens_out FROM llm_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return row["response"], Usage(tokens_in=row["tokens_in"] or 0, tokens_out=row["tokens_out"] or 0)

    def put(self, key: str, text: str, usage: Usage) -> None:
        """Store a completed response. Callers never store partial output from a failed stream."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, response, tokens_in, tokens_out, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    key,
                    text,
                    usage.tokens_in,
                    usage.tokens_out,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )


class AnthropicGenerator:
    """Streams answers from an Anthropic model. Reads ``ANTHROPIC_API_KEY`` on first use.

    Requests opt into server-side fallback, so a request the model declines
    can be answered by another model within the same call. ``Usage.model`` and
    ``Usage.stop_reason`` record which model answered and why it stopped. See
    docs/decisions/D52-generation-outcomes.md.
    """

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
        with ``usage.cached`` set. Only a stream that ends normally
        (``end_turn``) on the requested model is cached: a refusal, a
        truncated answer, a fallback answer, or a stream that fails midway is
        never stored.

        Raises:
            PromptNotFoundError: If the configured prompt version does not exist.
            ConfigError: If thinking is off at ``xhigh`` or ``max`` effort,
                which the model rejects.
        """
        if self.cfg.thinking == "off" and self.cfg.effort not in _THINKING_OFF_MAX_EFFORT:
            raise ConfigError(f"generate.effort {self.cfg.effort!r} needs thinking: adaptive")
        return _AnthropicGeneration(self, render(self.cfg.prompt_version, self.repo, self.sha, ctx, question))

    def params(self) -> dict[str, Any]:
        """The request settings that, with the model and prompt, determine a response."""
        return {
            "thinking": self.cfg.thinking,
            "effort": self.cfg.effort,
            "max_tokens": self.cfg.max_tokens,
            "fallbacks": "default",
        }

    def client(self) -> Any:
        """The Anthropic client, created once, on first use."""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client


class _AnthropicGeneration:
    """One call: iterate for text deltas; ``usage`` is set once iteration completes."""

    def __init__(self, gen: AnthropicGenerator, prompt: str) -> None:
        self.gen = gen
        self.prompt = prompt
        self.prompt_hash = prompt_hash(prompt)
        self.usage: Usage | None = None

    def __iter__(self) -> Iterator[str]:
        cfg, cache = self.gen.cfg, self.gen.cache
        key = LlmCache.key(cfg.model, self.gen.params(), self.prompt, self.gen.repeat)
        hit = cache.get(key) if cache is not None else None
        if hit is not None:
            text, usage = hit
            self.usage = replace(usage, cached=True, model=cfg.model, stop_reason="end_turn")
            if text:
                yield text
            return

        thinking = {"type": "between_tools"} if cfg.thinking == "off" else {"type": "adaptive"}
        parts: list[str] = []
        with self.gen.client().beta.messages.stream(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            thinking=thinking,
            output_config={"effort": cfg.effort},
            messages=[{"role": "user", "content": self.prompt}],
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            for text in stream.text_stream:
                parts.append(text)
                yield text
            final = stream.get_final_message()
        u = final.usage
        attempts = tuple(
            Attempt(
                model=a.model,
                tokens_in=a.input_tokens,
                tokens_out=a.output_tokens,
                cache_read_tokens=a.cache_read_input_tokens or 0,
                cache_write_tokens=a.cache_creation_input_tokens or 0,
            )
            for a in (getattr(u, "iterations", None) or [])
            if getattr(a, "model", None)
        )
        self.usage = Usage(
            tokens_in=u.input_tokens,
            tokens_out=u.output_tokens,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            cache_write_tokens=u.cache_creation_input_tokens or 0,
            model=final.model,
            stop_reason=final.stop_reason,
            attempts=attempts if len(attempts) > 1 else (),
        )
        if cache is not None and final.stop_reason == "end_turn" and final.model == cfg.model:
            cache.put(key, "".join(parts), self.usage)


def cost_usd(model: str, usage: Usage, prices: dict[str, Any]) -> float:
    """Return the dollar cost of one call from the dated price table in ``configs/prices.yaml``.

    Prompt-cache reads and writes are priced at the model's ``cached_input``
    and ``cache_write`` rates. A response from the local cache costs nothing.
    When ``usage.attempts`` lists several attempts (a refusal that fell back
    to another model), each is priced at its own model's rates and ``model``
    is not used; this counts every reported attempt, an upper bound on the
    bill.

    Example:
        At $2 input and $10 output per million tokens, 3,300 tokens in and
        400 out cost ``(3300 * 2 + 400 * 10) / 1e6 = 0.0106``.

    Raises:
        ConfigError: If a model has no price entry, or a price the call needs
            is null. An unpriced model is an error rather than a silent zero.
    """
    if usage.cached:
        return 0.0
    attempts = usage.attempts or (
        Attempt(model, usage.tokens_in, usage.tokens_out, usage.cache_read_tokens, usage.cache_write_tokens),
    )
    return sum(_attempt_cost(a, prices) for a in attempts)


def _attempt_cost(a: Attempt, prices: dict[str, Any]) -> float:
    table = prices.get("usd_per_million_tokens") or {}
    if a.model not in table:
        raise ConfigError(f"configs/prices.yaml: no prices for {a.model!r}")
    rates = table[a.model] or {}
    total = 0.0
    for tokens, rate in (
        (a.tokens_in, "input"),
        (a.tokens_out, "output"),
        (a.cache_read_tokens, "cached_input"),
        (a.cache_write_tokens, "cache_write"),
    ):
        if tokens:
            if rates.get(rate) is None:
                raise ConfigError(f"configs/prices.yaml: {a.model} has no {rate} price")
            total += tokens * float(rates[rate])
    return total / 1_000_000


def load_prices(path: Path) -> dict[str, Any]:
    """Load and check the dated price table.

    Raises:
        ConfigError: If the file is missing or unreadable, has no ``as_of``
            date, or holds a price that is neither a number nor null.
    """
    try:
        prices = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read the price table {path}: {e}") from e
    if not isinstance(prices, dict) or not prices.get("as_of"):
        raise ConfigError(f"{path}: the price table needs an as_of date")
    table = prices.get("usd_per_million_tokens")
    if not isinstance(table, dict):
        raise ConfigError(f"{path}: usd_per_million_tokens must map models to prices")
    for model, rates in table.items():
        for rate, value in (rates or {}).items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
                raise ConfigError(f"{path}: {model}.{rate} must be a number or null, got {value!r}")
    return prices
