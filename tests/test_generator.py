"""The streaming generator, against a fake client unless live tests are enabled."""

import importlib.util
import os
from types import SimpleNamespace

import pytest

from cqa.config import load_config
from cqa.errors import ConfigError, PromptNotFoundError
from cqa.generate.llm import FALLBACK_BETA, AnthropicGenerator, LlmCache
from cqa.types import Context

from helpers import ROOT, chunk

CTX = Context(
    chunks=[chunk(7, "src/auth/session.py", 88, 90, kind="method", symbol="SessionManager.validate_token")],
    token_count=40,
    low_confidence=False,
)


def gen_cfg(**changes):
    return load_config(ROOT / "configs/base.yaml").generate.model_copy(update=changes)


class FakeStream:
    def __init__(self, deltas, model, stop_reason, fail_after=None, iterations=None):
        self.deltas, self.model, self.stop_reason, self.fail_after = deltas, model, stop_reason, fail_after
        self.iterations = iterations

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        for i, d in enumerate(self.deltas):
            if self.fail_after is not None and i == self.fail_after:
                raise ConnectionError("stream dropped")
            yield d

    def get_final_message(self):
        usage = SimpleNamespace(
            input_tokens=120,
            output_tokens=30,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
            iterations=self.iterations,
        )
        return SimpleNamespace(model=self.model, stop_reason=self.stop_reason, usage=usage)


class FakeClient:
    def __init__(self, **stream_kwargs):
        self.requests = []
        self.stream_kwargs = stream_kwargs
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        return FakeStream(**self.stream_kwargs)


def generator(client, cache=None, repeat=0, **cfg_changes):
    gen = AnthropicGenerator(gen_cfg(**cfg_changes), "toy", "abc123", cache=cache, repeat=repeat)
    gen._client = client
    return gen


def answer(**overrides):
    kwargs = {
        "deltas": ["Tokens are ", "checked [C1]."],
        "model": "claude-sonnet-5-5",
        "stop_reason": "end_turn",
    }
    return FakeClient(**{**kwargs, **overrides})


def test_the_request_carries_thinking_effort_and_fallback():
    client = answer()
    list(generator(client).stream("q", CTX))
    (req,) = client.requests
    assert req["model"] == "claude-sonnet-5-5"
    assert req["thinking"] == {"type": "between_tools"}
    assert req["output_config"] == {"effort": "medium"}
    assert req["betas"] == [FALLBACK_BETA] and req["fallbacks"] == "default"
    assert "temperature" not in req
    assert "[C1] src/auth/session.py L88-90" in req["messages"][0]["content"]


def test_adaptive_thinking():
    client = answer()
    list(generator(client, thinking="adaptive", effort="xhigh").stream("q", CTX))
    assert client.requests[0]["thinking"] == {"type": "adaptive"}


def test_deltas_stream_in_order_and_usage_comes_after():
    generation = generator(answer()).stream("q", CTX)
    assert generation.usage is None and len(generation.prompt_hash) == 64
    assert "".join(generation) == "Tokens are checked [C1]."
    u = generation.usage
    assert (u.tokens_in, u.tokens_out, u.model, u.stop_reason, u.cached) == (
        120,
        30,
        "claude-sonnet-5-5",
        "end_turn",
        False,
    )


def test_a_cache_hit_costs_no_call(caches_db):
    cache = LlmCache(caches_db)
    first = generator(answer(), cache=cache).stream("q", CTX)
    text = "".join(first)
    client = answer()
    second = generator(client, cache=cache).stream("q", CTX)
    assert list(second) == [text]
    assert client.requests == []
    assert second.usage.cached and second.usage.stop_reason == "end_turn"
    assert second.prompt_hash == first.prompt_hash


def test_each_repeat_gets_its_own_answer(caches_db):
    cache = LlmCache(caches_db)
    list(generator(answer(), cache=cache, repeat=0).stream("q", CTX))
    client = answer()
    list(generator(client, cache=cache, repeat=1).stream("q", CTX))
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"stop_reason": "refusal"},
        {"stop_reason": "max_tokens"},
        {"model": "claude-sonnet-5"},
    ],
    ids=["refusal", "truncated", "fallback-model"],
)
def test_only_normal_answers_from_the_requested_model_are_cached(caches_db, overrides):
    cache = LlmCache(caches_db)
    generation = generator(answer(**overrides), cache=cache).stream("q", CTX)
    list(generation)
    assert generation.usage.stop_reason == overrides.get("stop_reason", "end_turn")
    assert generation.usage.model == overrides.get("model", "claude-sonnet-5-5")
    assert caches_db.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 0


def test_a_stream_that_fails_midway_caches_nothing(caches_db):
    generation = generator(answer(fail_after=1), cache=LlmCache(caches_db)).stream("q", CTX)
    with pytest.raises(ConnectionError):
        list(generation)
    assert generation.usage is None
    assert caches_db.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 0


def test_an_unknown_prompt_fails_before_any_request():
    client = answer()
    with pytest.raises(PromptNotFoundError):
        generator(client, prompt_version="no-such-prompt").stream("q", CTX)
    assert client.requests == []


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_thinking_off_above_high_effort_is_refused_before_any_request(effort):
    client = answer()
    with pytest.raises(ConfigError, match="adaptive"):
        generator(client, effort=effort).stream("q", CTX)
    assert client.requests == []


@pytest.mark.skipif(
    not os.environ.get("CQA_LIVE_TESTS")
    or not os.environ.get("ANTHROPIC_API_KEY")
    or importlib.util.find_spec("anthropic") is None,
    reason="calls the API (about $0.01): set CQA_LIVE_TESTS=1 and ANTHROPIC_API_KEY with the [llm] extra",
)
def test_a_live_answer():
    ctx = Context(
        chunks=[
            chunk(
                1,
                "src/math.py",
                1,
                2,
                kind="function",
                symbol="double",
                text="def double(x):\n    return 2 * x",
            )
        ],
        token_count=20,
        low_confidence=False,
    )
    generation = AnthropicGenerator(gen_cfg(max_tokens=200), "toy", "abc123").stream(
        "In one sentence, what does double return?", ctx
    )
    text = "".join(generation)
    assert text.strip()
    assert generation.usage.stop_reason == "end_turn" and generation.usage.tokens_in > 0
    assert generation.usage.model.startswith("claude-")


def test_a_fallback_records_every_attempt():
    iterations = [
        SimpleNamespace(
            type="message",
            model="claude-sonnet-5-5",
            input_tokens=800,
            output_tokens=12,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        ),
        SimpleNamespace(
            type="fallback_message",
            model="claude-sonnet-5",
            input_tokens=820,
            output_tokens=30,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    ]
    generation = generator(answer(model="claude-sonnet-5", iterations=iterations)).stream("q", CTX)
    list(generation)
    assert [(a.model, a.tokens_in, a.tokens_out) for a in generation.usage.attempts] == [
        ("claude-sonnet-5-5", 800, 12),
        ("claude-sonnet-5", 820, 30),
    ]


def test_a_single_attempt_records_no_attempt_list():
    single = [
        SimpleNamespace(
            type="message",
            model="claude-sonnet-5-5",
            input_tokens=120,
            output_tokens=30,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
        )
    ]
    generation = generator(answer(iterations=single)).stream("q", CTX)
    list(generation)
    assert generation.usage.attempts == ()
