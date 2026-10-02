"""The model response cache."""

from cqa.generate.llm import LlmCache
from cqa.types import Usage


def test_the_key_depends_on_every_input():
    key = LlmCache.key("m", {"temperature": 0}, "prompt", 0)
    assert len(key) == 64
    assert key == LlmCache.key("m", {"temperature": 0}, "prompt", 0)
    assert key != LlmCache.key("m", {"temperature": 0}, "prompt", 1)
    assert key != LlmCache.key("m", {"temperature": 0.5}, "prompt", 0)
    assert key != LlmCache.key("m2", {"temperature": 0}, "prompt", 0)


def test_parameter_order_does_not_matter():
    assert LlmCache.key("m", {"a": 1, "b": 2}, "p", 0) == LlmCache.key("m", {"b": 2, "a": 1}, "p", 0)


def test_round_trip(caches_db):
    cache = LlmCache(caches_db)
    assert cache.get("k") is None
    cache.put("k", "answer", Usage(tokens_in=120, tokens_out=30))
    text, usage = cache.get("k")
    assert text == "answer" and (usage.tokens_in, usage.tokens_out) == (120, 30)
