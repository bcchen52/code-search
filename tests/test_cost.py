"""Cost accounting from the dated price table."""

import pytest

from cqa.errors import ConfigError
from cqa.generate.llm import cost_usd, load_prices
from cqa.types import Attempt, Usage

from helpers import ROOT

PRICES = {
    "as_of": "2026-10-01",
    "usd_per_million_tokens": {
        "model-a": {"input": 3.0, "output": 15.0, "cached_input": 0.3, "cache_write": 3.75},
        "model-b": {"input": None, "output": None},
    },
}


def test_every_kind_of_token_is_priced():
    usage = Usage(
        tokens_in=1_000_000, tokens_out=100_000, cache_read_tokens=2_000_000, cache_write_tokens=400_000
    )
    assert cost_usd("model-a", usage, PRICES) == pytest.approx(3.0 + 1.5 + 0.6 + 1.5)


def test_a_cached_response_is_free():
    assert cost_usd("model-a", Usage(tokens_in=500, tokens_out=50, cached=True), PRICES) == 0.0


@pytest.mark.parametrize("model", ["model-b", "model-c"])
def test_a_missing_price_is_an_error_not_zero(model):
    with pytest.raises(ConfigError):
        cost_usd(model, Usage(tokens_in=500, tokens_out=50), PRICES)


FALLBACK_PRICES = {
    "as_of": "2026-10-01",
    "usd_per_million_tokens": {
        "primary": {"input": 2.0, "output": 10.0},
        "fallback": {"input": 1.0, "output": 5.0},
    },
}


def test_each_attempt_is_priced_at_its_own_model():
    usage = Usage(
        tokens_in=1_000_000,
        tokens_out=100_000,
        model="fallback",
        attempts=(Attempt("primary", 1_000_000, 50_000), Attempt("fallback", 1_000_000, 100_000)),
    )
    assert cost_usd("primary", usage, FALLBACK_PRICES) == pytest.approx((2.0 + 0.5) + (1.0 + 0.5))


def test_a_fallback_to_an_unpriced_model_is_an_error():
    usage = Usage(
        tokens_in=10, tokens_out=10, attempts=(Attempt("primary", 10, 0), Attempt("unpriced", 10, 10))
    )
    with pytest.raises(ConfigError, match="unpriced"):
        cost_usd("primary", usage, FALLBACK_PRICES)


def test_a_refused_answer_still_costs_its_tokens():
    usage = Usage(tokens_in=1_000_000, tokens_out=0, stop_reason="refusal")
    assert cost_usd("primary", usage, FALLBACK_PRICES) == pytest.approx(2.0)


def test_a_needed_null_price_is_an_error_but_an_unneeded_one_is_not():
    prices = {
        "as_of": "x",
        "usd_per_million_tokens": {"m": {"input": 1.0, "output": 2.0, "cached_input": None}},
    }
    assert cost_usd("m", Usage(tokens_in=1_000_000, tokens_out=0), prices) == pytest.approx(1.0)
    with pytest.raises(ConfigError, match="cached_input"):
        cost_usd("m", Usage(tokens_in=0, tokens_out=0, cache_read_tokens=10), prices)


def test_the_real_price_table_prices_a_baseline_answer():
    prices = load_prices(ROOT / "configs/prices.yaml")
    assert prices["as_of"]
    cost = cost_usd("claude-sonnet-5-5", Usage(tokens_in=3300, tokens_out=400), prices)
    assert cost == pytest.approx(0.0106)
    assert "claude-sonnet-5" in prices["usd_per_million_tokens"]


@pytest.mark.parametrize(
    "text, message",
    [
        ("usd_per_million_tokens: {m: {input: 1.0}}\n", "as_of"),
        ('as_of: "2026-10-01"\nusd_per_million_tokens: {m: {input: cheap}}\n', "number or null"),
        ('as_of: "2026-10-01"\nusd_per_million_tokens: {m: {input: true}}\n', "number or null"),
        ('as_of: "2026-10-01"\n', "usd_per_million_tokens"),
    ],
)
def test_load_prices_rejects_bad_tables(tmp_path, text, message):
    (tmp_path / "prices.yaml").write_text(text)
    with pytest.raises(ConfigError, match=message):
        load_prices(tmp_path / "prices.yaml")


def test_a_missing_price_table_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_prices(tmp_path / "missing.yaml")
