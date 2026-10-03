"""Pipeline configuration: YAML files validated into immutable models.

A configuration file may inherit from another with ``extends: <path>``,
resolved relative to the inheriting file. Mappings merge recursively; every
other value, lists included, replaces the inherited one. Models reject
unknown keys, so a misspelled option fails validation instead of being
ignored. Every configurable value lives in YAML, so the models declare no
defaults; the factories that build pipeline stages always pass the
configured values, whatever defaults a constructor has for direct use.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from cqa.errors import ConfigError


class StrictModel(BaseModel):
    """Base for configuration models: unknown keys are rejected and instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class IndexConfig(StrictModel):
    """How a repository is chunked, embedded, and stored."""

    chunker: Literal["fixed", "ast"]
    min_tokens: int
    target_tokens: int
    max_tokens: int
    split_overlap_lines: int
    fallback_window_lines: int
    fallback_overlap_lines: int
    skeleton_chunks: bool
    enrich_header: bool
    stores: list[Literal["vectors", "lexical", "symbols"]]
    embedder: str
    dims: int
    store: Literal["flat", "hnswlib", "cpp"]


class FusionConfig(StrictModel):
    """How ranked lists are combined."""

    method: Literal["rrf"]
    k: int
    weights: dict[str, float]


class RetrieveConfig(StrictModel):
    """Which retrievers run, and how many candidates each contributes."""

    lists: list[Literal["dense", "bm25", "symbol"]]
    dense_k: int
    bm25_k: int
    bm25_weights: tuple[float, float, float]
    symbol_refs_k: int
    fusion: FusionConfig
    candidates: int


class RerankConfig(StrictModel):
    """Reranking and the confidence gate. ``gate_tau=None`` disables the gate."""

    model: Literal["none", "cross-encoder", "llm"]
    model_id: str | None
    keep: int
    gate_tau: float | None


class ContextConfig(StrictModel):
    """How retrieved chunks become the model's context."""

    budget_tokens: int
    order: Literal["rank", "by_file_best_first", "ends"]


class GenerateConfig(StrictModel):
    """The answering model, how much it reasons, and the prompt.

    ``thinking`` is ``off`` (answer directly) or ``adaptive`` (the model decides
    how much to reason before answering); ``effort`` sets how much it spends.
    Sampling parameters such as temperature are not configurable: current
    models reject non-default values. See docs/decisions/D48-generation-knobs.md.
    """

    model: str
    thinking: Literal["off", "adaptive"]
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    max_tokens: int
    prompt_version: str


class AgentConfig(StrictModel):
    """The tool-using agent and its budgets."""

    model: str
    max_tool_calls: int
    max_input_tokens: int
    max_tool_output_tokens: int
    wall_clock_s: int


STORE_FOR_LIST = {"dense": "vectors", "bm25": "lexical", "symbol": "symbols"}
"""The index store each retriever reads."""


class Config(StrictModel):
    """A complete pipeline configuration."""

    index: IndexConfig
    retrieve: RetrieveConfig
    rerank: RerankConfig
    context: ContextConfig
    generate: GenerateConfig
    agent: AgentConfig

    @model_validator(mode="after")
    def _lists_have_stores(self) -> Config:
        missing = [name for name in self.retrieve.lists if STORE_FOR_LIST[name] not in self.index.stores]
        if missing:
            needed = [STORE_FOR_LIST[name] for name in missing]
            raise ValueError(f"retrieve.lists {missing} need index.stores {needed}")
        return self


def load_config(path: str | Path) -> Config:
    """Load a configuration file, resolving its ``extends`` chain.

    Args:
        path: The configuration file to load.

    Returns:
        The validated, fully merged configuration.

    Raises:
        ConfigError: If a file in the chain is missing, the chain has a cycle,
            or the merged configuration fails validation.
    """
    merged = _load_raw(Path(path), seen=set())
    try:
        return Config.model_validate(merged)
    except ValidationError as e:
        raise ConfigError(f"invalid configuration {path}:\n{e}") from e


def _load_raw(path: Path, seen: set[Path]) -> dict[str, Any]:
    """Read one file and merge it over its ``extends`` parent, unvalidated."""
    resolved = path.resolve()
    if resolved in seen:
        raise ConfigError(f"extends cycle: {path} is reached twice")
    seen.add(resolved)
    try:
        data = yaml.safe_load(resolved.read_text())
    except FileNotFoundError as e:
        raise ConfigError(f"configuration file not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the top level must be a mapping")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str):
        raise ConfigError(f"{path}: extends must be a relative path")
    return deep_merge(_load_raw(resolved.parent / parent, seen), data)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` over ``base`` without modifying either.

    Mappings merge recursively; any other value in ``override`` replaces the
    value in ``base``.

    Example:
        ``deep_merge({"a": {"x": 1, "y": 2}, "l": [1, 2]}, {"a": {"y": 3}, "l": [9]})``
        returns ``{"a": {"x": 1, "y": 3}, "l": [9]}``.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def canonical_json(obj: Any) -> str:
    """Serialize to JSON with sorted keys and no insignificant whitespace.

    Equal values always produce the same string, whatever their key order.
    NaN and infinities are rejected, since they have no stable JSON form.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def config_hash(cfg: Config) -> str:
    """Return the SHA-256 hex digest of the configuration's canonical JSON.

    The hash identifies a configuration on every index, query, and evaluation
    run. It changes with any value and never with key order or YAML layout.
    """
    return hashlib.sha256(canonical_json(cfg.model_dump(mode="json")).encode()).hexdigest()
