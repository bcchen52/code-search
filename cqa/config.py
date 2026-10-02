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

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


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
    """The answering model and prompt."""

    model: str
    temperature: float
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
    raise NotImplementedError


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` over ``base`` without modifying either.

    Mappings merge recursively; any other value in ``override`` replaces the
    value in ``base``.

    Example:
        ``deep_merge({"a": {"x": 1, "y": 2}, "l": [1, 2]}, {"a": {"y": 3}, "l": [9]})``
        returns ``{"a": {"x": 1, "y": 3}, "l": [9]}``.
    """
    raise NotImplementedError


def canonical_json(obj: Any) -> str:
    """Serialize to JSON with sorted keys and no insignificant whitespace.

    Equal values always produce the same string, whatever their key order.
    """
    raise NotImplementedError


def config_hash(cfg: Config) -> str:
    """Return the SHA-256 hex digest of the configuration's canonical JSON.

    The hash identifies a configuration on every index, query, and evaluation
    run. It changes with any value and never with key order or YAML layout.
    """
    raise NotImplementedError
