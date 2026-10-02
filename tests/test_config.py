"""Configuration loading, inheritance, validation, and hashing."""

import pytest
import yaml
from pydantic import ValidationError

from cqa.config import Config, canonical_json, config_hash, deep_merge, load_config
from cqa.errors import ConfigError

from helpers import ROOT

CONFIGS = ROOT / "configs"


def raw(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text())


@pytest.mark.parametrize("name", ["base.yaml", "design.yaml"])
def test_standalone_configs_validate(name):
    Config.model_validate(raw(name))


def test_an_unknown_key_is_an_error():
    cfg = raw("base.yaml")
    cfg["index"]["chunkr"] = "ast"
    with pytest.raises(ValidationError):
        Config.model_validate(cfg)


def test_a_retriever_needs_its_store():
    cfg = raw("base.yaml")
    cfg["retrieve"]["lists"] = ["dense", "bm25"]
    with pytest.raises(ValidationError, match="lexical"):
        Config.model_validate(cfg)


def test_deep_merge_merges_mappings_and_replaces_lists():
    base = {"a": {"x": 1, "y": 2}, "l": [1, 2]}
    override = {"a": {"y": 3}, "l": [9]}
    assert deep_merge(base, override) == {"a": {"x": 1, "y": 3}, "l": [9]}
    assert base == {"a": {"x": 1, "y": 2}, "l": [1, 2]}


def write_chain(tmp_path):
    """base.yaml <- lists.yaml <- weight.yaml, the way experiment files extend each other."""
    (tmp_path / "exp").mkdir()
    (tmp_path / "base.yaml").write_text((CONFIGS / "base.yaml").read_text())
    (tmp_path / "exp/lists.yaml").write_text(
        "extends: ../base.yaml\nindex: {stores: [vectors, lexical]}\nretrieve: {lists: [dense, bm25]}\n"
    )
    (tmp_path / "exp/weight.yaml").write_text(
        "extends: lists.yaml\nretrieve: {fusion: {weights: {bm25: 2}}}\n"
    )
    return tmp_path / "exp/weight.yaml"


def test_extends_overrides_only_what_it_names(tmp_path):
    cfg = load_config(write_chain(tmp_path))
    assert cfg.retrieve.fusion.weights["bm25"] == 2
    assert cfg.retrieve.fusion.weights["dense"] == 1
    assert cfg.retrieve.fusion.k == 60


def test_extends_chains_resolve(tmp_path):
    cfg = load_config(write_chain(tmp_path))
    assert cfg.retrieve.lists == ["dense", "bm25"]
    assert cfg.index.embedder == raw("base.yaml")["index"]["embedder"]


@pytest.mark.parametrize("path", sorted((CONFIGS / "exp").glob("*.yaml")), ids=lambda p: p.name)
def test_every_experiment_loads(path):
    load_config(path)


def test_an_extends_cycle_is_an_error(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "a.yaml")


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_canonical_json_ignores_key_order():
    assert (
        canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1}) == '{"a":[1,2],"b":1}'
    )


def test_hash_ignores_key_order_and_yaml_layout(tmp_path):
    data = raw("base.yaml")
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(data, sort_keys=True))
    reordered = dict(reversed(list(data.items())))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(reordered, sort_keys=False, default_flow_style=True))
    assert config_hash(load_config(tmp_path / "a.yaml")) == config_hash(load_config(tmp_path / "b.yaml"))


def test_hash_changes_with_any_value():
    base = load_config(CONFIGS / "base.yaml")
    changed = Config.model_validate(deep_merge(base.model_dump(), {"context": {"budget_tokens": 8000}}))
    assert len(config_hash(base)) == 64
    assert config_hash(base) != config_hash(changed)
