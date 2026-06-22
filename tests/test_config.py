"""Tests for ConfigLoader model/retrieve_model env-override unification (P7).

Precedence contract (Approach A):
    caller-explicit (user_opt kwarg)  >  MODEL_NAME / RETRIEVE_MODEL_NAME env  >  config.yaml default

These tests pin the behavior added in P7 so the CLI (main.py) and the server
path (PageIndexClient -> ConfigLoader) share ONE resolution path for model names.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pageindex_mutil.utils import ConfigLoader


# Keys that ConfigLoader must NOT treat as unknown when they come from the
# caller (they exist in the default config.yaml). Env vars are never passed
# through _validate_keys because they are applied as merged dict values, not keys.


class TestConfigLoaderModelEnv:
    """MODEL_NAME env overrides config.yaml `model` default."""

    def test_model_name_env_overrides_yaml_default(self, monkeypatch):
        monkeypatch.setenv("MODEL_NAME", "gpt-4o-mini")
        # Ensure no explicit caller override interferes.
        cfg = ConfigLoader().load(None)
        assert cfg.model == "gpt-4o-mini"

    def test_retrieve_model_name_env_overrides_yaml_default(self, monkeypatch):
        monkeypatch.setenv("RETRIEVE_MODEL_NAME", "text-embedding-3-small")
        cfg = ConfigLoader().load(None)
        assert cfg.retrieve_model == "text-embedding-3-small"

    def test_no_env_falls_back_to_yaml_default(self, monkeypatch):
        # Clean env to guarantee the config.yaml default is used.
        monkeypatch.delenv("MODEL_NAME", raising=False)
        monkeypatch.delenv("RETRIEVE_MODEL_NAME", raising=False)
        cfg = ConfigLoader().load(None)
        # config.yaml ships qwen-plus for both (see pageindex_mutil/config.yaml).
        assert cfg.model == "qwen-plus"
        assert cfg.retrieve_model == "qwen-plus"

    def test_precedence_explicit_kwarg_beats_env(self, monkeypatch):
        """Caller-explicit model in load(user_opt) must win over MODEL_NAME env."""
        monkeypatch.setenv("MODEL_NAME", "env-should-lose")
        cfg = ConfigLoader().load({"model": "explicit-wins"})
        assert cfg.model == "explicit-wins"

    def test_precedence_explicit_kwarg_beats_env_retrieve(self, monkeypatch):
        monkeypatch.setenv("RETRIEVE_MODEL_NAME", "env-should-lose")
        cfg = ConfigLoader().load({"retrieve_model": "explicit-r-wins"})
        assert cfg.retrieve_model == "explicit-r-wins"

    def test_precedence_env_beats_yaml_default(self, monkeypatch):
        """MODEL_NAME env must win over the config.yaml default when no kwarg."""
        monkeypatch.setenv("MODEL_NAME", "env-over-yaml")
        cfg = ConfigLoader().load(None)
        assert cfg.model == "env-over-yaml"

    def test_env_does_not_create_unknown_keys(self, monkeypatch):
        """Env overrides only touch existing keys (model/retrieve_model); they must
        not surface as unknown-key errors from _validate_keys."""
        monkeypatch.setenv("MODEL_NAME", "gpt-4o-mini")
        monkeypatch.setenv("RETRIEVE_MODEL_NAME", "gpt-4o-mini")
        # If env leaked into _validate_keys as dict keys, this would raise ValueError.
        cfg = ConfigLoader().load(None)
        assert cfg.model == "gpt-4o-mini"
        assert cfg.retrieve_model == "gpt-4o-mini"

    def test_combined_env_and_kwarg(self, monkeypatch):
        """MODEL_NAME env sets default, explicit kwarg overrides only model;
        retrieve_model still comes from RETRIEVE_MODEL_NAME env."""
        monkeypatch.setenv("MODEL_NAME", "env-model")
        monkeypatch.setenv("RETRIEVE_MODEL_NAME", "env-rmodel")
        cfg = ConfigLoader().load({"model": "kwarg-model"})
        assert cfg.model == "kwarg-model"
        assert cfg.retrieve_model == "env-rmodel"

    # --- FIX 1 (rework iteration): whitespace / empty env value handling ---
    # Contract: env values that are empty OR whitespace-only must fall back to
    # the config.yaml default rather than corrupting the model name; non-empty
    # values must be stripped (leading/trailing whitespace removed).

    def test_whitespace_model_name_falls_back_to_yaml(self, monkeypatch):
        """MODEL_NAME=' ' (whitespace only) must NOT override yaml — fall back to qwen-plus."""
        monkeypatch.setenv("MODEL_NAME", " ")
        cfg = ConfigLoader().load(None)
        assert cfg.model == "qwen-plus"  # NOT " "

    def test_whitespace_model_name_is_stripped(self, monkeypatch):
        """MODEL_NAME=' gpt-4o ' must be normalized to 'gpt-4o' (stripped)."""
        monkeypatch.setenv("MODEL_NAME", " gpt-4o ")
        cfg = ConfigLoader().load(None)
        assert cfg.model == "gpt-4o"

    def test_empty_string_model_name_falls_back(self, monkeypatch):
        """MODEL_NAME='' (empty string) must fall back to yaml default."""
        monkeypatch.setenv("MODEL_NAME", "")
        cfg = ConfigLoader().load(None)
        assert cfg.model == "qwen-plus"

    def test_whitespace_retrieve_model_name_falls_back(self, monkeypatch):
        """RETRIEVE_MODEL_NAME=' ' (whitespace only) must fall back to yaml retrieve_model."""
        monkeypatch.setenv("RETRIEVE_MODEL_NAME", " ")
        cfg = ConfigLoader().load(None)
        assert cfg.retrieve_model == "qwen-plus"
