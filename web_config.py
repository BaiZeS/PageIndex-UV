"""Disk-persistence helpers for the web console config endpoints.

Pure functions taking explicit paths — no server/global imports — so they are
trivially unit-testable with tmp_path fixtures and never touch the real
.env / config.yaml unless called with those exact paths at runtime.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import yaml
from dotenv import set_key

DEFAULT_CONFIG_YAML = Path(__file__).resolve().parent / "pageindex_mutil" / "config.yaml"
DEFAULT_ENV = Path(__file__).resolve().parent / ".env"


def mask_key(key: Optional[str]) -> str:
    """Return a masked copy: first 8 + '****' + last 4. Short keys fully hidden."""
    if not key:
        return ""
    if len(key) <= 12:
        return "*" * len(key)
    return key[:8] + "****" + key[-4:]


def backup_file(path: Path) -> Path:
    """Copy path -> path.with_suffix(<suffix>+'.bak'). Overwrites existing bak."""
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    return bak


def read_yaml_model(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_config_yaml_model(path: Path, *, model: Optional[str] = None,
                          retrieve_model: Optional[str] = None) -> None:
    """Targeted line replacement of `model:` / `retrieve_model:` lines.

    Preserves every comment and all other lines (yaml.dump would nuke comments).
    Only rewrites lines whose stripped text starts with the exact key + ':'.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        nl = "\n" if line.endswith("\n") else ""
        if model is not None and stripped.startswith("model:") and not stripped.startswith("model_"):
            lines[i] = f'{indent}model: "{model}"{nl}'
        elif retrieve_model is not None and stripped.startswith("retrieve_model:"):
            lines[i] = f'{indent}retrieve_model: "{retrieve_model}"{nl}'
    path.write_text("".join(lines), encoding="utf-8")


def set_env_fields(env_path: Path, fields: dict) -> None:
    """set_key each field: uncomments/updates/appends, preserves other lines."""
    for k, v in fields.items():
        set_key(str(env_path), k, str(v))


import os
from typing import Any

from dotenv import dotenv_values

from pageindex_mutil.utils import get_llm_config, ConfigLoader


def derive_provider() -> str:
    """openai | dashscope | none — mirrors _resolve_llm_config key priority."""
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("DASHSCOPE_API_KEY"):
        return "dashscope"
    return "none"


def read_config_snapshot(client: Any, config_yaml_path: Path = DEFAULT_CONFIG_YAML,
                          env_path: Path = DEFAULT_ENV) -> dict:
    key, url = get_llm_config()
    yaml_data = read_yaml_model(config_yaml_path) if Path(config_yaml_path).exists() else {}
    opt = ConfigLoader().load(None)
    doc_count = 0
    db = getattr(client, "db", None)
    if db is not None:
        try:
            doc_count = len(db.get_all_documents())
        except Exception:
            pass
    return {
        "model": opt.model,
        "retrieve_model": getattr(opt, "retrieve_model", None) or opt.model,
        "model_yaml": yaml_data.get("model"),
        "retrieve_model_yaml": yaml_data.get("retrieve_model"),
        "provider": derive_provider(),
        "base_url": url,
        "api_key_masked": mask_key(key),
        "has_api_key": bool(key),
        "document_count": doc_count,
        "env_overrides": {
            "MODEL_NAME": os.getenv("MODEL_NAME"),
            "RETRIEVE_MODEL_NAME": os.getenv("RETRIEVE_MODEL_NAME"),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
        },
    }
