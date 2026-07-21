"""Pure config helpers — no server/global imports, trivially unit-testable."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import yaml
from dotenv import set_key

DEFAULT_CONFIG_YAML = Path(__file__).resolve().parent.parent / "pageindex_mutil" / "config.yaml"
DEFAULT_ENV = Path(__file__).resolve().parent.parent / ".env"


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
