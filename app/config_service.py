"""Runtime config snapshot — depends on live env and PageIndexClient."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pageindex_mutil.utils import get_llm_config, ConfigLoader
from app.config_utils import (
    DEFAULT_CONFIG_YAML,
    DEFAULT_ENV,
    mask_key,
    read_yaml_model,
)


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
