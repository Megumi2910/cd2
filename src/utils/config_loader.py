"""Utilities for loading and resolving project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "robot_config.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML configuration and attach resolved project root."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML root in {path}: expected mapping")

    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def resolve_project_path(relative_or_abs: str | Path) -> Path:
    """Resolve a path relative to project root if not already absolute."""
    p = Path(relative_or_abs)
    return p if p.is_absolute() else PROJECT_ROOT / p
