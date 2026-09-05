"""
Configuration loader for the AI Wildlife Monitoring System.

Loads configs/config.yaml into a nested, dot-accessible object and allows
environment variables to override any leaf value using a double-underscore
path convention, e.g.:

    DATABASE__PASSWORD=secret
    DETECTION__DEVICE=cpu
    NOTIFICATIONS__TELEGRAM__ENABLED=true

Usage:
    from utils.config_loader import load_config
    cfg = load_config()
    print(cfg.detection.model_path)
    print(cfg["detection"]["model_path"])   # also works
"""

from __future__ import annotations

import os
import copy
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ConfigNode(dict):
    """A dict subclass that also supports attribute-style access, recursively."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            value = ConfigNode(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted_path: str, default: Any = None) -> Any:
        """Get a nested value using a dotted path, e.g. 'database.host'."""
        node: Any = self
        for part in dotted_path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return ConfigNode({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _coerce(value: str) -> Any:
    """Best-effort coercion of an environment variable string to bool/int/float/str."""
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _apply_env_overrides(config: Dict[str, Any], prefix: str = "") -> None:
    """
    Recursively walks the config dict and applies any matching environment
    variable overrides in-place, using DOUBLE_UNDERSCORE__PATH__STYLE keys.
    """
    for key, value in list(config.items()):
        env_key = f"{prefix}{key}".upper()
        if isinstance(value, dict):
            _apply_env_overrides(value, prefix=f"{env_key}__")
        else:
            if env_key in os.environ:
                config[key] = _coerce(os.environ[env_key])


def load_config(path: str | Path | None = None) -> ConfigNode:
    """
    Load and return the system configuration as a ConfigNode.

    Args:
        path: optional override path to a YAML config file. Defaults to
              configs/config.yaml relative to the project root.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw = copy.deepcopy(raw)
    _apply_env_overrides(raw)
    return _wrap(raw)


def load_species_map(path: str | Path | None = None) -> ConfigNode:
    """Load configs/species_map.yaml (or an override path)."""
    species_path = Path(path) if path else (PROJECT_ROOT / "configs" / "species_map.yaml")
    if not species_path.exists():
        raise FileNotFoundError(f"Species map file not found: {species_path}")
    with open(species_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _wrap(raw)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config that is relative to the project root."""
    p = Path(relative_path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


if __name__ == "__main__":
    cfg = load_config()
    print(f"Loaded config for project: {cfg.project.name} (v{cfg.project.version})")
    print(f"Detection model: {cfg.detection.model_type} @ {cfg.detection.model_path}")
    print(f"Database: {cfg.database.host}:{cfg.database.port}/{cfg.database.name}")
