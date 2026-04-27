from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cluster_app.config.schema import AppConfig, config_from_dict, config_to_dict

DEFAULT_CONFIG_PATH = Path("config.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        try:
            return _load_json(path)
        except json.JSONDecodeError:
            raise RuntimeError(
                "PyYAML is required to read YAML config files. Install the project venv first."
            ) from exc
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    config_path = Path(path or DEFAULT_CONFIG_PATH)
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = _load_json(config_path) if config_path.suffix == ".json" else _load_yaml(config_path)
    if overrides:
        raw = _merge_dicts(raw, overrides)
    cfg = config_from_dict(raw)
    cfg.ensure_directories()
    return cfg


def write_default_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    target = Path(path)
    if target.exists():
        return target
    cfg = AppConfig()
    if target.suffix == ".json":
        target.write_text(json.dumps(config_to_dict(cfg), indent=2), encoding="utf-8")
        return target
    try:
        import yaml
    except ModuleNotFoundError:
        target.write_text(json.dumps(config_to_dict(cfg), indent=2), encoding="utf-8")
    else:
        target.write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")
    return target


def _merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
