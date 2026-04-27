from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cluster_app.config.schema import AppConfig, PathMappingConfig

PATH_MAPPINGS_FILE = "path-mappings.json"
PATH_MAPPINGS_FILE_ENV = "CLUSTER_APP_PATH_MAPPINGS_FILE"
PATH_MAPPINGS_JSON_ENV = "CLUSTER_APP_PATH_MAPPINGS_JSON"

_ALIAS_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def mapping_file(config: AppConfig) -> Path:
    return config.paths.state_dir / PATH_MAPPINGS_FILE


def load_path_mappings(config: AppConfig) -> PathMappingConfig:
    mappings = PathMappingConfig(
        enabled=config.path_mappings.enabled,
        auto_cwd=config.path_mappings.auto_cwd,
        auto_home=config.path_mappings.auto_home,
        mappings=dict(config.path_mappings.mappings),
    )
    path = mapping_file(config)
    if path.exists():
        _merge_mapping_payload(mappings, _read_json_file(path))
    return mappings


def load_path_mappings_from_env() -> PathMappingConfig:
    raw_json = os.environ.get(PATH_MAPPINGS_JSON_ENV)
    if raw_json:
        try:
            return mapping_config_from_payload(json.loads(raw_json))
        except json.JSONDecodeError:
            return PathMappingConfig()
    path = os.environ.get(PATH_MAPPINGS_FILE_ENV)
    if path:
        return mapping_config_from_payload(_read_json_file(Path(path)))
    return PathMappingConfig()


def save_path_mappings(config: AppConfig, payload: dict[str, Any]) -> PathMappingConfig:
    current = load_path_mappings(config)
    _merge_mapping_payload(current, payload)
    config.path_mappings.enabled = current.enabled
    config.path_mappings.auto_cwd = current.auto_cwd
    config.path_mappings.auto_home = current.auto_home
    config.path_mappings.mappings = dict(current.mappings)
    path = mapping_file(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(current), indent=2), encoding="utf-8")
    return current


def mapping_config_from_payload(payload: dict[str, Any] | None) -> PathMappingConfig:
    config = PathMappingConfig()
    _merge_mapping_payload(config, payload or {})
    return config


def mapping_payload(config: AppConfig) -> dict[str, Any]:
    mappings = load_path_mappings(config)
    return {
        **asdict(mappings),
        "file": str(mapping_file(config)),
        "strategies": ["mapping", "cwd", "home", "identity"],
    }


def path_specs(paths: list[str | Path], mappings: PathMappingConfig | None = None) -> list[dict[str, Any]]:
    active = mappings or load_path_mappings_from_env()
    return [describe_path(path, active) for path in paths]


def describe_path(path: str | Path, mappings: PathMappingConfig | None = None) -> dict[str, Any]:
    active = mappings or load_path_mappings_from_env()
    requested = str(path)
    absolute = Path(path).expanduser().resolve(strict=False)
    if not active.enabled:
        return _spec("identity", requested, absolute=absolute)

    for alias, root in _sorted_mappings(active.mappings):
        root_path = Path(root).expanduser().resolve(strict=False)
        relative = _relative_to(absolute, root_path)
        if relative is not None:
            return _spec("mapping", requested, alias=alias, relative=relative, absolute=absolute)

    if active.auto_cwd:
        relative = _relative_to(absolute, Path.cwd().resolve(strict=False))
        if relative is not None:
            return _spec("cwd", requested, relative=relative, absolute=absolute)

    if active.auto_home:
        relative = _relative_to(absolute, Path.home().resolve(strict=False))
        if relative is not None:
            return _spec("home", requested, relative=relative, absolute=absolute)

    return _spec("identity", requested, absolute=absolute)


def resolve_path_spec(
    spec: dict[str, Any],
    mappings: PathMappingConfig | None = None,
) -> Path:
    active = mappings or load_path_mappings_from_env()
    strategy = str(spec.get("strategy") or "identity")
    relative = str(spec.get("relative") or "")
    if strategy == "mapping":
        alias = str(spec.get("alias") or "")
        root = active.mappings.get(alias)
        if root:
            return Path(root).expanduser().joinpath(relative).resolve(strict=False)
    if strategy == "cwd":
        return Path.cwd().joinpath(relative).resolve(strict=False)
    if strategy == "home":
        return Path.home().joinpath(relative).resolve(strict=False)
    return Path(str(spec.get("absolute") or spec.get("original") or "")).expanduser().resolve(strict=False)


def resolved_paths_exist(
    specs: list[dict[str, Any]],
    mappings: PathMappingConfig | None = None,
) -> dict[str, dict[str, Any]]:
    active = mappings or load_path_mappings_from_env()
    results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        resolved = resolve_path_spec(spec, active)
        original = str(spec.get("original") or resolved)
        results[original] = {
            "exists": resolved.exists(),
            "resolved": str(resolved),
            "strategy": str(spec.get("strategy") or "identity"),
            "alias": spec.get("alias"),
        }
    return results


def validate_mapping_payload(payload: dict[str, Any]) -> dict[str, Any]:
    mappings = payload.get("mappings")
    if mappings is None:
        return payload
    if not isinstance(mappings, dict):
        raise ValueError("mappings must be an object of alias -> local path")
    for alias, path in mappings.items():
        if not isinstance(alias, str) or not _ALIAS_PATTERN.match(alias):
            raise ValueError(f"Invalid path mapping alias: {alias!r}")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Path mapping {alias!r} must have a non-empty local path")
    return payload


def _spec(
    strategy: str,
    original: str,
    *,
    absolute: Path,
    alias: str | None = None,
    relative: Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "original": original,
        "strategy": strategy,
        "absolute": str(absolute),
    }
    if alias:
        payload["alias"] = alias
    if relative is not None:
        payload["relative"] = relative.as_posix()
    return payload


def _relative_to(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _sorted_mappings(mappings: dict[str, str]) -> list[tuple[str, str]]:
    return sorted(
        mappings.items(),
        key=lambda item: len(str(Path(item[1]).expanduser())),
        reverse=True,
    )


def _merge_mapping_payload(config: PathMappingConfig, payload: dict[str, Any]) -> None:
    validate_mapping_payload(payload)
    if "enabled" in payload:
        config.enabled = bool(payload["enabled"])
    if "auto_cwd" in payload:
        config.auto_cwd = bool(payload["auto_cwd"])
    if "auto_home" in payload:
        config.auto_home = bool(payload["auto_home"])
    if "mappings" in payload:
        config.mappings = {
            str(alias): str(path)
            for alias, path in dict(payload.get("mappings") or {}).items()
        }


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
