"""Preset loading and resolution for strategy configs."""
import copy
import glob
import json
import os
from collections.abc import Mapping
from typing import Any, Dict, Optional, Union


def _load_json_preset(name: str, presets_dir: str) -> dict:
    """Load a JSON preset file from the given directory.

    Tries ``{name}.json`` first; if not found, falls back to the highest-version
    ``{name}_v*.json`` file (versioned naming scheme).

    The preset name must be a simple identifier (no path separators) and the
    resolved file path must stay inside ``presets_dir`` even after following
    symlinks — this protects against ``../etc/passwd``-style traversal and
    symlink-race attacks.
    """
    # Reject obviously hostile names early; presets are simple identifiers.
    if not name or "/" in name or "\\" in name or name in (".", "..") or "\x00" in name:
        raise ValueError(f"Invalid preset name: {name!r}")

    real_dir = os.path.realpath(presets_dir)
    real_dir_prefix = real_dir.rstrip(os.sep) + os.sep

    def _safe_resolve(candidate: str) -> Optional[str]:
        resolved = os.path.realpath(candidate)
        if resolved == real_dir or resolved.startswith(real_dir_prefix):
            return resolved
        return None

    path = os.path.join(presets_dir, f"{name}.json")
    resolved = _safe_resolve(path)
    if resolved is None:
        raise ValueError(f"Preset name '{name}' resolves outside allowed directory")

    if not os.path.isfile(resolved):
        # Fall back to versioned filename. Filter the glob output through the
        # realpath check inside the loop so a symlink injected between glob and
        # realpath cannot escape the presets directory.
        matches = sorted(glob.glob(os.path.join(presets_dir, f"{name}_v*.json")))
        valid: list[str] = []
        for m in matches:
            safe = _safe_resolve(m)
            if safe is not None and os.path.isfile(safe):
                valid.append(safe)
        if not valid:
            raise FileNotFoundError(f"Preset '{name}' not found at {path}")
        resolved = valid[-1]  # highest lexicographic = highest version

    with open(resolved, "r") as f:
        data: dict = json.load(f)
    data.pop("_meta", None)  # strip embedded metadata before returning content
    return data


def _resolve_section(
    value: "Optional[Union[str, Dict[str, Any]]]",
    section_dir: str,
    strategy_dir: Optional[str],
) -> "Optional[Dict[str, Any]]":
    """Resolve a config section: string loads preset file, dict/None passes through.

    Dicts are deep-copied so that downstream mutation of the returned object
    cannot leak back into the caller's strategy data structure.
    """
    if value is None:
        return value
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if isinstance(value, str):
        base = os.path.dirname(strategy_dir) if strategy_dir else os.getcwd()
        return _load_json_preset(value, os.path.join(base, section_dir))
    return value


PRESET_SECTIONS: dict[str, str] = {
    "pipeline": "pipelines",
    "model": "models",
    "validation": "validations",
    "filters": "filters",
    "resources": "resources",
    "risk_params": "risk_params",
    "regime_filter": "regime_filters",
}


def resolve_presets(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of strategy *data* with every preset reference replaced by its content.

    This is the resolution ``StrategyConfig.from_dict`` applies, so a run's
    result depends on preset content, never on a preset's name.
    """
    strategy_dir = data.get("_strategy_dir")
    resolved = dict(data)
    for key, section_dir in PRESET_SECTIONS.items():
        if key in resolved:
            resolved[key] = _resolve_section(resolved[key], section_dir, strategy_dir)
    return resolved
