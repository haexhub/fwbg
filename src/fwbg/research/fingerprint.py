"""Config fingerprints for the research register.

A fingerprint identifies what a run evaluated: the strategy metadata as the
CLI hands it to ``run_optimizer`` (and stores it as ``strategy.json``),
including CLI overrides, plus the resolved symbol list. The metadata must be
preset-resolved (``fwbg.core.presets.resolve_presets``), so an edited preset
changes the fingerprint and a preset's name does not. Descriptive, location
and bookkeeping keys are excluded, so renaming, re-tagging or moving a config
does not make it look new.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

FINGERPRINT_VERSION = 1

# Never change a result: descriptive text, scheduling, file location, dashboard bookkeeping.
EXCLUDED_KEYS = frozenset(
    {
        "name",
        "description",
        "category",
        "tags",
        "hypothesis",
        "expected_outcome",
        "baseline_run",
        "changes",
        "resources",
        "_strategy_dir",
        "_refs",
    }
)


def _normalize(value: Any) -> Any:
    # JSON files write 2, the CLI passes 2.0 — both must hash alike.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize *value* deterministically: sorted keys, no whitespace."""
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value: Any) -> str:
    """Return ``sha256:<hex>`` over the canonical JSON of *value*."""
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_config(
    strategy_metadata: Mapping[str, Any], symbols: tuple[str, ...]
) -> dict[str, Any]:
    """Return the part of a run configuration that determines its result."""
    return {
        "strategy": {
            key: value for key, value in strategy_metadata.items() if key not in EXCLUDED_KEYS
        },
        "symbols": sorted({symbol.upper() for symbol in symbols}),
    }


def fingerprint(strategy_metadata: Mapping[str, Any], symbols: tuple[str, ...]) -> str:
    """Identical result-relevant configuration, identical fingerprint."""
    return digest(canonical_config(strategy_metadata, symbols))
