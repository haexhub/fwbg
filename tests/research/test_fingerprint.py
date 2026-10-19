import json

import pytest

from fwbg.core.presets import resolve_presets
from fwbg.research.fingerprint import (
    EXCLUDED_KEYS,
    FINGERPRINT_VERSION,
    canonical_json,
    fingerprint,
)

BASE = {
    "name": "CUSUM",
    "description": "first try",
    "tags": ["event"],
    "timeframe": "DAY_1",
    "pipeline": {"indicators": [{"name": "cusum_events", "params": {"threshold": 1.5}}]},
    "cost_multiplier": 2.0,
}
SYMBOLS = ("GOLD", "EURUSD")


def test_excluded_keys_do_not_change_fingerprint():
    changed = {
        **BASE,
        "name": "other",
        "description": "second try",
        "category": "momentum",
        "tags": ["x"],
        "hypothesis": "h",
        "expected_outcome": "o",
        "baseline_run": "run-1",
        "changes": ["raised threshold"],
        "resources": {"max_concurrent_assets": 4},
        "_strategy_dir": "/elsewhere/strategies/configs",
        "_refs": {"pipeline": "orb_simple_v1"},
    }
    assert EXCLUDED_KEYS <= changed.keys()
    assert fingerprint(changed, SYMBOLS) == fingerprint(BASE, SYMBOLS)


def test_parameter_change_changes_fingerprint():
    changed = {
        **BASE,
        "pipeline": {"indicators": [{"name": "cusum_events", "params": {"threshold": 2.0}}]},
    }
    assert fingerprint(changed, SYMBOLS) != fingerprint(BASE, SYMBOLS)


@pytest.mark.parametrize(
    "override",
    [
        {"end_date": "2025-04-30"},
        {"start_date": "2020-01-01"},
        {"cost_multiplier": 3.0},
        {"timeframe": "HOUR_1"},
        {"_cli_data_path": "/data/other"},
    ],
)
def test_run_overrides_change_fingerprint(override):
    assert fingerprint({**BASE, **override}, SYMBOLS) != fingerprint(BASE, SYMBOLS)


def test_key_order_is_irrelevant():
    reordered = dict(reversed(list(BASE.items())))
    assert fingerprint(reordered, SYMBOLS) == fingerprint(BASE, SYMBOLS)


def test_symbol_order_and_case_are_irrelevant():
    assert fingerprint(BASE, ("eurusd", "GOLD")) == fingerprint(BASE, SYMBOLS)


def test_duplicate_symbols_collapse():
    assert fingerprint(BASE, ("GOLD", "gold", "EURUSD")) == fingerprint(BASE, ("EURUSD", "GOLD"))


def test_symbol_set_changes_fingerprint():
    assert fingerprint(BASE, ("GOLD",)) != fingerprint(BASE, SYMBOLS)


def test_integral_float_and_int_hash_alike():
    assert fingerprint({**BASE, "cost_multiplier": 2}, SYMBOLS) == fingerprint(BASE, SYMBOLS)


def test_integral_float_nested_in_list_hashes_alike():
    as_int = {
        **BASE,
        "pipeline": {"indicators": [{"name": "cusum_events", "params": {"threshold": 2}}]},
    }
    as_float = {
        **BASE,
        "pipeline": {"indicators": [{"name": "cusum_events", "params": {"threshold": 2.0}}]},
    }
    assert fingerprint(as_int, SYMBOLS) == fingerprint(as_float, SYMBOLS)


def test_bool_is_not_confused_with_int():
    assert canonical_json({"flag": True}) != canonical_json({"flag": 1})


def test_nan_is_rejected():
    with pytest.raises(ValueError):
        fingerprint({**BASE, "cost_multiplier": float("nan")}, SYMBOLS)


def test_fingerprint_format():
    value = fingerprint(BASE, SYMBOLS)
    assert value.startswith("sha256:")
    assert len(value) == len("sha256:") + 64


def test_golden_fingerprint():
    # A serialisation change breaks this on purpose: bump FINGERPRINT_VERSION, then update.
    assert FINGERPRINT_VERSION == 1
    metadata = {
        "name": "golden",
        "timeframe": "DAY_1",
        "cost_multiplier": 2.0,
        "pipeline": {"indicators": [{"name": "cusum_events", "params": {"threshold": 1.5}}]},
    }
    assert fingerprint(metadata, ("gold", "EURUSD")) == (
        "sha256:8b36d1b1d1d79b8d14688911678553592dd9bdc2382960d9429f1dd37221fe2c"
    )


def _preset_layout(base, threshold):
    """Write <base>/strategies/pipelines/cusum.json; return metadata referencing it by name."""
    pipelines = base / "strategies" / "pipelines"
    pipelines.mkdir(parents=True, exist_ok=True)
    preset = {"indicators": [{"name": "cusum_events", "params": {"threshold": threshold}}]}
    (pipelines / "cusum.json").write_text(json.dumps(preset))
    return {
        "timeframe": "DAY_1",
        "pipeline": "cusum",
        "_strategy_dir": str(base / "strategies" / "configs"),
    }


def test_preset_content_changes_fingerprint(tmp_path):
    metadata = _preset_layout(tmp_path, 1.5)
    before = fingerprint(resolve_presets(metadata), SYMBOLS)

    _preset_layout(tmp_path, 2.0)

    assert fingerprint(resolve_presets(metadata), SYMBOLS) != before


def test_same_layout_in_different_directories_has_same_fingerprint(tmp_path):
    first = _preset_layout(tmp_path / "a", 1.5)
    second = _preset_layout(tmp_path / "b", 1.5)
    assert first["_strategy_dir"] != second["_strategy_dir"]

    assert fingerprint(resolve_presets(first), SYMBOLS) == fingerprint(
        resolve_presets(second), SYMBOLS
    )
