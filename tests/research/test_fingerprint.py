import pytest

from fwbg.research.fingerprint import canonical_json, fingerprint

BASE = {
    "name": "CUSUM",
    "description": "first try",
    "tags": ["event"],
    "timeframe": "DAY_1",
    "pipeline": {"indicators": [{"name": "cusum_events", "params": {"threshold": 1.5}}]},
    "cost_multiplier": 2.0,
}
SYMBOLS = ["GOLD", "EURUSD"]


def test_descriptive_keys_do_not_change_fingerprint():
    renamed = {
        **BASE,
        "name": "other",
        "description": "second try",
        "tags": ["x"],
        "hypothesis": "h",
        "resources": {"max_concurrent_assets": 4},
    }
    assert fingerprint(renamed, SYMBOLS) == fingerprint(BASE, SYMBOLS)


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
    assert fingerprint(BASE, ["eurusd", "GOLD"]) == fingerprint(BASE, SYMBOLS)


def test_symbol_set_changes_fingerprint():
    assert fingerprint(BASE, ["GOLD"]) != fingerprint(BASE, SYMBOLS)


def test_integral_float_and_int_hash_alike():
    assert fingerprint({**BASE, "cost_multiplier": 2}, SYMBOLS) == fingerprint(BASE, SYMBOLS)


def test_bool_is_not_confused_with_int():
    assert canonical_json({"flag": True}) != canonical_json({"flag": 1})


def test_fingerprint_format():
    value = fingerprint(BASE, SYMBOLS)
    assert value.startswith("sha256:")
    assert len(value) == len("sha256:") + 64
