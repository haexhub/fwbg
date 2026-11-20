"""Which columns an indicator declares as signal columns, checked against its behaviour.

``get_signal_columns`` lists the unshifted decision columns: row i uses bar i's own
close. The ML feature pool leaves them out, so a mislabelled column either lets a
leak into the models (unshifted but undeclared) or silently removes a valid feature
(shifted but declared). ``get_calendar_columns`` is the subset of signal columns
that is known before the bar opens (timestamp and a calendar fixed in advance).

The probe perturbs bar i of synthetic data, up and down, for every row of a day and
looks at row i of the output: a column that moves depends on its own bar.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from fwbg.pipeline import get_registry

get_registry().auto_discover()

from fwbg_sdk import INDICATOR_REGISTRY  # noqa: E402

# Need a fitted model, a neural network or heavy topology: not part of the probe.
SKIP = {
    "adversarial_validation",
    "autoencoder_features",
    "topological_features",
    "wavelets",
    "computed_signal",  # rules are defined per instance, not per class
}

# Settings under which the plugin's own-bar columns actually fire on hourly data.
PARAMS = {
    "opening_range": {"sessions": [8], "range_bars": 2, "pre_range_bars": 1},
    "noise_boundary": {
        "session_tz": "UTC",
        "session_open": "09:00",
        "session_close": "17:00",
        "first_check": "10:00",
        "last_check": "16:00",
        "check_every_minutes": 60,
    },
}

# Declared columns the probe cannot trigger: they fire only after a sequence (a
# breakout, then a pullback to the retest level), never from one perturbed bar.
# opening_range leaves them out of shift_features (ORB_SIGNAL_SUFFIXES).
UNPROBED_SUFFIXES = {"opening_range": ("_retest_bull", "_retest_bear")}

# Calendar plugins whose inputs are external (event file, loaded macro columns):
# their calendar columns are checked in their own test modules.
EXTERNAL_INPUTS = {"scheduled_events", "rebalancing_pressure"}

N_BARS = 1500
FIRST_ROW = 1008  # midnight: the window covers every hour of one day
ROWS = range(FIRST_ROW, FIRST_ROW + 24)

NAMES = sorted(n for n in INDICATOR_REGISTRY if n not in SKIP)


def make_ohlc() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 0.3, N_BARS))
    open_ = close + rng.normal(0, 0.1, N_BARS)
    high = np.maximum(close, open_) + np.abs(rng.normal(0, 0.2, N_BARS))
    low = np.minimum(close, open_) - np.abs(rng.normal(0, 0.2, N_BARS))
    index = pd.date_range("2021-01-04", periods=N_BARS, freq="h")
    return pd.DataFrame(
        {
            "O": open_,
            "H": high,
            "L": low,
            "C": close,
            "V": rng.integers(100, 1000, N_BARS).astype(float),
        },
        index=index,
    )


def column_method(plugin, name: str, params: dict) -> list:
    method = getattr(plugin, name, None)
    if method is None:
        return []
    try:
        return list(method(params))
    except TypeError:
        return list(method())


@lru_cache(maxsize=None)
def probe(name: str) -> dict:
    cls = INDICATOR_REGISTRY[name]
    plugin = cls()
    params = {**cls.get_default_params(), **PARAMS.get(name, {})}
    df = make_ohlc()
    base = plugin.compute(df.copy(), **params)
    new = [c for c in base.columns if c not in df.columns and not c.startswith("_")]
    own_bar = set()
    for row in ROWS:
        for move in (60.0, -60.0):
            moved = df.copy()
            for col in ("O", "H", "L", "C"):
                moved.iloc[row, moved.columns.get_loc(col)] += move
            out = plugin.compute(moved, **params)
            for col in new:
                a, b = base[col].iloc[row], out[col].iloc[row]
                if not (pd.isna(a) and pd.isna(b)) and a != b:
                    own_bar.add(col)
    return {
        "produced": set(base.columns),
        "own_bar": own_bar,
        "signals": column_method(plugin, "get_signal_columns", params),
        "calendar": column_method(plugin, "get_calendar_columns", params),
    }


@pytest.mark.parametrize("name", NAMES)
def test_declared_columns_are_produced(name):
    found = probe(name)
    missing = [c for c in found["signals"] if c not in found["produced"]]
    assert not missing, f"{name} declares signal columns it does not produce: {missing}"


@pytest.mark.parametrize("name", NAMES)
def test_calendar_columns_are_signal_columns(name):
    found = probe(name)
    assert set(found["calendar"]) <= set(found["signals"])


@pytest.mark.parametrize("name", NAMES)
def test_columns_reading_their_own_bar_are_declared(name):
    """An unshifted column that is not declared would reach the model pool."""
    found = probe(name)
    undeclared = sorted(found["own_bar"] - set(found["signals"]))
    assert not undeclared, (
        f"{name}: columns use bar i's own prices but are not signal columns: {undeclared}"
    )


@pytest.mark.parametrize("name", NAMES)
def test_declared_signal_columns_are_unshifted(name):
    """A shifted column declared as signal would be dropped from the pool for nothing."""
    found = probe(name)
    calendar = set(found["calendar"])
    unprobed = UNPROBED_SUFFIXES.get(name, ())
    shifted = sorted(
        c
        for c in found["signals"]
        if c not in calendar and c not in found["own_bar"] and not c.endswith(unprobed)
    )
    assert not shifted, (
        f"{name}: declared signal columns that do not read their own bar: {shifted}"
    )


@pytest.mark.parametrize("name", [n for n in NAMES if n not in EXTERNAL_INPUTS])
def test_calendar_columns_ignore_prices(name):
    found = probe(name)
    if not found["calendar"]:
        pytest.skip("no calendar columns")
    cls = INDICATOR_REGISTRY[name]
    plugin = cls()
    params = {**cls.get_default_params(), **PARAMS.get(name, {})}
    df = make_ohlc()
    base = plugin.compute(df.copy(), **params)
    scaled = df.copy()
    scaled[["O", "H", "L", "C"]] = scaled[["O", "H", "L", "C"]] * 1.7
    scaled["V"] = scaled["V"] * 3
    out = plugin.compute(scaled, **params)
    for col in found["calendar"]:
        pd.testing.assert_series_equal(base[col], out[col], check_names=False)


# Classification decisions that are not visible from the probe alone.
@pytest.mark.parametrize(
    "name",
    [
        "adx", "aroon", "balance_zones", "cci", "cusum_events", "displacement",
        "efficiency_ratio", "ema", "fair_value_gap", "fractal_dimension",
        "liquidity_levels", "liquidity_sweep", "macd", "market_structure",
        "momentum", "previous_day_levels", "price_action", "pullback_momentum",
        "sma", "supertrend", "supertrend_mtf", "supply_demand_flip", "volatility",
        "vwap", "weekly_opening_range",
    ],
)  # fmt: skip
def test_shifted_indicators_declare_no_signal_columns(name):
    """These indicators go through shift_features: nothing in them is an own-bar decision."""
    assert probe(name)["signals"] == []


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ibs", {"ibs", "ibs_down", "ibs_above_prev_high"}),
        ("zscore", {"zscore_20", "zscore_20_above_mean", "zscore_20_below_mean"}),
        (
            "noise_boundary",
            {"nb_upper", "nb_lower", "nb_vwap", "nb_long", "nb_short", "nb_exit_long", "nb_exit_short"},
        ),
    ],
)  # fmt: skip
def test_close_derived_indicators_declare_every_column(name, expected):
    assert set(probe(name)["signals"]) == expected
    assert probe(name)["calendar"] == []


@pytest.mark.parametrize(
    "name", ["gotobi", "turn_of_month", "time_season", "calendar_events", "rebalancing_pressure", "scheduled_events"]
)  # fmt: skip
def test_calendar_indicators_keep_their_signals_in_the_pool(name):
    found = probe(name)
    assert found["signals"] and set(found["signals"]) == set(found["calendar"])


def test_opening_range_declares_breakouts_and_retests_only():
    signals = probe("opening_range")["signals"]
    assert signals
    assert all(
        c.endswith(("_breakout_up", "_breakout_down", "_retest_bull", "_retest_bear"))
        for c in signals
    )
    # the structural levels are NaN while the range forms: known before the bar opens
    assert not any(
        c.endswith(("_or_high", "_or_low", "_sl_dist", "_range")) for c in signals
    )
