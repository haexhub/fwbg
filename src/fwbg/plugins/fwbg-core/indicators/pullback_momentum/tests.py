"""Tests for the pullback_momentum indicator plugin."""

import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_mod = import_plugin_module("fwbg-core", "indicators", "pullback_momentum")
PullbackMomentumIndicator = _mod.PullbackMomentumIndicator

# short EMA / swing windows keep the hand-built scenario small
PARAMS = {"ema_period": 30, "swing_lookback": 10, "atr_period": 5}
COLUMNS = PullbackMomentumIndicator().get_feature_columns()
PRICE_STEPS = 0.2  # half range of every bar around its close


def _frame(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {"O": close, "H": close + PRICE_STEPS, "L": close - PRICE_STEPS, "C": close},
        index=pd.date_range("2024-01-01", periods=len(close), freq="h"),
    )


def _mirror(df, level=320.0):
    """The same market upside down: every long setup becomes a short setup."""
    return pd.DataFrame(
        {
            "O": level - df["O"],
            "H": level - df["L"],
            "L": level - df["H"],
            "C": level - df["C"],
        },
        index=df.index,
    )


# +1 per bar for 61 bars (breakout of the 10-bar swing high on every bar), a
# pullback that is deep enough (>= 38.2 % of the impulse range) for the entry on
# bar 66, and one trailing bar so the shifted signal is visible.
UPTREND = list(100 + np.arange(61.0))
PULLBACK = [158.0, 156.0, 154.5, 154.0, 153.8, 155.4, 156.0]
CLOSE = np.array(UPTREND + PULLBACK)
ENTRY_BAR = 66


def _reference_atr(df, window):
    high, low, close = (df[c].to_numpy() for c in ("H", "L", "C"))
    tr = high - low
    tr[1:] = np.maximum.reduce(
        [high[1:] - low[1:], abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])]
    )
    atr = np.full(len(df), np.nan)
    atr[window - 1] = tr[:window].mean()
    for i in range(window, len(df)):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return atr


@pytest.fixture
def indicator():
    return PullbackMomentumIndicator()


@pytest.fixture
def long_setup():
    return _frame(CLOSE)


def test_long_entry_fires_on_the_break_of_the_local_lower_high(indicator, long_setup):
    out = indicator.compute(long_setup, **PARAMS)
    # rows hold the signal of the previous bar
    entries = out.index[out["tpm_entry_long"] == 1]
    assert list(entries) == [long_setup.index[ENTRY_BAR + 1]]
    assert out["tpm_entry_short"].dropna().sum() == 0


def test_stop_distance_is_pullback_depth_plus_atr_buffer(indicator, long_setup):
    out = indicator.compute(long_setup, **PARAMS)
    atr = _reference_atr(long_setup, 5)[ENTRY_BAR]
    pullback_low = long_setup["L"].iloc[60 : ENTRY_BAR + 1].min()
    expected = (long_setup["C"].iloc[ENTRY_BAR] - pullback_low) + 0.25 * atr
    assert out["tpm_sl_dist_long"].iloc[ENTRY_BAR + 1] == pytest.approx(expected)
    assert out["tpm_sl_dist_long"].drop(out.index[ENTRY_BAR + 1]).isna().all()
    assert out["tpm_sl_dist_short"].isna().all()


def test_pullback_depth_is_measured_against_the_broken_swing_high(
    indicator, long_setup
):
    """Impulse range: broken swing high (highest high of the 10 bars before the
    break) down to the swing low of the same window."""
    out = indicator.compute(long_setup, **PARAMS)
    swing_high = long_setup["H"].iloc[50:60].max()
    swing_low = long_setup["L"].iloc[50:60].min()
    for bar in range(61, ENTRY_BAR):
        expected = (swing_high - long_setup["C"].iloc[bar]) / (swing_high - swing_low)
        assert out["tpm_pullback_pct"].iloc[bar + 1] == pytest.approx(expected)
        assert out["tpm_in_pullback_long"].iloc[bar + 1] == 1.0
    # the entry bar closes the setup
    assert out["tpm_in_pullback_long"].iloc[ENTRY_BAR + 1] == 0.0
    assert out["tpm_pullback_pct"].iloc[ENTRY_BAR + 1] == 0.0


def test_no_entry_when_the_pullback_is_too_shallow(indicator, long_setup):
    out = indicator.compute(long_setup, min_pullback_pct=0.8, **PARAMS)
    assert out["tpm_entry_long"].dropna().sum() == 0


def test_setup_fails_when_close_falls_below_the_impulse_low(indicator):
    close = np.array(UPTREND + [158.0, 140.0, 139.0, 141.0, 150.0, 155.0])
    out = indicator.compute(_frame(close), **PARAMS)
    assert out["tpm_in_pullback_long"].iloc[62] == 1.0  # bar 61 inside the pullback
    assert (out["tpm_in_pullback_long"].iloc[63:] == 0.0).all()  # bar 62 broke the low
    assert out["tpm_entry_long"].dropna().sum() == 0


def test_short_entry_is_the_mirror_image_of_the_long_entry(indicator, long_setup):
    long_out = indicator.compute(long_setup, **PARAMS)
    short_out = indicator.compute(_mirror(long_setup), **PARAMS)
    pd.testing.assert_series_equal(
        long_out["tpm_entry_long"], short_out["tpm_entry_short"], check_names=False
    )
    pd.testing.assert_series_equal(
        long_out["tpm_sl_dist_long"], short_out["tpm_sl_dist_short"], check_names=False
    )
    pd.testing.assert_series_equal(
        long_out["tpm_pullback_pct"], short_out["tpm_pullback_pct"], check_names=False
    )
    pd.testing.assert_series_equal(
        long_out["tpm_trend_ok_long"],
        short_out["tpm_trend_ok_short"],
        check_names=False,
    )
    assert short_out["tpm_entry_long"].dropna().sum() == 0


def test_trend_flags_follow_the_ema(indicator, long_setup):
    out = indicator.compute(long_setup, **PARAMS)
    assert (out["tpm_trend_ok_long"].iloc[30:] == 1.0).all()
    assert (out["tpm_trend_ok_short"].iloc[30:] == 0.0).all()


def test_warm_up_before_the_ema_exists_is_nan(indicator, long_setup):
    out = indicator.compute(long_setup, **PARAMS)
    # EMA(30) exists from bar 29; features are shifted by one bar
    assert out[COLUMNS].iloc[:30].isna().all().all()
    for column in COLUMNS:
        if column.startswith("tpm_sl_dist"):
            continue
        assert out[column].iloc[30:].notna().all(), column


def test_entry_needs_a_valid_atr(indicator, long_setup):
    # ATR(70) never completes on 68 bars: the buffer is unknown, so no entry
    out = indicator.compute(long_setup, ema_period=30, swing_lookback=10, atr_period=70)
    assert out["tpm_entry_long"].dropna().sum() == 0
    assert out["tpm_sl_dist_long"].isna().all()
    assert (
        out["tpm_in_pullback_long"].iloc[ENTRY_BAR] == 1.0
    )  # the setup is still tracked


def test_features_use_only_past_bars(indicator):
    """Truncating the data after bar n-1 leaves rows 0..n-1 unchanged, for every n."""
    # a long setup followed by its mirror image: one long and one short entry
    second = 256.0 - CLOSE
    df = _frame(np.concatenate([CLOSE, second]))
    params = {"ema_period": 20, "swing_lookback": 8, "atr_period": 5}
    full = indicator.compute(df, **params)
    assert full["tpm_entry_long"].sum() >= 1
    assert full["tpm_entry_short"].sum() >= 1
    for n in range(10, len(df)):
        part = indicator.compute(df.iloc[:n], **params)
        pd.testing.assert_frame_equal(part[COLUMNS], full[COLUMNS].iloc[:n])


def test_declared_columns_are_produced(indicator, long_setup):
    out = indicator.compute(long_setup, **PARAMS)
    assert set(indicator.get_feature_columns()) == {
        c for c in out.columns if c.startswith("tpm_")
    }
    assert set(indicator.get_signal_columns()) <= set(out.columns)
