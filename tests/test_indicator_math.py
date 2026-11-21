"""Shared Wilder ATR / RSI helpers: NaN warm-up instead of ta's 0.0 seed."""

import numpy as np
import pandas as pd
import ta

from fwbg.utils.indicator_math import wilder_atr, wilder_rsi


def _bars(n=40, seed=3):
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + np.cumsum(rng.normal(size=n)))
    high = close + rng.random(n)
    low = close - rng.random(n)
    return high, low, close


def test_atr_is_nan_during_warm_up_where_ta_reports_zero():
    high, low, close = _bars()
    theirs = ta.volatility.average_true_range(high, low, close, window=14)
    assert (theirs.iloc[:13] == 0.0).all()  # the trap this helper removes

    ours = wilder_atr(high, low, close, window=14)
    assert ours.iloc[:13].isna().all()
    assert ours.iloc[13:].notna().all()
    np.testing.assert_allclose(ours.iloc[13:], theirs.iloc[13:])


def test_atr_concrete_first_value_is_the_mean_true_range():
    # constant range 2, no gaps -> TR = 2 everywhere
    close = pd.Series(np.full(6, 10.0))
    atr = wilder_atr(close + 1, close - 1, close, window=3)
    assert atr.iloc[:2].isna().all()
    np.testing.assert_allclose(atr.iloc[2:], 2.0)


def test_atr_with_fewer_bars_than_window_is_all_nan():
    high, low, close = _bars(n=5)
    assert wilder_atr(high, low, close, window=14).isna().all()


def test_rsi_is_nan_until_a_full_window_of_changes_exists():
    close = pd.Series([10, 11, 10, 12, 13, 12, 14, 15, 14, 16], dtype=float)
    rsi = wilder_rsi(close, window=3)
    assert rsi.iloc[:3].isna().all()
    assert rsi.iloc[3:].notna().all()
    # first value: mean gain of changes 1..3 is (1 + 0 + 2) / 3, mean loss 1 / 3
    np.testing.assert_allclose(rsi.iloc[3], 100 - 100 / (1 + 3.0))


def test_rsi_edge_cases():
    up = pd.Series(np.arange(10.0))
    down = pd.Series(np.arange(10.0)[::-1])
    flat = pd.Series(np.full(10, 5.0))
    assert wilder_rsi(up, 3).iloc[3:].eq(100.0).all()
    assert wilder_rsi(down, 3).iloc[3:].eq(0.0).all()
    assert wilder_rsi(flat, 3).iloc[3:].eq(50.0).all()


def test_momentum_indicator_uses_the_shared_rsi():
    from fwbg.plugins import import_plugin_module

    mod = import_plugin_module("fwbg-core", "indicators", "momentum")
    high, low, close = _bars(n=60)
    df = pd.DataFrame({"O": close, "H": high, "L": low, "C": close, "V": 0.0})
    out = mod.MomentumIndicators().compute(df, rsi_periods=[14])
    expected = wilder_rsi(close, 14).shift(1)
    pd.testing.assert_series_equal(out["mom_rsi_14"], expected, check_names=False)
