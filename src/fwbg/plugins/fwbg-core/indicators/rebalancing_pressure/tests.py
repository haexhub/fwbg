"""Tests for the rebalancing_pressure indicator plugin."""

import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

mod = import_plugin_module("fwbg-core", "indicators", "rebalancing_pressure")
Indicator = mod.RebalancingPressureIndicator
# NYSE trading days, March 2024: the 29th (Good Friday) is closed, the 28th is the last
MARCH = pd.bdate_range("2024-02-29", "2024-04-02").drop(pd.Timestamp("2024-03-29"))


def frame(spy_path, bond_path) -> pd.DataFrame:
    """Bars on MARCH; each input column holds the previous trading day's close."""
    spy = pd.Series(spy_path, index=MARCH).shift(1)
    bond = pd.Series(bond_path, index=MARCH).shift(1)
    return pd.DataFrame(
        {"C": 1.0, "macro_spy": spy, "macro_tlt": bond, "macro_ief": bond}, index=MARCH
    )


def run(df, **params):
    return Indicator().compute(df, **params)


def test_drift_is_month_to_date_from_the_previous_months_last_close():
    n = len(MARCH)
    spy = np.full(n, 100.0)
    spy[1:] = 103.0  # +3 % on 1 March, flat after
    out = run(frame(spy, np.full(n, 50.0)))
    march = out.loc["2024-03-01":"2024-03-28", "rebal_drift"]
    # 1 March sees Feb 29's close (the base): zero drift; from 4 March on: +3 %
    assert march.iloc[0] == pytest.approx(0.0)
    assert march.loc["2024-03-04":].to_numpy() == pytest.approx(0.03)


def test_entry_three_trading_days_before_the_holiday_shortened_month_end():
    n = len(MARCH)
    spy = np.full(n, 100.0)
    spy[1:] = 103.0
    out = run(frame(spy, np.full(n, 50.0)))
    assert list(out.index[out["rebal_entry_short"] == 1].strftime("%m-%d")) == ["03-25"]
    assert out["rebal_entry_long"].loc["2024-03-01":].sum() == 0
    assert list(out.index[out["rebal_exit"] == 1].strftime("%m-%d")) == [
        "03-28"
    ]  # 02-29 is the first row (NaN)


def test_long_entry_below_threshold_and_none_inside_it():
    n = len(MARCH)
    falling = run(frame(np.full(n, 100.0), np.r_[50.0, np.full(n - 1, 52.0)]))  # bonds +4 %
    assert list(falling.index[falling["rebal_entry_long"] == 1].strftime("%m-%d")) == ["03-25"]
    small = run(frame(np.r_[100.0, np.full(n - 1, 101.0)], np.full(n, 50.0)))  # +1 % < 2 %
    assert small[["rebal_entry_long", "rebal_entry_short"]].loc["2024-03-01":].sum().sum() == 0


def test_month_without_its_first_trading_day_has_no_drift():
    df = frame(np.full(len(MARCH), 100.0), np.full(len(MARCH), 50.0)).drop(
        pd.Timestamp("2024-03-01")
    )
    out = run(df)
    assert out.loc["2024-03-04":"2024-03-28", "rebal_drift"].isna().all()
    assert out.loc["2024-03-04":"2024-03-28", "rebal_entry_long"].sum() == 0


def test_first_row_nan_prefix_stable_and_missing_inputs():
    df = frame(np.linspace(100, 110, len(MARCH)), np.linspace(50, 49, len(MARCH)))
    full = run(df)
    assert full[mod.REBAL_COLUMNS].iloc[0].isna().all()
    pd.testing.assert_frame_equal(full.iloc[:15], run(df.iloc[:15]))
    bare = run(df[["C"]])
    assert bare[mod.REBAL_COLUMNS].isna().all().all()
