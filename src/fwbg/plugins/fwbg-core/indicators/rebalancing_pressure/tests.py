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


def test_weekend_bars_do_not_repeat_month_end_signals():
    df = frame(np.full(len(MARCH), 100.0), np.full(len(MARCH), 50.0))
    weekend = pd.DataFrame(
        {"C": 1.0, "macro_spy": 100.0, "macro_tlt": 50.0, "macro_ief": 50.0},
        index=pd.DatetimeIndex(["2024-03-30", "2024-03-31"]),
    )
    out = run(pd.concat([df, weekend]).sort_index())
    assert list(out.index[out["rebal_exit"] == 1].strftime("%Y-%m-%d")) == [
        "2024-03-28"
    ]


def test_long_entry_below_threshold_and_none_inside_it():
    n = len(MARCH)
    falling = run(
        frame(np.full(n, 100.0), np.r_[50.0, np.full(n - 1, 52.0)])
    )  # bonds +4 %
    assert list(falling.index[falling["rebal_entry_long"] == 1].strftime("%m-%d")) == [
        "03-25"
    ]
    small = run(
        frame(np.r_[100.0, np.full(n - 1, 101.0)], np.full(n, 50.0))
    )  # +1 % < 2 %
    assert (
        small[["rebal_entry_long", "rebal_entry_short"]].loc["2024-03-01":].sum().sum()
        == 0
    )


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


def own_close_frame(spy_path, bond_path) -> pd.DataFrame:
    """Bars on MARCH whose input columns hold the bar's OWN close (no lag)."""
    return pd.DataFrame(
        {
            "C": 1.0,
            "macro_spy": pd.Series(spy_path, index=MARCH),
            "macro_tlt": pd.Series(bond_path, index=MARCH),
            "macro_ief": pd.Series(bond_path, index=MARCH),
        },
        index=MARCH,
    )


def test_inputs_lagged_false_lags_the_inputs_by_one_bar():
    n = len(MARCH)
    spy = np.linspace(100, 108, n)
    bond = np.linspace(50, 49, n)
    lagged = run(frame(spy, bond))
    unlagged = run(own_close_frame(spy, bond), inputs_lagged=False)
    pd.testing.assert_frame_equal(
        lagged[mod.REBAL_COLUMNS], unlagged[mod.REBAL_COLUMNS]
    )
    # the same data read as already lagged leaks the bar's own close
    leaked = run(own_close_frame(spy, bond))
    assert not np.allclose(
        leaked["rebal_drift"].iloc[2:], lagged["rebal_drift"].iloc[2:], equal_nan=True
    )


def test_bar_does_not_see_its_own_close_when_inputs_are_not_lagged():
    n = len(MARCH)
    spy = np.full(n, 100.0)
    bond = np.full(n, 50.0)
    base = run(own_close_frame(spy, bond), inputs_lagged=False)
    for position in range(2, n):
        bumped = spy.copy()
        bumped[position] = 130.0  # +30 % on one bar
        out = run(own_close_frame(bumped, bond), inputs_lagged=False)
        day = MARCH[position]
        # that bar's own row is unchanged, and so is every earlier row
        pd.testing.assert_series_equal(
            out.loc[:day, "rebal_drift"], base.loc[:day, "rebal_drift"]
        )


def test_inputs_lagged_false_refuses_intraday_bars():
    index = pd.date_range("2024-03-04", periods=6, freq="12h")
    df = pd.DataFrame(
        {"C": 1.0, "macro_spy": 100.0, "macro_tlt": 50.0, "macro_ief": 50.0},
        index=index,
    )
    with pytest.raises(ValueError, match="one bar per date"):
        run(df, inputs_lagged=False)
    run(df)  # the default does not care
