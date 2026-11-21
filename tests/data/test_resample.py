"""Resampled bars are labelled with their start time and never use later data."""

import numpy as np
import pandas as pd
import pytest

from fwbg.data.resample import resample_ohlcv


def _daily(start: str, periods: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="D")
    rng = np.random.default_rng(7)
    close = 100 + rng.normal(size=periods).cumsum()
    return pd.DataFrame(
        {
            "O": close + 0.1,
            "H": close + 1.0,
            "L": close - 1.0,
            "C": close,
            "V": np.arange(1, periods + 1, dtype=float),
        },
        index=idx,
    )


def test_weekly_bars_are_labelled_with_the_monday_that_starts_them():
    daily = _daily("2024-01-01", 21)  # Mon 2024-01-01 .. Sun 2024-01-21
    weekly = resample_ohlcv(daily, "WEEK_1")

    assert list(weekly.index) == list(
        pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"])
    )
    assert all(ts.dayofweek == 0 for ts in weekly.index)


def test_weekly_bar_aggregates_exactly_its_own_monday_to_sunday():
    daily = _daily("2024-01-01", 21)
    weekly = resample_ohlcv(daily, "WEEK_1")

    for label, bar in weekly.iterrows():
        week = daily.loc[label : label + pd.Timedelta(days=6)]
        assert bar["O"] == week["O"].iloc[0]
        assert bar["H"] == week["H"].max()
        assert bar["L"] == week["L"].min()
        assert bar["C"] == week["C"].iloc[-1]
        assert bar["V"] == week["V"].sum()


@pytest.mark.parametrize("timeframe", ["DAY_1", "WEEK_1"])
def test_resampling_has_no_lookahead_at_every_cut_point(timeframe):
    """Truncating the input at any row leaves every already-complete bar unchanged."""
    base = _daily("2024-01-01", 60)
    # Intraday bars so day-level bars also get exercised on a partial last bar.
    idx = pd.date_range("2024-01-01", periods=60 * 4, freq="6h")
    rng = np.random.default_rng(3)
    close = 100 + rng.normal(size=len(idx)).cumsum()
    intraday = pd.DataFrame(
        {"O": close, "H": close + 1, "L": close - 1, "C": close + 0.5, "V": 1.0},
        index=idx,
    )
    for frame in (base, intraday):
        full = resample_ohlcv(frame, timeframe)
        for cut in range(2, len(frame) + 1):
            part = resample_ohlcv(frame.iloc[:cut], timeframe)
            complete = part.iloc[:-1]  # the last bar may still be forming
            pd.testing.assert_frame_equal(complete, full.loc[complete.index])
            # A bar's label never lies after the data it was built from.
            assert part.index[-1] <= frame.index[cut - 1]
