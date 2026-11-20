"""Bars per calendar year are derived from the data, not from the timeframe."""

import numpy as np
import pandas as pd
import pytest

from fwbg.optimization.annualization import bars_per_year


def _session_index(
    years: int, bars_per_day: int, start_hour: int = 9
) -> pd.DatetimeIndex:
    """Weekday sessions of *bars_per_day* hourly bars (a 7-bar/day instrument)."""
    days = pd.bdate_range("2020-01-01", periods=years * 261)
    return pd.DatetimeIndex(
        [
            d + pd.Timedelta(hours=start_hour + h)
            for d in days
            for h in range(bars_per_day)
        ]
    )


def test_seven_bars_per_weekday_is_about_1800_not_6000():
    index = _session_index(3, 7)
    value = bars_per_year(index)
    # 7 bars * ~261 weekdays
    assert value == pytest.approx(7 * 261, rel=0.02)
    # the old bars_per_hour * 24 * 250 gave 6000 and inflated the Sharpe by this factor
    assert np.sqrt(6000 / value) == pytest.approx(1.84, abs=0.05)


def test_round_the_clock_hourly_is_8760():
    index = pd.date_range("2020-01-01", periods=24 * 365 * 2, freq="h")
    assert bars_per_year(index) == pytest.approx(8766, rel=0.01)


def test_daily_weekdays():
    index = pd.bdate_range("2019-01-01", "2023-12-31")
    assert bars_per_year(index) == pytest.approx(261, rel=0.02)


def test_partial_first_and_last_day_do_not_move_the_median():
    index = _session_index(2, 7)
    trimmed = index[3:-4]
    assert bars_per_year(trimmed) == pytest.approx(bars_per_year(index), rel=0.01)


def test_fallback_for_a_non_datetime_index():
    assert bars_per_year(pd.RangeIndex(100), fallback=6000.0) == 6000.0
    with pytest.raises(ValueError):
        bars_per_year(pd.RangeIndex(100))
