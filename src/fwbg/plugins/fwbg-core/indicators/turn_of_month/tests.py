"""Tests for the turn_of_month indicator plugin."""

import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

mod = import_plugin_module("fwbg-core", "indicators", "turn_of_month")
TurnOfMonthIndicator = mod.TurnOfMonthIndicator


def daily(dates) -> pd.DataFrame:
    return pd.DataFrame({"C": 1.0}, index=pd.DatetimeIndex(pd.to_datetime(dates)))


def compute(dates, **params) -> pd.DataFrame:
    """Values for *dates*, after a warm-up bar the day before (the first row is NaN)."""
    index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    warm_up = index[:1] - pd.Timedelta(days=1)
    return TurnOfMonthIndicator().compute(daily(warm_up.append(index)), **params).iloc[1:]


def test_first_row_is_nan():
    out = TurnOfMonthIndicator().compute(daily(["2024-01-02", "2024-01-03"]))
    assert out[mod.TOM_COLUMNS].iloc[0].isna().all() and out["tom_day"].iloc[1] == 2.0


def test_counts_follow_the_exchange_calendar():
    # March 2024: Good Friday (29th) is an NYSE holiday, so the 28th is the last trading day
    out = compute(["2024-03-26", "2024-03-27", "2024-03-28", "2024-04-01", "2024-04-02"])
    assert out["tom_days_left"].tolist()[:3] == [2.0, 1.0, 0.0]
    assert out["tom_day"].tolist() == [18.0, 19.0, 20.0, 1.0, 2.0]
    weekdays = compute(["2024-03-28"], calendar="weekdays")
    assert weekdays["tom_days_left"].iloc[0] == 1.0


def test_calendar_trading_week_controls_generated_dates(monkeypatch):
    # XSAU trades Sunday through Thursday; Friday is not a trading day.
    monkeypatch.setattr(mod, "CALENDARS", [*mod.CALENDARS, "XSAU"])
    monkeypatch.setattr(mod.holidays, "financial_holidays", lambda *args, **kwargs: {})
    days = mod.trading_days("XSAU", pd.Timestamp("2024-09-01"), pd.Timestamp("2024-09-08"))
    assert days.strftime("%Y-%m-%d").tolist()[:6] == [
        "2024-09-01",
        "2024-09-02",
        "2024-09-03",
        "2024-09-04",
        "2024-09-05",
        "2024-09-08",
    ]
    assert not any(day.dayofweek in (4, 5) for day in days)


def test_missing_bars_do_not_shift_the_counts():
    full = compute(pd.bdate_range("2024-05-20", "2024-06-07"))
    sparse = compute(["2024-05-30", "2024-06-05"])  # Memorial Day 27 May is closed
    assert sparse["tom_days_left"].iloc[0] == 1.0 and sparse["tom_day"].iloc[1] == 3.0
    pd.testing.assert_frame_equal(full.loc[sparse.index], sparse)


def test_entry_and_exit_flags():
    out = compute(pd.bdate_range("2024-01-25", "2024-02-08"), entry_days_left=2, exit_day=4)
    assert list(out.index[out["tom_entry_long"] == 1].strftime("%m-%d")) == ["01-29"]
    assert list(out.index[out["tom_exit_long"] == 1].strftime("%m-%d")) == ["02-06"]
    default = compute(pd.bdate_range("2024-01-25", "2024-02-08"))
    assert list(default.index[default["tom_entry_long"] == 1].strftime("%m-%d")) == ["01-30"]
    assert list(default.index[default["tom_exit_long"] == 1].strftime("%m-%d")) == ["02-05"]


def test_bar_on_a_closed_day_counts_the_trading_days_before_it():
    # New Year's Day 2024 is closed: no trading day of January on or before it
    out = compute(["2024-01-01", "2024-01-02", "2024-03-29"])
    assert out["tom_day"].tolist() == [0.0, 1.0, 20.0]
    assert out["tom_days_left"].tolist() == [21.0, 20.0, 0.0]


def test_prefix_stable_and_time_zone_aware_index():
    dates = pd.bdate_range("2023-12-15", "2024-02-15")
    full = compute(dates)
    pd.testing.assert_frame_equal(full.iloc[:20], compute(dates[:20]))
    aware = TurnOfMonthIndicator().compute(daily(dates).tz_localize("UTC"))
    assert aware["tom_day"].iloc[1:].tolist() == full["tom_day"].iloc[1:].tolist()


def test_unknown_calendar_and_empty_frame():
    with pytest.raises(ValueError, match="calendar"):
        compute(["2024-01-02"], calendar="MOON")
    empty = TurnOfMonthIndicator().compute(daily([]))
    assert list(empty.columns) == ["C", *mod.TOM_COLUMNS] and empty.empty
