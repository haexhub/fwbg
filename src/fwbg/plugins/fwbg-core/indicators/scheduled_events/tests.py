"""Tests for the scheduled_events indicator plugin."""

import logging

import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

mod = import_plugin_module("fwbg-core", "indicators", "scheduled_events")
ScheduledEventsIndicator = mod.ScheduledEventsIndicator


def daily(dates) -> pd.DataFrame:
    return pd.DataFrame({"C": 1.0}, index=pd.DatetimeIndex(pd.to_datetime(dates)))


def bdays(start, end) -> list[str]:
    """NYSE trading days from *start* to *end*."""
    days = mod._trading_days("NYSE", pd.Timestamp(start), pd.Timestamp(end))
    return [d for d in days.strftime("%Y-%m-%d") if start <= d <= end]


@pytest.fixture
def events(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-10", "2024-01-11", "2024-01-25", "2024-02-20"],
            "kind": ["A", "B", "A", "B"],
        }
    ).to_csv(path, index=False)
    return str(path)


def compute(dates, **params) -> pd.DataFrame:
    """Values for *dates*, after a warm-up bar the day before (the first row is NaN)."""
    index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    warm_up = index[:1] - pd.Timedelta(days=1)
    return ScheduledEventsIndicator().compute(daily(warm_up.append(index)), **params).iloc[1:]


def test_first_row_is_nan(events):
    out = ScheduledEventsIndicator().compute(
        daily(["2024-01-02", "2024-01-03"]), events_file=events
    )
    assert out[mod.EVENT_COLUMNS].iloc[0].isna().all()
    assert out["evt_days_ahead"].iloc[1] == 5.0  # Jan 3 -> Jan 10


def test_days_ahead_counts_exchange_trading_days(events):
    # Jan 15 2024 (MLK day) is an NYSE holiday: from Jan 12 to Jan 25 are 8 trading days
    out = compute(["2024-01-10", "2024-01-11", "2024-01-12"], events_file=events)
    assert out["evt_days_ahead"].tolist() == [0.0, 0.0, 8.0]


def test_window_merges_clustered_events_and_marks_its_edges(events):
    dates = bdays("2023-12-20", "2024-01-26")
    out = compute(dates, events_file=events, window_start=6, window_end=2).set_index(
        pd.Index(dates)
    )
    on = out.index[out["evt_window"] == 1].tolist()
    # Jan 10 and 11 merge: from 6 trading days before Jan 10 (Jan 2) to 2 before Jan 11
    # (Jan 9); Jan 25: from Jan 17 to Jan 23 (Jan 15 is an NYSE holiday)
    assert on == [
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-17",
        "2024-01-18",
        "2024-01-19",
        "2024-01-22",
        "2024-01-23",
    ]
    assert out.index[out["evt_window_start"] == 1].tolist() == ["2024-01-02", "2024-01-17"]
    assert out.index[out["evt_window_end"] == 1].tolist() == ["2024-01-10", "2024-01-24"]


def test_filter_selects_events(events):
    out = compute(
        ["2024-01-09", "2024-01-12"], events_file=events, filter_column="kind", filter_values=["B"]
    )
    # only Jan 11 and Feb 20 count
    assert out["evt_days_ahead"].tolist() == [2.0, 25.0]


def test_events_after_the_last_bar_are_counted(events):
    out = compute(["2024-02-14"], events_file=events)
    assert out["evt_days_ahead"].iloc[0] == 3.0  # Feb 19 is Presidents Day


def test_month_end_event_on_closed_day_maps_to_next_month_trading_day(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame({"date": ["2024-03-31"]}).to_csv(path, index=False)  # Sunday

    out = compute(["2024-03-28"], events_file=str(path))

    assert out["evt_days_ahead"].iloc[0] == 2.0  # Mar 29 and Apr 1


def test_no_later_event_is_nan(events):
    out = compute(["2024-03-01"], events_file=events)
    assert np.isnan(out["evt_days_ahead"].iloc[0]) and out["evt_window"].iloc[0] == 0.0


@pytest.mark.parametrize(
    "params", [{}, {"events_file": "/nonexistent/events.csv"}, {"date_column": "when"}]
)
def test_missing_file_or_column_warns_and_gives_nan(events, caplog, params):
    params = {"events_file": events, **params} if "events_file" not in params and params else params
    with caplog.at_level(logging.WARNING):
        out = compute(["2024-01-09"], **params)
    assert out[mod.EVENT_COLUMNS].isna().all().all()
    assert "scheduled_events" in caplog.text


def test_window_end_after_window_start_is_refused(events):
    with pytest.raises(ValueError, match="window_end"):
        compute(["2024-01-09"], events_file=events, window_start=2, window_end=3)


def test_a_prefix_of_the_data_gives_the_same_values(events):
    dates = bdays("2024-01-02", "2024-02-28")
    full = compute(dates, events_file=events)
    part = compute(dates[:20], events_file=events)
    pd.testing.assert_frame_equal(full.iloc[:20], part)
