"""Tests for the scheduled_events indicator plugin."""

import logging

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
    return (
        ScheduledEventsIndicator()
        .compute(daily(warm_up.append(index)), **params)
        .iloc[1:]
    )


def test_first_row_is_nan(events):
    out = ScheduledEventsIndicator().compute(
        daily(["2024-01-02", "2024-01-03"]), events_file=events
    )
    assert out[mod.EVENT_COLUMNS].iloc[0].isna().all()
    assert out["evt_days_ahead"].iloc[1] == 5.0  # Jan 3 -> Jan 10


def test_event_on_closed_day_before_first_trading_day_maps_forward(tmp_path):
    path = tmp_path / "events.csv"
    pd.DataFrame({"date": ["2024-01-01"]}).to_csv(path, index=False)  # NYSE holiday

    out = compute(["2024-01-02"], events_file=str(path))

    assert out["evt_days_ahead"].iloc[0] == 0.0


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
    assert out.index[out["evt_window_start"] == 1].tolist() == [
        "2024-01-02",
        "2024-01-17",
    ]
    assert out.index[out["evt_window_end"] == 1].tolist() == [
        "2024-01-10",
        "2024-01-24",
    ]


def test_filter_selects_events(events):
    out = compute(
        ["2024-01-09", "2024-01-12"],
        events_file=events,
        filter_column="kind",
        filter_values=["B"],
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

    assert (
        out["evt_days_ahead"].iloc[0] == 1.0
    )  # Mar 29 is an NYSE holiday; next day is Apr 1


def test_no_later_event_is_nan(events):
    out = compute(["2024-03-01"], events_file=events)
    assert (
        out[["evt_days_ahead", "evt_window", "evt_window_start", "evt_window_end"]]
        .isna()
        .all(axis=None)
    )


def test_window_is_nan_after_the_last_event_but_known_before_it(events):
    # last event: 2024-02-20 (Tue). Feb 16 is 1 trading day before it (Feb 19 closed);
    # Feb 21 and later have no known later event.
    out = compute(bdays("2024-02-12", "2024-02-23"), events_file=events).set_index(
        pd.Index(bdays("2024-02-12", "2024-02-23"))
    )
    assert out.loc["2024-02-14", "evt_window"] == 1.0  # 3 trading days ahead
    assert out.loc["2024-02-16", "evt_window"] == 0.0  # 1 ahead, inside window_end
    assert out.loc["2024-02-16", "evt_window_end"] == 1.0
    assert out.loc["2024-02-20", "evt_window"] == 0.0  # event day, days_ahead 0
    cols = mod.EVENT_COLUMNS
    assert out.loc["2024-02-21":, cols].isna().all(axis=None)
    assert out.loc[:"2024-02-20", cols].notna().all(axis=None)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"events_file": "/nonexistent/events.csv"},
        {"date_column": "when"},
        {"date_column": ""},
    ],
)
def test_missing_file_or_column_warns_and_gives_nan(events, caplog, params):
    params = (
        {"events_file": events, **params}
        if "events_file" not in params and params
        else params
    )
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


def test_mixed_date_formats_are_all_read(tmp_path):
    path = tmp_path / "mixed.csv"
    pd.DataFrame(
        {
            "date": [
                "2024-01-10",
                "2024-01-11 14:00:00",
                "Jan 12, 2024",
                "2024/01/16",
                "2024-01-17T00:00:00+02:00",
                "",
            ]
        }
    ).to_csv(path, index=False)

    dates = mod.event_dates(str(path), "date")

    assert dates.strftime("%Y-%m-%d").tolist() == [
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
        "2024-01-16",
        "2024-01-17",
    ]


def test_unparsable_date_raises_instead_of_dropping_the_row(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-10", "10th of Jan", "not a date"]}).to_csv(
        path, index=False
    )

    with pytest.raises(mod.EventDateError, match="2 unparsable"):
        mod.event_dates(str(path), "date")
    # and the indicator does not turn it into an all-NaN warning
    with pytest.raises(ValueError, match="unparsable"):
        compute(["2024-01-09"], events_file=str(path))


def test_last_event_can_still_close_the_final_window(tmp_path):
    path = tmp_path / "single.csv"
    path.write_text("date\n2025-01-10\n")
    dates = bdays("2025-01-02", "2025-01-15")
    out = compute(
        dates, events_file=str(path), calendar="weekdays", window_start=3, window_end=0
    ).set_axis(dates)
    assert out.index[out["evt_window_start"] == 1].tolist() == ["2025-01-07"]
    assert out.index[out["evt_window_end"] == 1].tolist() == ["2025-01-13"]
    assert pd.isna(out.loc["2025-01-13", "evt_days_ahead"])
    assert pd.isna(out.loc["2025-01-14", "evt_window_end"])
