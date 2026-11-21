"""Unit tests for fwbg.data.trading_hours."""

import json

import numpy as np
import pandas as pd
import pytest

from fwbg.data import assets, trading_hours
from fwbg.data.trading_hours import TradingHoursError, asset_sessions, parse_trading_hours

FX = {"tz": "America/New_York", "windows": [{"days": "Sun-Thu", "start": "17:00", "end": "17:00"}]}
CASH = {
    "tz": "America/New_York",
    "windows": [{"days": "Mon-Fri", "start": "09:30", "end": "16:00"}],
}


def utc(*stamps: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(stamps)))  # naive = UTC


def test_forex_week_opens_sunday_and_closes_friday_new_york_time():
    hours = parse_trading_hours(FX)
    # winter (EST = UTC-5): open Sunday 22:00 UTC, close Friday 22:00 UTC
    bars = utc(
        "2026-01-04 21:00",
        "2026-01-04 22:00",
        "2026-01-09 21:00",
        "2026-01-09 22:00",
        "2026-01-10 12:00",
    )
    assert hours.overlaps(bars, 60).tolist() == [False, True, True, False, False]
    # summer (EDT = UTC-4): the same local hours are an hour earlier in UTC
    assert hours.overlaps(utc("2026-07-05 21:00", "2026-07-10 21:00"), 60).tolist() == [True, False]


def test_window_inside_a_day_and_partial_bars():
    hours = parse_trading_hours(CASH)
    bars = utc(
        "2026-03-02 14:15", "2026-03-02 14:30", "2026-03-02 20:45", "2026-03-02 21:00"
    )  # EST
    assert hours.open_minutes(bars, 15).tolist() == [0, 15, 15, 0]
    assert hours.open_minutes(utc("2026-03-02 14:00"), 60).tolist() == [
        30
    ]  # 09:00-10:00 opens at 09:30
    assert not hours.overlaps(utc("2026-03-07 15:00"), 60)[0]  # Saturday


def test_daily_bars_are_open_when_any_part_of_the_day_is():
    hours = parse_trading_hours(FX)
    days = pd.date_range("2026-01-03", periods=3, freq="D")  # Sat, Sun, Mon at 00:00 UTC
    assert hours.overlaps(days, 1440).tolist() == [False, True, True]
    assert hours.open_minutes(days, 1440).tolist() == [0, 120, 1440]


def test_day_ranges_wrap_and_lists_combine():
    hours = parse_trading_hours(
        {"tz": "UTC", "windows": [{"days": "Fri-Mon", "start": "00:00", "end": "24:00"}]}
    )
    days = pd.date_range("2026-01-05", periods=7, freq="D")  # Mon..Sun
    assert hours.overlaps(days, 60).tolist() == [True, False, False, False, True, True, True]
    listed = parse_trading_hours(
        {"tz": "UTC", "windows": [{"days": "Mon,Wed", "start": "00:00", "end": "00:00"}]}
    )
    assert listed.overlaps(days, 60).tolist() == [True, False, True, False, False, False, False]


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"windows": FX["windows"]}, "tz"),
        ({"tz": "Mars/Base", "windows": FX["windows"]}, "time zone"),
        ({"tz": "UTC", "windows": []}, "windows"),
        ({"tz": "UTC", "windows": [{"days": "Mon-Fry", "start": "09:00", "end": "17:00"}]}, "day"),
        ({"tz": "UTC", "windows": [{"days": "Mon", "start": "9:00", "end": "17:00"}]}, "start"),
        ({"tz": "UTC", "windows": [{"days": "Mon", "start": "09:00", "end": "24:30"}]}, "end"),
        ("UTC", "object"),
    ],
)
def test_malformed_definitions_raise(raw, match):
    with pytest.raises(TradingHoursError, match=match):
        parse_trading_hours(raw)


@pytest.fixture
def no_overrides(monkeypatch):
    monkeypatch.setattr(trading_hours, "load_asset_overrides", lambda: {})


def test_builtin_sessions_by_group_class_and_currency_pair(no_overrides):
    assert set(asset_sessions("NAS100")) == {"tradable", "cash"}
    assert asset_sessions("EURGBP")["tradable"].tz == "America/New_York"  # registry class FOREX
    assert "tradable" in asset_sessions("AUDNZD")  # not in the registry, but a pair of currencies
    assert asset_sessions("BTCUSD")["tradable"].tz == "UTC"
    assert asset_sessions("QQQ") == {}  # unknown: no forex hours by default


def test_override_from_asset_meta_wins(monkeypatch):
    monkeypatch.setattr(
        trading_hours,
        "load_asset_overrides",
        lambda: {"EURGBP": {"trading_hours": {"tradable": CASH}}},
    )
    hours = asset_sessions("EURGBP")["tradable"]
    assert hours.windows[0].start == 9 * 60 + 30
    monkeypatch.setattr(
        trading_hours,
        "load_asset_overrides",
        lambda: {"EURGBP": {"trading_hours": {"tradable": {}}}},
    )
    with pytest.raises(TradingHoursError, match="EURGBP"):
        asset_sessions("EURGBP")


def test_override_is_read_from_the_workspace_file(tmp_path, monkeypatch):
    (tmp_path / "_asset_meta.json").write_text(
        json.dumps({"QQQ": {"trading_hours": {"tradable": CASH}}})
    )
    monkeypatch.setattr(assets, "_asset_meta_path", lambda: tmp_path / "_asset_meta.json")
    monkeypatch.setattr(assets, "_OVERRIDES_CACHE", None)
    hours = asset_sessions("QQQ")["tradable"]
    assert np.array_equal(hours.overlaps(utc("2026-03-02 15:00"), 60), [True])
