"""Tests for the gotobi indicator plugin."""

import pandas as pd

from fwbg.plugins import import_plugin_module

mod = import_plugin_module("fwbg-core", "indicators", "gotobi")
GotobiIndicator = mod.GotobiIndicator


def hourly(start: str, end: str) -> pd.DataFrame:
    return pd.DataFrame({"C": 1.0}, index=pd.date_range(start, end, freq="h", inclusive="left"))


def dates(first: str, last: str) -> list[str]:
    return list(mod.gotobi_dates(pd.Timestamp(first), pd.Timestamp(last)).strftime("%m-%d"))


def test_settlement_days_roll_back_over_weekends_holidays_and_bank_closures():
    # March 2024: 10th is a Sunday -> 8th (Fri); 20th is a holiday (Vernal Equinox) -> 19th;
    # month end 31st is a Sunday -> 29th
    assert dates("2024-03-01", "2024-03-31") == [
        "03-05",
        "03-08",
        "03-15",
        "03-19",
        "03-25",
        "03-29",
    ]
    # 31 Dec is a bank closure; 2024-12-31 is a Tuesday -> 30th
    assert dates("2024-12-01", "2024-12-31")[-1] == "12-30"
    # 5 January 2025 is a Sunday; rollback crosses New Year's closure into December.
    january_dates = mod.gotobi_dates(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-31"))
    assert january_dates[0] == pd.Timestamp("2024-12-30")


def test_flags_follow_the_tokyo_close_of_each_bar():
    out = GotobiIndicator().compute(hourly("2024-03-14 12:00", "2024-03-15 12:00"))
    entry = out.index[out["gotobi_entry"] == 1]
    exit_ = out.index[out["gotobi_exit"] == 1]
    # 15 March: the bar 00:00-01:00 UTC closes 10:00 Tokyo, the bar 05:00-06:00 UTC closes 15:00
    assert list(entry.strftime("%d %H:%M")) == ["15 00:00"]
    assert list(exit_.strftime("%d %H:%M")) == ["15 05:00"]
    # the 23:00 UTC bar of the 14th closes at 09:00 on the 15th in Tokyo: a gotobi bar
    assert out.loc["2024-03-14 23:00", "gotobi_day"] == 1.0
    assert out.loc["2024-03-14 13:00", "gotobi_day"] == 0.0


def test_entry_requires_an_exact_hour_close():
    index = pd.DatetimeIndex(
        ["2024-03-14 23:00:00", "2024-03-15 00:00:00", "2024-03-15 00:00:00.000000001"]
    )
    out = GotobiIndicator().compute(pd.DataFrame({"C": 1.0}, index=index))
    assert out["gotobi_entry"].iloc[1:].tolist() == [1.0, 0.0]


def test_hours_and_bar_length_are_parameters():
    df = pd.DataFrame({"C": 1.0}, index=pd.date_range("2024-03-14 20:00", periods=40, freq="15min"))
    out = GotobiIndicator().compute(df, entry_hour=8, exit_hour=9, bar_minutes=15)
    assert list(out.index[out["gotobi_entry"] == 1].strftime("%H:%M")) == ["22:45"]
    assert list(out.index[out["gotobi_exit"] == 1].strftime("%H:%M")) == ["23:45"]


def test_first_row_nan_prefix_stable_and_tz_aware():
    df = hourly("2024-03-14 00:00", "2024-03-21 00:00")
    full = GotobiIndicator().compute(df)
    assert full[mod.GOTOBI_COLUMNS].iloc[0].isna().all()
    pd.testing.assert_frame_equal(full.iloc[:50], GotobiIndicator().compute(df.iloc[:50]))
    aware = GotobiIndicator().compute(df.tz_localize("UTC"))
    assert aware["gotobi_entry"].iloc[1:].tolist() == full["gotobi_entry"].iloc[1:].tolist()


def test_empty_frame():
    out = GotobiIndicator().compute(pd.DataFrame({"C": []}, index=pd.DatetimeIndex([])))
    assert list(out.columns) == ["C", *mod.GOTOBI_COLUMNS] and out.empty
