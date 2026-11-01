"""Gotobi Indicator Plugin.

Japanese companies settle invoices on "gotobi" days: the 5th, 10th, 15th, 20th and
25th of a month and the month's last day, moved to the previous Japanese business day
when that day is a weekend, a national holiday (``holidays.country_holidays("JP")``) or a bank closure
(31 December to 3 January). Importers buy USD at the Tokyo 9:55 fixing on these days
(Ito & Yamada 2017 on order imbalances at the Tokyo fix).

The columns mark bars by the Tokyo time of their close (start + ``bar_minutes``):

- ``gotobi_day``: 1 when the bar closes on a gotobi date in Tokyo
- ``gotobi_entry``: 1 on a gotobi date for the bar closing at ``entry_hour``:00 Tokyo
- ``gotobi_exit``: 1 on a gotobi date for the bar closing at ``exit_hour``:00 Tokyo,
  an exit column for ``signal_exit``

The settlement calendar is known in advance, so the columns are unshifted and meant
for a decision at the bar's close (``entry_delay=0``); as for every indicator, the
first row is NaN. Naive timestamps are UTC.
"""

from typing import List

import holidays
import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator

GOTOBI_COLUMNS = ["gotobi_day", "gotobi_entry", "gotobi_exit"]
TOKYO = "Asia/Tokyo"
SETTLEMENT_DAYS = (5, 10, 15, 20, 25)


def _closed(day: pd.Timestamp, jp: holidays.HolidayBase) -> bool:
    """Return whether *day* is not an open Japanese settlement day."""
    bank_closure = (day.month == 12 and day.day == 31) or (day.month == 1 and day.day <= 3)
    return day.dayofweek >= 5 or day in jp or bank_closure


def gotobi_dates(first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    """Gotobi dates (naive, Tokyo calendar) of every month from *first* to *last*."""
    months = pd.date_range(first.normalize().replace(day=1), last.normalize(), freq="MS")
    jp = holidays.country_holidays("JP", years=range(first.year - 1, last.year + 2))
    out = set()
    for start in months:
        days = [start.replace(day=k) for k in SETTLEMENT_DAYS]
        for day in [*days, start + pd.offsets.MonthEnd(0)]:
            while _closed(day, jp):
                day -= pd.Timedelta(days=1)
            if not _closed(day, jp):
                out.add(day)
    return pd.DatetimeIndex(sorted(out))


@register_indicator("gotobi")
class GotobiIndicator(BaseIndicator):
    """Japanese settlement (gotobi) days with entry/exit flags at fixed Tokyo hours."""

    name = "gotobi"
    version = "1.0.0"
    group = "calendar"
    benefits_from_stationary = False

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Append gotobi flags using each bar's Tokyo-local closing timestamp."""
        p = {**self.get_default_params(), **params}
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("gotobi needs a DatetimeIndex")
        out = pd.DataFrame(index=df.index, columns=GOTOBI_COLUMNS, dtype=float)
        if len(df):
            utc = df.index.tz_localize("UTC") if df.index.tz is None else df.index
            closes = (utc + pd.Timedelta(minutes=int(p["bar_minutes"]))).tz_convert(TOKYO)
            dates = closes.tz_localize(None).normalize()
            gotobi = dates.isin(gotobi_dates(dates.min(), dates.max()))
            on_hour = (
                (closes.minute == 0)
                & (closes.second == 0)
                & (closes.microsecond == 0)
                & (closes.nanosecond == 0)
            )
            out["gotobi_day"] = gotobi.astype(float)
            out["gotobi_entry"] = (gotobi & on_hour & (closes.hour == p["entry_hour"])).astype(
                float
            )
            out["gotobi_exit"] = (gotobi & on_hour & (closes.hour == p["exit_hour"])).astype(float)
            out.iloc[0] = np.nan  # indicator contract: the first row is NaN
        return pd.concat([df, out], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        return list(GOTOBI_COLUMNS)

    def get_signal_columns(self, params=None) -> List[str]:
        return list(GOTOBI_COLUMNS)

    @classmethod
    def get_default_params(cls) -> dict:
        return {"entry_hour": 10, "exit_hour": 15, "bar_minutes": 60}

    @classmethod
    def get_param_schema(cls) -> dict:
        """Describe supported entry, exit, and bar-length parameters."""
        return {
            "entry_hour": {
                "type": "int",
                "default": 10,
                "min": 0,
                "max": 23,
                "description": "gotobi_entry marks the bar closing at this hour (Tokyo)",
            },
            "exit_hour": {
                "type": "int",
                "default": 15,
                "min": 0,
                "max": 23,
                "description": "gotobi_exit marks the bar closing at this hour (Tokyo)",
            },
            "bar_minutes": {
                "type": "int",
                "default": 60,
                "min": 1,
                "max": 1440,
                "description": "Bar length in minutes; a bar closes at its start plus this",
            },
        }


__all__ = ["GotobiIndicator", "GOTOBI_COLUMNS", "gotobi_dates"]
