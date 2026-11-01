"""Turn-of-the-month Indicator Plugin.

Counts exchange trading days within each calendar month, so rules can hold a
position from the last trading days of one month into the first trading days of
the next (Ariel 1987; Lakonishok & Smidt 1988; McConnell & Xu 2008).

Trading days follow the selected exchange's trading week and exclude its
holidays (``holidays.financial_holidays``, e.g. ``NYSE``); ``calendar="weekdays"``
uses every Monday-Friday date. The counts come from the calendar, not from the bars in
the data, so a missing bar does not shift them. The exchange calendar is
published in advance, so the columns are unshifted: each refers to the bar's own
trading date and is meant for a decision at that bar's close (``entry_delay=0``).
As for every indicator, the first row is NaN. Bars are expected to be daily bars
labelled with their trading date.

Columns:

- ``tom_day``: trading day of the month, 1 on the first trading day
- ``tom_days_left``: trading days left in the month after this one, 0 on the last
- ``tom_entry_long``: 1 when ``tom_days_left == entry_days_left``
- ``tom_exit_long``: 1 when ``tom_day == exit_day``, an exit column for
  ``signal_exit``; entering at ``entry_days_left = 1`` and exiting at
  ``exit_day = 3`` holds the classic window from trading day -1 to +3
"""

from typing import List

import holidays
import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator

TOM_COLUMNS = ["tom_day", "tom_days_left", "tom_entry_long", "tom_exit_long"]
WEEKDAYS = "weekdays"
CALENDARS = [WEEKDAYS, *sorted(holidays.list_supported_financial())]
TRADING_WEEKMASKS = {"XSAU": "Sun Mon Tue Wed Thu"}


def trading_days(calendar: str, first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    """Trading dates of *calendar* covering the whole months from *first* to *last*."""
    start = first.normalize().replace(day=1)
    end = last.normalize() + pd.offsets.MonthEnd(0)
    if calendar != WEEKDAYS and calendar not in CALENDARS:
        raise ValueError(f"turn_of_month: unknown calendar {calendar!r}; use one of {CALENDARS}")
    weekmask = TRADING_WEEKMASKS.get(calendar, "Mon Tue Wed Thu Fri")
    days = pd.date_range(
        start,
        end,
        freq=pd.offsets.CustomBusinessDay(weekmask=weekmask),
    )
    if calendar == WEEKDAYS:
        return days
    closed = holidays.financial_holidays(calendar, years=range(start.year, end.year + 1))
    return days[~days.isin(pd.DatetimeIndex(list(closed.keys())))]


@register_indicator("turn_of_month")
class TurnOfMonthIndicator(BaseIndicator):
    """Trading day of the month, trading days left, and turn-of-month entry/exit flags."""

    name = "turn_of_month"
    version = "1.0.0"
    group = "calendar"
    benefits_from_stationary = False

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Append trading-day positions and turn-of-month entry and exit flags."""
        p = {**self.get_default_params(), **params}
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("turn_of_month needs a DatetimeIndex")
        out = pd.DataFrame(index=df.index, columns=TOM_COLUMNS, dtype=float)
        if len(df):
            dates = df.index.tz_localize(None) if df.index.tz is not None else df.index
            dates = dates.normalize()
            open_days = trading_days(p["calendar"], dates.min(), dates.max())
            positions = pd.Series(np.arange(len(open_days)), index=open_days.to_period("M"))
            by_month = positions.groupby(level=0)
            bar_months = dates.to_period("M")
            first = by_month.min().reindex(bar_months).to_numpy()
            last = by_month.max().reindex(bar_months).to_numpy()
            # last trading day on or before the bar's date (a bar on a closed day
            # counts the trading days before it)
            pos = open_days.searchsorted(dates, side="right") - 1
            out["tom_day"] = np.clip(pos - first + 1, 0, None).astype(float)
            out["tom_days_left"] = (last - pos).astype(float)
            out["tom_entry_long"] = (out["tom_days_left"] == p["entry_days_left"]).astype(float)
            out["tom_exit_long"] = (out["tom_day"] == p["exit_day"]).astype(float)
            out.iloc[0] = np.nan  # indicator contract: the first row is NaN
        return pd.concat([df, out], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        """Return the names of the indicator's feature columns."""
        return list(TOM_COLUMNS)

    def get_signal_columns(self, params=None) -> List[str]:
        """Return the entry and exit signal column names."""
        return ["tom_entry_long", "tom_exit_long"]

    @classmethod
    def get_default_params(cls) -> dict:
        """Return the default exchange calendar and turn-of-month window."""
        return {"calendar": "NYSE", "entry_days_left": 1, "exit_day": 3}

    @classmethod
    def get_param_schema(cls) -> dict:
        """Describe the supported calendar and window parameters."""
        return {
            "calendar": {
                "type": "choice",
                "choices": CALENDARS,
                "default": "NYSE",
                "description": "Exchange holiday calendar (holidays.financial_holidays) "
                "or 'weekdays' for every weekday",
            },
            "entry_days_left": {
                "type": "int",
                "default": 1,
                "min": 0,
                "max": 15,
                "description": "tom_entry_long marks the bar with this many "
                "trading days left in the month",
            },
            "exit_day": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 15,
                "description": "tom_exit_long marks this trading day of the month",
            },
        }


__all__ = ["TurnOfMonthIndicator", "TOM_COLUMNS", "trading_days"]
