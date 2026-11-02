"""Scheduled events Indicator Plugin.

Reads dated events from a CSV file (for example Treasury auctions, central bank
meetings or index rebalancing dates) and counts exchange trading days from each bar
to the next event, with a window flag for holding a position a set number of trading
days before the events.

The file needs a date column; a filter column and values select a subset of its rows.
Trading days follow ``turn_of_month``'s exchange calendar, so the counts do not
depend on the bars in the data and also cover events after the last bar. The columns
are unshifted and meant for a decision at the bar's close (``entry_delay=0``): the
file must only hold events that were scheduled in advance, which is what makes the
counts free of lookahead. As for every indicator, the first row is NaN. Bars are
expected to be daily bars labelled with their trading date.

Columns:

- ``evt_days_ahead``: trading days from the bar's date to the next event on or
  after it, 0 on an event day; NaN when the file holds no later event
- ``evt_window``: 1 when an event lies between ``window_end`` and ``window_start``
  trading days ahead (inclusive), else 0; overlapping windows of clustered events
  merge into one
- ``evt_window_start``: 1 on the bar where ``evt_window`` turns on (an entry column)
- ``evt_window_end``: 1 on the bar where ``evt_window`` turns off (an exit column
  for ``signal_exit``)

Holding from ``evt_window_start`` to ``evt_window_end`` with ``window_start=6`` and
``window_end=2`` is exposed to the returns of the five trading days before each event.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator

log = logging.getLogger(__name__)

EVENT_COLUMNS = ["evt_days_ahead", "evt_window", "evt_window_start", "evt_window_end"]


def _trading_days(calendar: str, first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    from fwbg.plugins import import_plugin_module

    tom = import_plugin_module("fwbg-core", "indicators", "turn_of_month")
    return tom.trading_days(calendar, first, last)


def _calendars() -> list:
    from fwbg.plugins import import_plugin_module

    return list(import_plugin_module("fwbg-core", "indicators", "turn_of_month").CALENDARS)


@lru_cache(maxsize=16)
def _read_events(path: str, mtime: float) -> pd.DataFrame:
    """The events file as strings (cached per path and modification time)."""
    del mtime  # part of the cache key only
    return pd.read_csv(path, dtype=str)


def event_dates(
    path: str, date_column: str, filter_column: str = "", filter_values: tuple = ()
) -> pd.DatetimeIndex:
    """Sorted unique event dates of *path*, optionally only rows whose
    *filter_column* is one of *filter_values*. Raises ``ValueError`` on a missing
    column and ``OSError`` on an unreadable file."""
    file = Path(path).expanduser()
    events = _read_events(str(file), file.stat().st_mtime)
    for column in (date_column, filter_column):
        if column and column not in events.columns:
            raise ValueError(f"scheduled_events: column {column!r} not in {file}")
    if filter_column:
        events = events[events[filter_column].isin(list(filter_values))]
    dates = pd.to_datetime(events[date_column], errors="coerce").dropna()
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    return pd.DatetimeIndex(dates.dt.normalize().unique()).sort_values()


@register_indicator("scheduled_events")
class ScheduledEventsIndicator(BaseIndicator):
    """Trading days to the next scheduled event and a window flag before events."""

    name = "scheduled_events"
    version = "1.0.0"
    group = "calendar"
    benefits_from_stationary = False

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Append days-to-event and the event window columns."""
        p = {**self.get_default_params(), **params}
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("scheduled_events needs a DatetimeIndex")
        if p["window_end"] > p["window_start"]:
            raise ValueError("scheduled_events: window_end must not exceed window_start")
        out = pd.DataFrame(index=df.index, columns=EVENT_COLUMNS, dtype=float)
        events = self._events(p)
        if len(df) and events is not None:
            dates = df.index.tz_localize(None) if df.index.tz is not None else df.index
            dates = dates.normalize()
            last = max(dates.max(), events.max()) if len(events) else dates.max()
            # Include a following month so a closed date at month end (for
            # example, a Sunday event) can map to its next trading day.
            last += pd.offsets.MonthEnd(1)
            days = _trading_days(p["calendar"], dates.min(), last)
            # an event or a bar on a closed day counts from the next trading day
            event_pos = np.unique(days.searchsorted(events[events >= days[0]], side="left"))
            event_pos = event_pos[event_pos < len(days)]
            bar_pos = days.searchsorted(dates, side="left")
            nxt = event_pos.searchsorted(bar_pos, side="left")
            ahead = np.full(len(bar_pos), np.nan)
            has_next = nxt < len(event_pos)
            ahead[has_next] = event_pos[nxt[has_next]] - bar_pos[has_next]
            first = event_pos.searchsorted(bar_pos + p["window_end"], side="left")
            in_window = np.zeros(len(bar_pos), dtype=bool)
            ok = first < len(event_pos)
            in_window[ok] = event_pos[first[ok]] <= bar_pos[ok] + p["window_start"]
            before = np.concatenate([[False], in_window[:-1]])
            out["evt_days_ahead"] = ahead
            out["evt_window"] = in_window.astype(float)
            out["evt_window_start"] = (in_window & ~before).astype(float)
            out["evt_window_end"] = (~in_window & before).astype(float)
        if len(out):
            out.iloc[0] = np.nan  # indicator contract: the first row is NaN
        return pd.concat([df, out], axis=1)

    @staticmethod
    def _events(p: dict):
        if not p["events_file"]:
            log.warning("scheduled_events: no events_file set; all values NaN")
            return None
        try:
            return event_dates(
                p["events_file"], p["date_column"], p["filter_column"], tuple(p["filter_values"])
            )
        except (OSError, ValueError) as exc:
            log.warning("scheduled_events: %s; all values NaN", exc)
            return None

    def get_feature_columns(self, params=None) -> List[str]:
        """Return the names of the indicator's feature columns."""
        return list(EVENT_COLUMNS)

    def get_signal_columns(self, params=None) -> List[str]:
        """Return the entry and exit signal column names."""
        return ["evt_window_start", "evt_window_end"]

    @classmethod
    def get_default_params(cls) -> dict:
        """Return the default parameters."""
        return {
            "events_file": "",
            "date_column": "date",
            "filter_column": "",
            "filter_values": [],
            "calendar": "NYSE",
            "window_start": 6,
            "window_end": 2,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        """Describe the events file, calendar and window parameters."""
        return {
            "events_file": {
                "type": "string",
                "default": "",
                "description": "CSV file with one dated event per row (~ is expanded)",
            },
            "date_column": {
                "type": "string",
                "default": "date",
                "description": "Column of the events file holding the event date",
            },
            "filter_column": {
                "type": "string",
                "default": "",
                "description": "Optional column whose value selects the events",
            },
            "filter_values": {
                "type": "list[string]",
                "default": [],
                "description": "Values of filter_column that count as events",
            },
            "calendar": {
                "type": "choice",
                "choices": _calendars(),
                "default": "NYSE",
                "description": "Exchange holiday calendar for counting trading days",
            },
            "window_start": {
                "type": "int",
                "default": 6,
                "min": 0,
                "max": 60,
                "description": "evt_window is on when an event is at most this many trading days ahead",
            },
            "window_end": {
                "type": "int",
                "default": 2,
                "min": 0,
                "max": 60,
                "description": "evt_window is on when an event is at least this many trading days ahead",
            },
        }


__all__ = ["ScheduledEventsIndicator", "EVENT_COLUMNS", "event_dates"]
