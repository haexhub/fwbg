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
  merge into one; NaN, like the two columns below, after the file's last event
  (``evt_days_ahead`` is NaN there), because later events are unknown, not absent
- ``evt_window_start``: 1 on the bar where ``evt_window`` turns on (an entry column)
- ``evt_window_end``: 1 on the bar where ``evt_window`` turns off (an exit column
  for ``signal_exit``)

Holding from ``evt_window_start`` to ``evt_window_end`` with ``window_start=6`` and
``window_end=2`` is exposed to the returns of the five trading days before each event.
"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator

log = logging.getLogger(__name__)

EVENT_COLUMNS = ["evt_days_ahead", "evt_window", "evt_window_start", "evt_window_end"]


def _trading_days(
    calendar: str, first: pd.Timestamp, last: pd.Timestamp
) -> pd.DatetimeIndex:
    from fwbg.plugins import import_plugin_module

    tom = import_plugin_module("fwbg-core", "indicators", "turn_of_month")
    return tom.trading_days(calendar, first, last)


def _calendars() -> list:
    from fwbg.plugins import import_plugin_module

    return list(
        import_plugin_module("fwbg-core", "indicators", "turn_of_month").CALENDARS
    )


@lru_cache(maxsize=16)
def _read_events(path: str, mtime: float) -> pd.DataFrame:
    """The events file as strings (cached per path and modification time)."""
    del mtime  # part of the cache key only
    return pd.read_csv(path, dtype=str)


class EventDateError(ValueError):
    """A non-blank cell of the date column is not a date."""


def _parse_dates(values: pd.Series, source: str) -> pd.DatetimeIndex:
    """Naive normalized dates of *values*; each cell is parsed on its own, so rows
    may mix formats (``2024-01-10``, ``2024-01-11 14:00``, ``Jan 12, 2024``). A
    timezone offset is dropped, keeping the date as written. Blank cells are
    skipped, any other unparsable cell raises ``EventDateError``. Ambiguous
    numeric dates such as ``01/02/2024`` are read month first; prefer ISO dates."""
    cells = values.dropna().astype(str).str.strip()
    cells = cells[cells != ""]
    parsed: dict[str, pd.Timestamp] = {}
    bad: list[str] = []
    for cell in cells.unique():
        try:
            # a cell without a four-digit year would silently get the current one
            stamp = pd.Timestamp(cell) if re.search(r"\d{4}", cell) else pd.NaT
        except (ValueError, OverflowError):
            stamp = pd.NaT
        if pd.isna(stamp):
            bad.append(cell)
            continue
        parsed[cell] = (stamp.tz_localize(None) if stamp.tzinfo else stamp).normalize()
    if bad:
        shown = ", ".join(repr(b) for b in bad[:5])
        raise EventDateError(
            f"scheduled_events: {len(bad)} unparsable date value(s) in {source}: {shown}"
        )
    return pd.DatetimeIndex(sorted(set(parsed.values())), dtype="datetime64[ns]")


def event_dates(
    path: str, date_column: str, filter_column: str = "", filter_values: tuple = ()
) -> pd.DatetimeIndex:
    """Sorted unique event dates of *path*, optionally only rows whose
    *filter_column* is one of *filter_values*. Raises ``ValueError`` on a missing
    column, ``EventDateError`` (a ``ValueError``) on an unparsable date and
    ``OSError`` on an unreadable file."""
    file = Path(path).expanduser()
    events = _read_events(str(file), file.stat().st_mtime)
    if not date_column or date_column not in events.columns:
        raise ValueError(f"scheduled_events: column {date_column!r} not in {file}")
    if filter_column and filter_column not in events.columns:
        raise ValueError(f"scheduled_events: column {filter_column!r} not in {file}")
    if filter_column:
        events = events[events[filter_column].isin(list(filter_values))]
    return _parse_dates(events[date_column], f"{file} column {date_column!r}")


@register_indicator("scheduled_events")
class ScheduledEventsIndicator(BaseIndicator):
    """Trading days to the next scheduled event and a window flag before events."""

    name = "scheduled_events"
    version = "1.1.0"
    group = "calendar"
    benefits_from_stationary = False

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Append days-to-event and the event window columns."""
        p = {**self.get_default_params(), **params}
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("scheduled_events needs a DatetimeIndex")
        if p["window_end"] > p["window_start"]:
            raise ValueError(
                "scheduled_events: window_end must not exceed window_start"
            )
        out = pd.DataFrame(index=df.index, columns=EVENT_COLUMNS, dtype=float)
        events = self._events(p)
        if len(df) and events is not None:
            dates = df.index.tz_localize(None) if df.index.tz is not None else df.index
            dates = dates.normalize()
            last = max(dates.max(), events.max()) if len(events) else dates.max()
            # Include a following month so a closed date at month end (for
            # example, a Sunday event) can map to its next trading day.
            last += pd.offsets.MonthEnd(1)
            prior_month_end = dates.min().replace(day=1) - pd.Timedelta(days=1)
            days = _trading_days(p["calendar"], prior_month_end, last)
            # an event or a bar on a closed day counts from the next trading day
            event_pos = np.unique(
                days.searchsorted(events[events >= days[0]], side="left")
            )
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
            # After the file's last event the future is unknown: not "no window"
            known = has_next
            out["evt_days_ahead"] = ahead
            out["evt_window"] = np.where(known, in_window, np.nan)
            out["evt_window_start"] = np.where(known, in_window & ~before, np.nan)
            out["evt_window_end"] = np.where(known, ~in_window & before, np.nan)
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
                p["events_file"],
                p["date_column"],
                p["filter_column"],
                tuple(p["filter_values"]),
            )
        except EventDateError:
            raise
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


__all__ = ["ScheduledEventsIndicator", "EVENT_COLUMNS", "EventDateError", "event_dates"]
