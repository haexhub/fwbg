"""Noise Boundary Indicator Plugin.

Intraday momentum bands after Zarattini, Aziz and Barbon (2024, "Beat the
Market"). Within each cash session the band around the open is the average
absolute move from the open to the same time of day over the previous
sessions; price beyond the band marks a move larger than ordinary intraday
noise.

Columns (all unshifted: each value uses data up to the bar's close and is meant
for a decision at that close, e.g. ``entry_delay=0``):

- ``nb_upper`` / ``nb_lower``: max/min(session open, previous session close)
  times (1 +/- average move)
- ``nb_vwap``: session VWAP of the typical price (equal weight without volume)
- ``nb_long`` / ``nb_short``: 1 at check bars where the close is beyond the band
  (and beyond the VWAP when ``use_vwap``)
- ``nb_exit_long`` / ``nb_exit_short``: 1 at check bars where the close is back
  inside the trailing level (band, or band and VWAP), at the session's last bar
  and on every bar outside the session

The index is read as UTC (naive timestamps are UTC, as the loader provides them)
and converted to ``session_tz``; a bar is labelled by its start time.
"""
from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator

NB_COLUMNS = [
    "nb_upper", "nb_lower", "nb_vwap",
    "nb_long", "nb_short", "nb_exit_long", "nb_exit_short",
]
NB_SIGNAL_COLUMNS = ["nb_long", "nb_short", "nb_exit_long", "nb_exit_short"]


def _minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def _bar_length(index: pd.DatetimeIndex) -> pd.Timedelta:
    steps = pd.Series(index).diff().dropna()
    if steps.empty:
        raise ValueError("noise_boundary needs at least two bars to infer the bar length")
    return steps.mode().iloc[0]


@register_indicator("noise_boundary")
class NoiseBoundaryIndicator(BaseIndicator):
    """Time-of-day noise bands around the session open with entry and exit flags."""

    name = "noise_boundary"
    version = "1.0.0"
    group = "session"
    benefits_from_stationary = False

    def compute(
        self,
        df: pd.DataFrame,
        session_tz: str = "America/New_York",
        session_open: str = "09:30",
        session_close: str = "16:00",
        first_check: str = "10:00",
        last_check: str = "15:30",
        check_every_minutes: int = 30,
        lookback_sessions: int = 14,
        use_vwap: bool = True,
        **params,
    ) -> pd.DataFrame:
        if df.empty:
            return pd.concat([df, pd.DataFrame(index=df.index, columns=NB_COLUMNS, dtype=float)], axis=1)
        utc = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        bar = _bar_length(df.index)
        end = (utc + bar).tz_convert(session_tz)
        end_min = np.asarray(end.hour * 60 + end.minute)
        open_min, close_min = _minutes(session_open), _minutes(session_close)
        in_session = (end_min > open_min) & (end_min <= close_min) & np.asarray(end.dayofweek < 5)

        frame = pd.DataFrame({
            "day": np.asarray(end.tz_localize(None).normalize()),
            "tod": end_min,
            "O": df["O"].values, "H": df["H"].values, "L": df["L"].values, "C": df["C"].values,
            "V": df["V"].values if "V" in df.columns else np.zeros(len(df)),
        })
        session = frame[in_session]

        first_end = open_min + int(bar / pd.Timedelta(minutes=1))
        opening_bars = session[session["tod"] == first_end]
        session_open_px = opening_bars.groupby("day")["O"].first()
        closing_bars = session[session["tod"] == close_min]
        session_close_px = closing_bars.groupby("day")["C"].last()
        days = session_open_px.index
        # The last session close strictly before each day (a day may lack its own close bar).
        all_days = days.union(session_close_px.index)
        prev_close = session_close_px.reindex(all_days).ffill().shift(1).reindex(days)

        open_of_bar = session["day"].map(session_open_px)
        move = (session["C"] / open_of_bar - 1).abs()
        move_table = pd.DataFrame({"day": session["day"], "tod": session["tod"], "move": move})
        move_table = move_table.dropna().pivot_table(index="day", columns="tod", values="move", aggfunc="last")
        # Per time of day, the mean of its last observations before the session:
        # a session missing that bar (half day, dropped flat bar) is skipped, not
        # a gap that blanks the slot for the next lookback_sessions sessions.
        noise = move_table.apply(
            lambda slot: slot.dropna().rolling(lookback_sessions).mean().shift(1)
        )
        noise_long = noise.stack().rename("noise").reset_index()
        sigma = session[["day", "tod"]].merge(noise_long, on=["day", "tod"], how="left")["noise"].values

        hi_anchor = np.maximum(open_of_bar.values, session["day"].map(prev_close).values)
        lo_anchor = np.minimum(open_of_bar.values, session["day"].map(prev_close).values)
        upper = hi_anchor * (1 + sigma)
        lower = lo_anchor * (1 - sigma)

        typical = (session["H"] + session["L"] + session["C"]) / 3
        volume = session["V"].clip(lower=0)
        cum_v = volume.groupby(session["day"]).cumsum()
        cum_pv = (typical * volume).groupby(session["day"]).cumsum()
        equal_vwap = typical.groupby(session["day"]).cumsum() / typical.groupby(session["day"]).cumcount().add(1)
        vwap = np.where(cum_v > 0, cum_pv / cum_v.where(cum_v > 0, 1.0), equal_vwap)

        tod = session["tod"].values
        close = session["C"].values
        check = (
            (tod % check_every_minutes == 0)
            & (tod >= _minutes(first_check))
            & (tod <= _minutes(last_check))
            & np.isfinite(upper) & np.isfinite(lower)
        )
        above_vwap = (close > vwap) if use_vwap else np.ones(len(close), dtype=bool)
        below_vwap = (close < vwap) if use_vwap else np.ones(len(close), dtype=bool)
        long_trail = np.maximum(upper, vwap) if use_vwap else upper
        short_trail = np.minimum(lower, vwap) if use_vwap else lower
        last_bar = tod == close_min

        out = pd.DataFrame(np.nan, index=df.index, columns=NB_COLUMNS)
        rows = np.flatnonzero(in_session)
        out.iloc[rows, out.columns.get_loc("nb_upper")] = upper
        out.iloc[rows, out.columns.get_loc("nb_lower")] = lower
        out.iloc[rows, out.columns.get_loc("nb_vwap")] = vwap
        flags = {
            "nb_long": check & (close > upper) & above_vwap,
            "nb_short": check & (close < lower) & below_vwap,
            "nb_exit_long": (check & (close < long_trail)) | last_bar,
            "nb_exit_short": (check & (close > short_trail)) | last_bar,
        }
        for column, values in flags.items():
            out[column] = 1.0 if column.startswith("nb_exit") else 0.0  # outside the session: flat
            out.iloc[rows, out.columns.get_loc(column)] = values.astype(float)
        # Until the first band exists nothing can be entered, so there is nothing to
        # flag either: the warm-up stays NaN like any indicator without history.
        banded = np.flatnonzero(out["nb_upper"].notna().values)
        out.iloc[: banded[0] if len(banded) else len(out)] = np.nan
        return pd.concat([df, out], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        return list(NB_COLUMNS)

    def get_signal_columns(self, params=None) -> List[str]:
        return list(NB_SIGNAL_COLUMNS)

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "session_tz": "America/New_York",
            "session_open": "09:30",
            "session_close": "16:00",
            "first_check": "10:00",
            "last_check": "15:30",
            "check_every_minutes": 30,
            "lookback_sessions": 14,
            "use_vwap": True,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "session_tz": {"type": "string", "default": "America/New_York",
                           "description": "IANA time zone of the session clock times"},
            "session_open": {"type": "string", "default": "09:30", "description": "Session open (HH:MM, session_tz)"},
            "session_close": {"type": "string", "default": "16:00", "description": "Session close (HH:MM, session_tz)"},
            "first_check": {"type": "string", "default": "10:00",
                            "description": "First bar close (HH:MM) at which entries and trailing exits are checked"},
            "last_check": {"type": "string", "default": "15:30",
                           "description": "Last bar close (HH:MM) at which entries and trailing exits are checked"},
            "check_every_minutes": {"type": "int", "default": 30, "min": 5, "max": 120, "step": 5,
                                    "description": "Checks happen at bar closes whose minute of day is a multiple of this"},
            "lookback_sessions": {"type": "int", "default": 14, "min": 2, "max": 100, "step": 1,
                                  "description": "Sessions averaged for the time-of-day noise band"},
            "use_vwap": {"type": "bool", "default": True,
                         "description": "Require the close beyond the session VWAP for entries and trail at band/VWAP"},
        }


__all__ = ["NoiseBoundaryIndicator", "NB_COLUMNS"]
