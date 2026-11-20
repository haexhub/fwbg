"""Timing helpers for study scripts that evaluate monthly rules on daily returns.

A monthly rule is decided at a month-end close, takes effect at the close of the first
session after that month end and earns from the session after that. Build the daily
position with :func:`monthly_position` instead of shifting a month-end series: shifting
``month_end_series.shift(1)`` on the *monthly* index delays every decision by a whole
extra month, and ``Series.diff(n)`` on a merged weekly/daily index counts rows, not weeks
(use :func:`change_over_weeks`).
"""

import pandas as pd

from fwbg.portfolio.overlay import change_over_weeks, exposure_mask

__all__ = ["change_over_weeks", "exposure_mask", "monthly_position"]


def _wall_clock(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Drop the timezone but keep the local wall-clock time (and so the date)."""
    return index.tz_localize(None) if index.tz is not None else index


def monthly_position(condition: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Return the 0/1 position that earns each day's return under a monthly rule.

    ``condition`` is True where the rule says to be invested; only its last value per
    calendar month counts, taken at the month end. ``index`` is the daily session index
    of the traded series. Days before the first decision are not invested.

    Multiply the result with the same day's return: ``position * returns``. The value
    on day ``d`` was decided on or before the close of the session before ``d``.

    Pass the full session index and slice the result afterwards. For the first session
    the preceding session is not part of ``index``; it is taken to be the previous
    business day, so the position carries over a decision that executed before the
    window. That is exact unless that business day was an exchange holiday.

    Either argument may be tz-aware; the rule is evaluated on local calendar dates and
    the result carries ``index`` unchanged.
    """
    if not isinstance(condition.index, pd.DatetimeIndex) or not isinstance(
        index, pd.DatetimeIndex
    ):
        raise ValueError("condition and index must be DatetimeIndex-based")
    if index.empty:
        return pd.Series(dtype=float, index=index)
    # Month ends and sessions are calendar dates: compare them as local wall-clock
    # dates, so a tz-aware index (or condition) behaves like its naive counterpart.
    sessions = _wall_clock(index)
    decisions = condition.dropna().astype(bool)
    decisions.index = _wall_clock(decisions.index)
    previous = sessions[0] - pd.offsets.BDay(1)
    mask = exposure_mask(decisions, sessions.insert(0, previous))
    position = mask.shift(1).iloc[1:].fillna(False).astype(float)
    position.index = index
    return position
