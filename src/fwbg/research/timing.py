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


def monthly_position(condition: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Return the 0/1 position that earns each day's return under a monthly rule.

    ``condition`` is True where the rule says to be invested; only its last value per
    calendar month counts, taken at the month end. ``index`` is the daily session index
    of the traded series. Days before the first decision are not invested.

    Multiply the result with the same day's return: ``position * returns``. The value
    on day ``d`` was decided on or before the close of the session before ``d``.
    """
    if not isinstance(condition.index, pd.DatetimeIndex) or not isinstance(
        index, pd.DatetimeIndex
    ):
        raise ValueError("condition and index must be DatetimeIndex-based")
    mask = exposure_mask(condition.dropna().astype(bool), index)
    return mask.shift(1).fillna(False).astype(float)
