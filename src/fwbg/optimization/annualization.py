"""Bars per calendar year, derived from the data instead of assumed from the timeframe."""

import numpy as np
import pandas as pd

_SECONDS_PER_YEAR = 365.25 * 86400.0


def bars_per_year(index: pd.Index, fallback: float | None = None) -> float:
    """Bars per calendar year observed in *index* (a DatetimeIndex).

    median bars per trading day (a calendar date with at least one bar) times the
    trading days per calendar year observed (distinct dates over the index's span).
    A 7-bar/day instrument that trades ~252 days gives ~1764, a 24/7 hourly one
    8760. Assuming ``bars_per_hour * 24 * 250`` instead gives 6000 for any hourly
    series and inflates the annualised Sharpe by sqrt(6000/1764) = 1.84 for the
    former.

    Falls back to *fallback* when the index is not a datetime index, has under two
    dates or no span; raises ValueError then if there is no fallback.
    """
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 2:
        span_years = (index[-1] - index[0]).total_seconds() / _SECONDS_PER_YEAR
        days = pd.Series(1, index=index.normalize()).groupby(level=0).sum()
        if span_years > 0 and len(days) >= 2:
            trading_days_per_year = len(days) / span_years
            return float(np.median(days.to_numpy()) * trading_days_per_year)
    if fallback is None:
        raise ValueError("bars per year needs a DatetimeIndex with at least two dates")
    return float(fallback)
