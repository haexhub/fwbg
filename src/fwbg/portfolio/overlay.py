"""Monthly exposure overlays driven by published macro series.

An overlay sets the whole book to cash for the next month when a condition on
already-published values holds. The condition is evaluated on the last completed
month end and applied from the next session's close, exactly like the allocation
signals in ``engine.run_portfolio``. No value is read before its publication date.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

OVERLAYS = ("fed_net_liquidity", "fed_balance_sheet", "us_japan_10y_spread")


@dataclass(frozen=True)
class OverlaySettings:
    name: str
    weeks: int = 13
    lag_days: int = 9
    threshold: float = 0.0

    def validate(self) -> None:
        """Reject unknown overlays, non-positive windows and non-finite thresholds."""
        if self.name not in OVERLAYS:
            raise ValueError(f"unknown overlay: {self.name}")
        if type(self.weeks) is not int or self.weeks < 1:
            raise ValueError("weeks must be a positive integer")
        if type(self.lag_days) is not int or self.lag_days < 0:
            raise ValueError("lag_days must be a non-negative integer")
        if not pd.notna(self.threshold):
            raise ValueError("threshold must be a finite number")


def read_series(path: Path, lag_days: int) -> pd.Series:
    """Read a two-column date/value CSV and shift each value to its publication date."""
    frame = pd.read_csv(path, parse_dates=["observation_date"], index_col="observation_date")
    series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()
    series.index = series.index + pd.Timedelta(days=lag_days)
    return series[~series.index.duplicated(keep="last")].sort_index()


def overlay_condition(settings: OverlaySettings, data_directory: Path) -> pd.Series:
    """Return the monthly condition: True means the book stays invested.

    ``fed_net_liquidity`` uses WALCL - WTREGEN - RRPONTSYD, ``fed_balance_sheet``
    WALCL alone, and ``us_japan_10y_spread`` the US 10-year yield minus the OECD
    Japan 10-year yield. The statistic is the change over ``weeks`` weeks; the
    condition holds while that change is above ``threshold``.
    """
    settings.validate()
    if settings.name == "us_japan_10y_spread":
        us = read_series(data_directory / "DGS10.csv", settings.lag_days).resample("ME").last()
        jp = read_series(data_directory / "IRLTLT01JPM156N.csv", 60).resample("ME").last()
        level = (us - jp).dropna()
    else:
        walcl = read_series(data_directory / "WALCL.csv", settings.lag_days)
        if settings.name == "fed_balance_sheet":
            level = walcl
        else:
            tga = read_series(data_directory / "WTREGEN.csv", 3)
            rrp = read_series(data_directory / "RRPONTSYD.csv", 1)
            index = walcl.index.union(tga.index).union(rrp.index)
            level = (walcl.reindex(index).ffill() - tga.reindex(index).ffill()
                     - rrp.reindex(index).ffill()).dropna()
    statistic = level.diff(settings.weeks)
    return (statistic > settings.threshold).dropna()


def exposure_mask(condition: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Map a monthly condition to a daily 0/1 exposure for the following month.

    A month-end value takes effect on the first session after that month end, so
    the month-end session itself still uses the previous month's decision.
    """
    monthly = condition.resample("ME").last()
    effective = monthly.copy()
    effective.index = monthly.index + pd.offsets.Day(1)
    mask = effective.reindex(index, method="ffill")
    return mask.fillna(False).astype(bool)
