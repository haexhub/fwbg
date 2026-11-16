"""Monthly exposure overlays driven by published macro series.

An overlay sets the whole book to cash for the next month when a condition on
already-published values is false. The condition is evaluated on the last
completed month end and applied from the next session's close, exactly like the
allocation signals in ``engine.run_portfolio``. No value is read before its
publication date.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

OVERLAYS = ("fed_net_liquidity", "fed_balance_sheet", "us_japan_10y_spread")

# Input files per overlay, with the smallest publication lag (days) each series
# can have. ``lag_days`` is applied to every series as a lower bound, so a larger
# configured lag always delays every component at least that much.
COMPONENTS: dict[str, dict[str, int]] = {
    "fed_net_liquidity": {"WALCL.csv": 0, "WTREGEN.csv": 0, "RRPONTSYD.csv": 0},
    "fed_balance_sheet": {"WALCL.csv": 0},
    "us_japan_10y_spread": {"DGS10.csv": 0, "IRLTLT01JPM156N.csv": 60},
}

# RRPONTSYD is published in billions of dollars, WALCL and WTREGEN in millions.
MILLIONS_PER_BILLION = 1000.0


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
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(self.threshold)
        ):
            raise ValueError("threshold must be a finite number")


def input_files(settings: OverlaySettings, data_directory: Path) -> list[Path]:
    """Return the CSV files the overlay reads, for provenance fingerprints."""
    return [data_directory / name for name in COMPONENTS[settings.name]]


def read_series(path: Path, lag_days: int) -> pd.Series:
    """Read a two-column date/value CSV and shift each value to its publication date."""
    frame = pd.read_csv(
        path, parse_dates=["observation_date"], index_col="observation_date"
    )
    series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()
    series.index = series.index + pd.Timedelta(days=lag_days)
    return series[~series.index.duplicated(keep="last")].sort_index()


def _component(settings: OverlaySettings, directory: Path, name: str) -> pd.Series:
    lag = max(settings.lag_days, COMPONENTS[settings.name][name])
    return read_series(directory / name, lag)


def _combine(parts: list[tuple[pd.Series, float]]) -> pd.Series:
    """Forward-fill every component on the union of publication dates and add them."""
    index = parts[0][0].index
    for series, _ in parts[1:]:
        index = index.union(series.index)
    level = sum(sign * series.reindex(index).ffill() for series, sign in parts)
    return level.dropna()


def change_over_weeks(level: pd.Series, weeks: int) -> pd.Series:
    """Change against the last value published ``weeks`` weeks (calendar time) earlier."""
    earlier = level.index - pd.Timedelta(weeks=weeks)
    past = level.asof(earlier)
    return (level - past.to_numpy()).dropna()


def overlay_condition(settings: OverlaySettings, data_directory: Path) -> pd.Series:
    """Return the condition on each publication date: True means the book stays invested.

    ``fed_net_liquidity`` uses WALCL - WTREGEN - RRPONTSYD (in millions of dollars;
    RRPONTSYD is converted from billions), ``fed_balance_sheet`` WALCL alone, and
    ``us_japan_10y_spread`` the US 10-year yield minus the OECD Japan 10-year
    yield. The statistic is the change over ``weeks`` calendar weeks; the
    condition holds while that change is above ``threshold``.
    """
    settings.validate()
    if settings.name == "us_japan_10y_spread":
        level = _combine(
            [
                (_component(settings, data_directory, "DGS10.csv"), 1.0),
                (_component(settings, data_directory, "IRLTLT01JPM156N.csv"), -1.0),
            ]
        )
    elif settings.name == "fed_balance_sheet":
        level = _component(settings, data_directory, "WALCL.csv")
    else:
        rrp = (
            _component(settings, data_directory, "RRPONTSYD.csv") * MILLIONS_PER_BILLION
        )
        level = _combine(
            [
                (_component(settings, data_directory, "WALCL.csv"), 1.0),
                (_component(settings, data_directory, "WTREGEN.csv"), -1.0),
                (rrp, -1.0),
            ]
        )
    return change_over_weeks(level, settings.weeks) > settings.threshold


def exposure_mask(condition: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Map a condition to a daily exposure that holds for the following month.

    The last condition of a month takes effect on the first session after that
    month end, so the month-end session itself still uses the previous month's
    decision. ``mask[date]`` is therefore the exposure to hold from the close of
    ``date``, which is the value a rebalance executing on ``date`` must read.
    Dates before the first condition are not invested.
    """
    monthly = condition.resample("ME").last().dropna()
    effective = monthly.copy()
    effective.index = monthly.index + pd.offsets.Day(1)
    mask = effective.reindex(index, method="ffill")
    return mask.fillna(False).astype(bool)
