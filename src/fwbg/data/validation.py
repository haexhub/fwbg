"""Data checks a run applies to its bars before any indicator sees them.

Errors are data a backtest must not run on: unordered or duplicate timestamps,
broken or non-finite prices, or so many bars outside the asset's trading hours
that the time stamps are wrong. Warnings may be real but deserve a look: isolated
moves that revert at once, outsized bar ranges, frozen prices, bars outside or
missing inside the trading hours.

The only change to the data is dropping filler bars, bars that record no trade
(flat and without volume, or flat outside the trading hours when the source has
no volume): Dukascopy writes them for weekends and holidays. Everything else is
reported, never altered: an extreme bar can be a real market event, and smoothing
it would flatter a backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .trading_hours import TradingHours

# Above this share of (non-filler) bars outside the trading hours the time stamps
# are wrong (time zone, bar labelling) rather than an early quote or holiday bar.
OUTSIDE_HOURS_ERROR_SHARE = 0.05
# A close-to-close move this many times the recent median absolute move that
# reverses by at least half on the next bar looks like a bad tick. Moves across a
# data gap (session open, weekend) are expected and skipped.
SPIKE_FACTOR = 20.0
SPIKE_REVERSAL = 0.5
# A bar range this many times the recent median range is reported.
RANGE_FACTOR = 15.0
SCALE_WINDOW = 100
# This many consecutive bars without any price change.
STALE_RUN_BARS = 5
# A single gap inside the trading hours longer than this many days of bars is a
# hole in the data, not a holiday.
LONG_GAP_DAYS = 5
EXAMPLES = 5


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    count: int
    message: str
    examples: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "count": self.count,
            "message": self.message,
            "examples": list(self.examples),
        }


@dataclass(frozen=True)
class DataCheckReport:
    symbol: str
    n_bars: int  # after dropping filler bars
    trading_hours: str | None  # time zone of the tradable session, None when unknown
    findings: tuple[Finding, ...] = ()

    def of(self, severity: Severity) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is severity)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return self.of(Severity.ERROR)

    def to_dict(self) -> dict:
        out: dict = {
            "symbol": self.symbol,
            "n_bars": self.n_bars,
            "trading_hours": self.trading_hours,
        }
        for severity in Severity:
            out[f"{severity.value}s"] = [f.to_dict() for f in self.of(severity)]
        return out

    def summary(self) -> str:
        parts = [f"{f.severity.value} {f.code} ({f.count}): {f.message}" for f in self.findings]
        return "; ".join(parts) if parts else "ok"


class DataCheckError(ValueError):
    """The bars failed a data check that blocks the run."""

    def __init__(self, report: DataCheckReport) -> None:
        super().__init__(f"{report.symbol}: data checks failed: {report.summary()}")
        self.report = report


def _gaps(index: pd.DatetimeIndex) -> np.ndarray:
    """Time from each bar to the next (the index may be stored in any unit)."""
    return np.diff(index.values).astype("timedelta64[ns]")


def _examples(index: pd.DatetimeIndex, mask: np.ndarray) -> tuple[str, ...]:
    return tuple(ts.isoformat() for ts in index[mask][:EXAMPLES])


def _flat(df: pd.DataFrame) -> np.ndarray:
    o, h, low, c = (df[k].to_numpy() for k in ("O", "H", "L", "C"))
    return np.asarray((o == h) & (h == low) & (low == c), dtype=bool)


def _structure(df: pd.DataFrame) -> list[Finding]:
    idx = df.index
    out = []
    backwards = np.concatenate([[False], _gaps(idx) < np.timedelta64(0, "ns")])
    duplicate = np.asarray(idx.duplicated())
    if backwards.any():
        out.append(
            Finding(
                "unordered_timestamps",
                Severity.ERROR,
                int(backwards.sum()),
                "timestamps go backwards",
                _examples(idx, backwards),
            )
        )
    if duplicate.any():
        out.append(
            Finding(
                "duplicate_timestamps",
                Severity.ERROR,
                int(duplicate.sum()),
                "timestamps occur more than once",
                _examples(idx, duplicate),
            )
        )
    o, h, low, c = (df[k].to_numpy(dtype=float) for k in ("O", "H", "L", "C"))
    finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(low) & np.isfinite(c)
    with np.errstate(invalid="ignore"):
        bad_value = ~finite | (np.fmin.reduce([o, h, low, c]) <= 0)
        tol = 1e-9 * np.abs(c)
        broken = ~bad_value & ((low > np.minimum(o, c) + tol) | (h < np.maximum(o, c) - tol))
    if bad_value.any():
        out.append(
            Finding(
                "invalid_prices",
                Severity.ERROR,
                int(bad_value.sum()),
                "non-finite or non-positive prices",
                _examples(idx, bad_value),
            )
        )
    if broken.any():
        out.append(
            Finding(
                "ohlc_inconsistent",
                Severity.ERROR,
                int(broken.sum()),
                "low above open/close or high below open/close",
                _examples(idx, broken),
            )
        )
    return out


def _missing_bars(
    index: pd.DatetimeIndex, hours: TradingHours, bar_minutes: int
) -> tuple[int, int, pd.Timestamp | None]:
    """Bars absent although the session is open: total, largest gap and where it starts."""
    step = pd.Timedelta(minutes=bar_minutes)
    total, largest, where = 0, 0, None
    for i in np.flatnonzero(_gaps(index) > step.to_timedelta64()):
        expected = pd.date_range(index[i] + step, index[i + 1] - step, freq=step)
        # a bar counts as missing when the session is open for at least half of it
        n = (
            int((2 * hours.open_minutes(expected, bar_minutes) >= bar_minutes).sum())
            if len(expected)
            else 0
        )
        total += n
        if n > largest:
            largest, where = n, index[i]
    return total, largest, where


def _hours(df: pd.DataFrame, hours: TradingHours, bar_minutes: int) -> list[Finding]:
    outside = ~hours.overlaps(df.index, bar_minutes)
    out = []
    if outside.any():
        share = outside.mean()
        severity = Severity.ERROR if share > OUTSIDE_HOURS_ERROR_SHARE else Severity.WARNING
        out.append(
            Finding(
                "outside_trading_hours",
                severity,
                int(outside.sum()),
                f"{share:.1%} of bars lie outside the trading hours ({hours.tz})",
                _examples(df.index, outside),
            )
        )
    missing, largest, where = _missing_bars(df.index, hours, bar_minutes)
    if missing:
        out.append(
            Finding(
                "missing_bars",
                Severity.INFO,
                missing,
                f"bars missing inside the trading hours, holidays included; largest gap {largest} bars",
            )
        )
    if where is not None and largest * bar_minutes > LONG_GAP_DAYS * 1440:
        out.append(
            Finding(
                "long_gap",
                Severity.WARNING,
                largest,
                f"{largest} bars missing inside the trading hours in one gap",
                (where.isoformat(),),
            )
        )
    return out


def _outliers(df: pd.DataFrame, bar_minutes: int | None) -> list[Finding]:
    idx = df.index
    c = df["C"].to_numpy(dtype=float)
    r = np.diff(np.log(c), prepend=np.nan)
    after_gap = np.zeros(len(df), dtype=bool)
    if bar_minutes:
        after_gap[1:] = _gaps(idx) > (1.5 * pd.Timedelta(minutes=bar_minutes)).to_timedelta64()
    scale = pd.Series(np.abs(r)).rolling(SCALE_WINDOW, min_periods=20).median().shift(1).to_numpy()
    nxt = np.append(r[1:], np.nan)
    rng = np.log(df["H"].to_numpy(dtype=float) / df["L"].to_numpy(dtype=float))
    rng_scale = pd.Series(rng).rolling(SCALE_WINDOW, min_periods=20).median().shift(1).to_numpy()
    with np.errstate(invalid="ignore"):
        spike = (
            (np.abs(r) > SPIKE_FACTOR * scale)
            & (r * nxt < 0)
            & (np.abs(nxt) >= SPIKE_REVERSAL * np.abs(r))
        )
        spike &= ~after_gap
        wide = (rng > RANGE_FACTOR * rng_scale) & (rng_scale > 0) & ~after_gap
    out = []
    if spike.any():
        out.append(
            Finding(
                "price_spikes",
                Severity.WARNING,
                int(spike.sum()),
                f"moves above {SPIKE_FACTOR:g}x the recent median that reverse on the next bar "
                "(bad ticks or real shocks)",
                _examples(idx, spike),
            )
        )
    if wide.any():
        out.append(
            Finding(
                "wide_bars",
                Severity.WARNING,
                int(wide.sum()),
                f"bar ranges above {RANGE_FACTOR:g}x the recent median",
                _examples(idx, wide),
            )
        )
    frozen = _flat(df) & (np.append(np.nan, c[:-1]) == c)
    runs = _run_starts(frozen, STALE_RUN_BARS)
    if runs.any():
        out.append(
            Finding(
                "stale_prices",
                Severity.WARNING,
                int(runs.sum()),
                f"runs of {STALE_RUN_BARS}+ bars without a price change",
                _examples(idx, runs),
            )
        )
    return out


def _run_starts(flag: np.ndarray, length: int) -> np.ndarray:
    """Mark the first bar of each run of at least *length* consecutive True values."""
    starts = np.zeros(len(flag), dtype=bool)
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        if run == length:
            starts[i - length + 1] = True
    return starts


def _filler(df: pd.DataFrame, hours: TradingHours | None, bar_minutes: int | None) -> np.ndarray:
    flat = _flat(df)
    volume = df["V"].to_numpy(dtype=float) if "V" in df else np.zeros(len(df))
    if (volume > 0).any():
        return np.asarray(flat & (volume == 0), dtype=bool)
    if hours is None or not bar_minutes:
        return np.zeros(len(df), dtype=bool)
    return np.asarray(flat & ~hours.overlaps(df.index, bar_minutes), dtype=bool)


def check_bars(
    df: pd.DataFrame, *, symbol: str, bar_minutes: int | None, hours: TradingHours | None
) -> tuple[pd.DataFrame, DataCheckReport]:
    """Drop filler bars from OHLC bars of one symbol and check the rest.

    *df* has a DatetimeIndex (naive = UTC) and columns O/H/L/C (V optional).
    *bar_minutes* is the bar length; with it and the asset's tradable *hours* the
    bars are checked against the trading hours too. Pure: no I/O.
    """
    findings = _structure(df)
    if any(f.severity is Severity.ERROR for f in findings):
        return df, DataCheckReport(symbol, len(df), hours.tz if hours else None, tuple(findings))
    filler = _filler(df, hours, bar_minutes)
    if filler.any():
        findings.append(
            Finding(
                "filler_bars_dropped",
                Severity.INFO,
                int(filler.sum()),
                "flat bars without trading (weekends, holidays) removed",
                _examples(df.index, filler),
            )
        )
        df = df[~filler]
    if hours is not None and bar_minutes:
        findings += _hours(df, hours, bar_minutes)
    findings += _outliers(df, bar_minutes)
    return df, DataCheckReport(symbol, len(df), hours.tz if hours else None, tuple(findings))
