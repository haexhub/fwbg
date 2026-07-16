"""Pure data-quality checks for downloaded OHLCV bar frames.

No I/O — callers own reading/writing; this module only inspects an in-memory
frame and returns a JSON-serializable report dict. Warn-only: no thresholds
here ever block or alter a download, they only populate ``warnings``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Timeframe key -> expected spacing between consecutive bars, in seconds.
# Mirrors fwbg.data.dukascopy.TIMEFRAMES; kept independent here so this module
# has no import-time dependency on the (I/O-heavy) downloader module.
_TIMEFRAME_SECONDS: dict[str, int] = {
    "MINUTE_1": 60,
    "MINUTE_5": 5 * 60,
    "MINUTE_15": 15 * 60,
    "MINUTE_30": 30 * 60,
    "HOUR_1": 60 * 60,
    "HOUR_4": 4 * 60 * 60,
    "DAY_1": 24 * 60 * 60,
}

# A consecutive-bar gap longer than this multiple of the expected spacing is
# flagged (subject to the weekend exclusion below).
_GAP_MULTIPLIER = 1.5

# Coverage below this ratio of (last-first)/(requested range) is flagged.
_MIN_COVERAGE = 0.95

# Forex closes Friday evening and reopens Sunday evening (UTC); the exact hour
# shifts by ~1h with US/EU DST. These bounds are intentionally generous (start
# earlier, end later than the documented 21/22:00 window) so real weekend
# closures are never misclassified as data gaps.
_WEEKEND_GAP_START_SEC = 4 * 86400 + 20 * 3600  # Friday 20:00 UTC
_WEEKEND_GAP_END_SEC = 6 * 86400 + 23 * 3600  # Sunday 23:00 UTC
# Sanity bound: a real weekend closure is ~2 days. Without this, a gap that
# merely *starts* near a Friday and *ends* near a Sunday of a *later* week
# (e.g. a full missing week) would alias into the time-of-week check above.
_WEEKEND_GAP_MAX_SECONDS = 3 * 86400


def _seconds_into_week(ts: pd.Timestamp) -> int:
    """Seconds since Monday 00:00 UTC of *ts*'s week."""
    return ts.weekday() * 86400 + ts.hour * 3600 + ts.minute * 60 + ts.second


def _is_weekend_gap(t1: pd.Timestamp, t2: pd.Timestamp) -> bool:
    """True if the gap (t1, t2) is expected forex weekend closure, per the
    heuristic in the module docstring above."""
    if (t2 - t1).total_seconds() > _WEEKEND_GAP_MAX_SECONDS:
        return False
    return (
        _seconds_into_week(t1) >= _WEEKEND_GAP_START_SEC
        and _seconds_into_week(t2) <= _WEEKEND_GAP_END_SEC
    )


def _as_utc_index(df: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    """Normalize either a ``T,O,H,L,C,V`` frame or a ``DatetimeIndex`` +
    ``O,H,L,C,V`` frame into a UTC ``DatetimeIndex`` plus the OHLCV arrays."""
    if "T" in df.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(df["T"], utc=True))
    else:
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    cols = {name: df[name].to_numpy(dtype=float) for name in ("O", "H", "L", "C", "V")}
    return idx, cols


def _to_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def assess_bars(
    df: pd.DataFrame,
    *,
    timeframe: str,
    requested_start: datetime,
    requested_end: datetime,
) -> dict:
    """Quality report for a mid-OHLC bar frame (columns ``T,O,H,L,C,V`` or a
    ``DatetimeIndex`` + ``O,H,L,C,V`` columns). Pure — no I/O.
    """
    if timeframe not in _TIMEFRAME_SECONDS:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    expected_spacing = _TIMEFRAME_SECONDS[timeframe]

    idx, cols = _as_utc_index(df)
    n_bars = len(idx)
    requested_start = _to_utc(requested_start)
    requested_end = _to_utc(requested_end)

    if n_bars == 0:
        return {
            "n_bars": 0,
            "first_bar": None,
            "last_bar": None,
            "coverage": 0.0,
            "expected_spacing_seconds": expected_spacing,
            "n_gaps": 0,
            "max_gap_seconds": 0.0,
            "weekend_gaps_ignored": 0,
            "non_monotonic": 0,
            "duplicate_timestamps": 0,
            "zero_volume_ratio": 0.0,
            "ohlc_violations": 0,
            "nan_bars": 0,
            "warnings": ["no bars"],
        }

    first_bar = idx[0]
    last_bar = idx[-1]

    requested_span = (requested_end - requested_start).total_seconds()
    coverage = (
        (last_bar - first_bar).total_seconds() / requested_span if requested_span > 0 else 0.0
    )
    coverage = min(max(coverage, 0.0), 1.0)

    # Seconds between consecutive bars. Divides by a timedelta unit rather than
    # assuming a fixed integer resolution (pandas' datetime64 storage
    # resolution is not guaranteed to be nanoseconds).
    deltas = np.diff(idx.values) / np.timedelta64(1, "s")
    non_monotonic = int(np.sum(deltas <= 0))
    duplicate_timestamps = int(np.sum(deltas == 0))

    gap_threshold = _GAP_MULTIPLIER * expected_spacing
    n_gaps = 0
    weekend_gaps_ignored = 0
    max_gap_seconds = 0.0
    for i in np.where(deltas > gap_threshold)[0]:
        t1, t2 = idx[i], idx[i + 1]
        if _is_weekend_gap(t1, t2):
            weekend_gaps_ignored += 1
        else:
            n_gaps += 1
            max_gap_seconds = max(max_gap_seconds, float(deltas[i]))

    o, h, low, c, v = cols["O"], cols["H"], cols["L"], cols["C"], cols["V"]
    finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(low) & np.isfinite(c)
    nan_bars = int(np.sum(~finite))

    lo = np.minimum(o, c)
    hi = np.maximum(o, c)
    ohlc_ok = (low <= lo) & (lo <= hi) & (hi <= h)
    ohlc_violations = int(np.sum(finite & ~ohlc_ok))

    # 1.0 means "volume unavailable" (the writer stores a 0 scalar when either
    # side lacks volume data), not necessarily "market inactive" — forex
    # volume is a synthetic tick-count proxy, not exchange volume. Report
    # only; deliberately no warning threshold (see module docstring).
    zero_volume_ratio = float(np.sum(v == 0) / n_bars)

    warnings: list[str] = []
    if coverage < _MIN_COVERAGE:
        warnings.append(f"coverage {coverage:.3f} below {_MIN_COVERAGE}")
    if n_gaps > 0:
        warnings.append(f"{n_gaps} gap(s) in bar sequence, max {max_gap_seconds:.0f}s")
    if non_monotonic > 0:
        warnings.append(f"{non_monotonic} non-monotonic timestamp(s)")
    if duplicate_timestamps > 0:
        warnings.append(f"{duplicate_timestamps} duplicate timestamp(s)")
    if ohlc_violations > 0:
        warnings.append(f"{ohlc_violations} OHLC violation(s) (L<=min(O,C)<=max(O,C)<=H broken)")
    if nan_bars > 0:
        warnings.append(f"{nan_bars} bar(s) with non-finite OHLC")

    return {
        "n_bars": n_bars,
        "first_bar": first_bar.isoformat(),
        "last_bar": last_bar.isoformat(),
        "coverage": coverage,
        "expected_spacing_seconds": expected_spacing,
        "n_gaps": n_gaps,
        "max_gap_seconds": max_gap_seconds,
        "weekend_gaps_ignored": weekend_gaps_ignored,
        "non_monotonic": non_monotonic,
        "duplicate_timestamps": duplicate_timestamps,
        "zero_volume_ratio": zero_volume_ratio,
        "ohlc_violations": ohlc_violations,
        "nan_bars": nan_bars,
        "warnings": warnings,
    }
