"""Tests for fwbg.data.quality.assess_bars — the Dukascopy download's
data-quality report (gaps, monotonicity, OHLC sanity, coverage).

No network: bar frames here are synthetic, and the integration test
monkeypatches the downloader's internal fetch helper instead of calling
Dukascopy for real.
"""

import json

import numpy as np
import pandas as pd
import pytest

from fwbg.data.quality import assess_bars

# Anchor a Monday regardless of which calendar date "2024-01-01" actually is,
# so the weekend-heuristic tests don't depend on a memorized calendar fact.
_ANCHOR = pd.Timestamp("2024-01-01", tz="UTC")
_MONDAY = _ANCHOR - pd.Timedelta(days=_ANCHOR.weekday())


def _bars(timestamps, o=1.0, h=1.0, l=1.0, c=1.0, v=100.0):
    n = len(timestamps)

    def _arr(x):
        return np.full(n, x, dtype=float) if np.isscalar(x) else np.asarray(x, dtype=float)

    idx = pd.DatetimeIndex(timestamps)
    return pd.DataFrame(
        {"O": _arr(o), "H": _arr(h), "L": _arr(l), "C": _arr(c), "V": _arr(v)}, index=idx
    )


def test_clean_hourly_week_has_no_warnings_and_full_coverage():
    week1 = [_MONDAY + pd.Timedelta(hours=hr) for hr in range(0, 117)]  # Mon 00:00 .. Fri 20:00
    next_monday = _MONDAY + pd.Timedelta(days=7)
    week2 = [next_monday + pd.Timedelta(hours=hr) for hr in range(0, 6)]  # Mon 00:00 .. 05:00
    timestamps = week1 + week2

    df = _bars(timestamps)
    report = assess_bars(
        df,
        timeframe="HOUR_1",
        requested_start=timestamps[0].to_pydatetime(),
        requested_end=timestamps[-1].to_pydatetime(),
    )

    assert report["warnings"] == []
    assert report["weekend_gaps_ignored"] >= 1
    assert report["n_gaps"] == 0
    assert report["coverage"] == pytest.approx(1.0)


def test_weekday_hole_is_flagged_as_a_gap():
    # Monday 00:00 .. Wednesday 08:00 hourly, a 5-hour hole, then a few more bars.
    part1 = [_MONDAY + pd.Timedelta(hours=hr) for hr in range(0, 57)]  # .. Wed 08:00
    part2 = [_MONDAY + pd.Timedelta(hours=hr) for hr in range(61, 71)]  # resumes Wed 13:00
    timestamps = part1 + part2

    df = _bars(timestamps)
    report = assess_bars(
        df,
        timeframe="HOUR_1",
        requested_start=timestamps[0].to_pydatetime(),
        requested_end=timestamps[-1].to_pydatetime(),
    )

    assert report["n_gaps"] == 1
    assert report["weekend_gaps_ignored"] == 0
    assert any("gap" in w for w in report["warnings"])


def test_weekend_only_gap_is_ignored_not_flagged():
    friday_close = _MONDAY + pd.Timedelta(hours=4 * 24 + 20)  # Fri 20:00
    monday_open = _MONDAY + pd.Timedelta(days=7)  # next Mon 00:00
    timestamps = [friday_close, monday_open]

    df = _bars(timestamps)
    report = assess_bars(
        df,
        timeframe="HOUR_1",
        requested_start=timestamps[0].to_pydatetime(),
        requested_end=timestamps[-1].to_pydatetime(),
    )

    assert report["n_gaps"] == 0
    assert report["weekend_gaps_ignored"] == 1


def test_duplicate_and_backwards_timestamp_both_counted():
    t0 = _MONDAY
    timestamps = [t0, t0, t0 - pd.Timedelta(hours=1)]

    df = _bars(timestamps)
    report = assess_bars(
        df,
        timeframe="HOUR_1",
        requested_start=(t0 - pd.Timedelta(hours=1)).to_pydatetime(),
        requested_end=t0.to_pydatetime(),
    )

    assert report["duplicate_timestamps"] > 0
    assert report["non_monotonic"] > 0


def test_h_less_than_l_bar_counts_as_ohlc_violation():
    timestamps = [_MONDAY, _MONDAY + pd.Timedelta(hours=1)]
    df = _bars(timestamps, o=[1.0, 1.0], h=[1.0, 0.5], l=[1.0, 1.5], c=[1.0, 1.0])

    report = assess_bars(
        df,
        timeframe="HOUR_1",
        requested_start=timestamps[0].to_pydatetime(),
        requested_end=timestamps[-1].to_pydatetime(),
    )

    assert report["ohlc_violations"] == 1


def test_nan_close_counts_as_nan_bar():
    timestamps = [_MONDAY, _MONDAY + pd.Timedelta(hours=1)]
    df = _bars(timestamps, c=[1.0, float("nan")])

    report = assess_bars(
        df,
        timeframe="HOUR_1",
        requested_start=timestamps[0].to_pydatetime(),
        requested_end=timestamps[-1].to_pydatetime(),
    )

    assert report["nan_bars"] == 1


def test_coverage_halved_when_requested_range_is_double_the_data_range():
    timestamps = [_MONDAY + pd.Timedelta(hours=hr) for hr in range(0, 11)]  # 10h span
    df = _bars(timestamps)
    span = timestamps[-1] - timestamps[0]

    report = assess_bars(
        df,
        timeframe="HOUR_1",
        requested_start=timestamps[0].to_pydatetime(),
        requested_end=(timestamps[-1] + span).to_pydatetime(),
    )

    assert report["coverage"] == pytest.approx(0.5, abs=1e-9)
    assert any("coverage" in w for w in report["warnings"])


def test_unknown_timeframe_raises():
    with pytest.raises(ValueError):
        assess_bars(
            _bars([_MONDAY]),
            timeframe="NOT_A_TIMEFRAME",
            requested_start=_MONDAY.to_pydatetime(),
            requested_end=_MONDAY.to_pydatetime(),
        )


def test_download_writes_quality_report_and_surfaces_warnings(tmp_path, monkeypatch):
    """Integration: download() must write a .quality.json next to the CSV, carry
    quality.warnings in the result dict, and never refuse to write the CSV even
    when the checks warn (warn-only, no hard gate)."""
    import fwbg.data.dukascopy as dk_mod

    start = _MONDAY.to_pydatetime()
    # Mon 00:00 .. Wed 08:00, a 5-hour weekday hole, then a few more bars —
    # guarantees at least one WARN-level anomaly to exercise the warn path.
    part1 = [_MONDAY + pd.Timedelta(hours=hr) for hr in range(0, 57)]
    part2 = [_MONDAY + pd.Timedelta(hours=hr) for hr in range(61, 71)]
    timestamps = pd.DatetimeIndex(part1 + part2)
    end = timestamps[-1].to_pydatetime()

    side_frame = pd.DataFrame(
        {
            "open": np.full(len(timestamps), 1.1),
            "high": np.full(len(timestamps), 1.2),
            "low": np.full(len(timestamps), 1.0),
            "close": np.full(len(timestamps), 1.1),
            "volume": np.full(len(timestamps), 100.0),
        },
        index=timestamps,
    )

    def fake_fetch(instrument, interval, side, fetch_start, fetch_end, on_frac):
        on_frac(1.0)
        return side_frame.copy()

    # Identical bid/ask -> spread is exactly 0.0, so download() never calls
    # save_asset_spread (guarded by `if spread > 0`) and this test can't
    # perturb the real asset-spread store.
    monkeypatch.setattr(dk_mod, "_fetch_with_progress", fake_fetch)

    results = dk_mod.download(tmp_path, ["EURUSD"], "HOUR_1", start, end)

    assert len(results) == 1
    result = results[0]
    csv_path = tmp_path / result["file"]
    assert csv_path.exists()  # never blocked by the quality checks

    quality_path = csv_path.with_suffix(".quality.json")
    assert quality_path.exists()
    report = json.loads(quality_path.read_text())
    assert report["n_gaps"] == 1
    assert report["warnings"]

    assert "quality" in result
    assert result["quality"]["warnings"] == report["warnings"]
    assert result["quality"]["n_gaps"] == 1
