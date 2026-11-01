"""Unit tests for fwbg.data.validation (pre-run data checks)."""

import numpy as np
import pandas as pd
import pytest

from fwbg.data.trading_hours import parse_trading_hours
from fwbg.data.validation import DataCheckError, Severity, check_bars

FX = parse_trading_hours(
    {"tz": "America/New_York", "windows": [{"days": "Sun-Thu", "start": "17:00", "end": "17:00"}]}
)


def fx_hours(start="2026-01-05", weeks=2, seed=1):
    """Hourly bars of an FX week (Sun 22:00 - Fri 22:00 UTC in winter), random walk."""
    idx = pd.date_range(start, periods=weeks * 7 * 24, freq="h")
    idx = idx[FX.overlaps(idx, 60)]
    rng = np.random.default_rng(seed)
    c = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame(
        {
            "O": o,
            "H": np.maximum(o, c) * 1.0005,
            "L": np.minimum(o, c) * 0.9995,
            "C": c,
            "V": 100.0,
        },
        index=idx,
    )


def codes(report, severity=None):
    return {f.code for f in report.findings if severity is None or f.severity is severity}


def test_clean_data_has_no_findings():
    df = fx_hours()
    out, report = check_bars(df, symbol="EURUSD", bar_minutes=60, hours=FX)
    assert report.findings == () and len(out) == len(df)
    assert report.to_dict()["errors"] == [] and report.summary() == "ok"


def test_filler_bars_are_dropped_and_reported():
    df = fx_hours()
    weekend = pd.date_range("2026-01-10 00:00", periods=3, freq="h")
    last = df["C"].iloc[-1]
    filler = pd.DataFrame({"O": last, "H": last, "L": last, "C": last, "V": 0.0}, index=weekend)
    df = pd.concat([df[df.index < weekend[0]], filler, df[df.index > weekend[-1]]])
    out, report = check_bars(df, symbol="EURUSD", bar_minutes=60, hours=FX)
    assert len(out) == len(df) - 3 and not out.index.isin(weekend).any()
    assert codes(report) == {"filler_bars_dropped"}


def test_flat_bars_are_kept_inside_hours_when_the_source_has_no_volume():
    df = fx_hours().assign(V=0.0)
    df.iloc[10, :4] = df["C"].iloc[9]  # one quiet, flat bar while the market is open
    out, report = check_bars(df, symbol="EURUSD", bar_minutes=60, hours=FX)
    assert len(out) == len(df) and "filler_bars_dropped" not in codes(report)


@pytest.mark.parametrize(
    "breaker, code",
    [
        (lambda df: df.iloc[::-1], "unordered_timestamps"),
        (lambda df: pd.concat([df, df.iloc[[5]]]).sort_index(), "duplicate_timestamps"),
        (lambda df: df.assign(C=df["C"].where(df.index != df.index[7], np.nan)), "invalid_prices"),
        (
            lambda df: df.assign(L=df["L"].where(df.index != df.index[7], df["H"] * 1.01)),
            "ohlc_inconsistent",
        ),
    ],
)
def test_structural_errors(breaker, code):
    _, report = check_bars(breaker(fx_hours()), symbol="EURUSD", bar_minutes=60, hours=FX)
    assert code in codes(report, Severity.ERROR)
    with pytest.raises(DataCheckError) as exc:
        raise DataCheckError(report)
    assert exc.value.report is report and code in str(exc.value)


def test_shifted_time_zone_is_an_error_a_stray_bar_a_warning():
    df = fx_hours()
    shifted = df.set_axis(df.index + pd.Timedelta(hours=10))  # e.g. local time stored as UTC
    _, report = check_bars(shifted, symbol="EURUSD", bar_minutes=60, hours=FX)
    assert "outside_trading_hours" in codes(report, Severity.ERROR)
    stray = pd.concat([df, df.iloc[[0]].set_axis([pd.Timestamp("2026-01-10 12:00")])]).sort_index()
    stray.loc["2026-01-10 12:00", "V"] = 50.0  # a real quote on Saturday
    _, report = check_bars(stray, symbol="EURUSD", bar_minutes=60, hours=FX)
    assert "outside_trading_hours" in codes(report, Severity.WARNING)


def test_missing_bars_and_long_gaps():
    df = fx_hours(weeks=3)
    holiday = df[(df.index >= "2026-01-07 00:00") & (df.index < "2026-01-08 00:00")].index
    _, report = check_bars(df.drop(holiday), symbol="EURUSD", bar_minutes=60, hours=FX)
    missing = {f.code: f for f in report.findings}
    assert missing["missing_bars"].count == 24 and missing["missing_bars"].severity is Severity.INFO
    assert "long_gap" not in missing
    week = df[(df.index >= "2026-01-12") & (df.index < "2026-01-20")].index  # 6 trading days
    _, report = check_bars(df.drop(week), symbol="EURUSD", bar_minutes=60, hours=FX)
    assert "long_gap" in codes(report, Severity.WARNING)


def test_reverting_spike_and_frozen_prices_are_warnings_not_changes():
    df = fx_hours(weeks=3)
    i = 200
    df.iloc[i, df.columns.get_loc("C")] *= 1.05  # bad tick: +5 % and straight back
    df.iloc[i, df.columns.get_loc("H")] = df["C"].iloc[i]
    df.iloc[i + 1, df.columns.get_loc("O")] = df["C"].iloc[i]
    df.iloc[i + 1, df.columns.get_loc("H")] = df["C"].iloc[i]
    frozen = df.index[300:307]
    df.loc[frozen, ["O", "H", "L", "C"]] = df["C"].iloc[299]
    out, report = check_bars(df, symbol="EURUSD", bar_minutes=60, hours=FX)
    found = {f.code: f for f in report.findings}
    assert found["price_spikes"].examples[0] == df.index[i].isoformat()
    assert found["stale_prices"].count == 1 and found["stale_prices"].severity is Severity.WARNING
    pd.testing.assert_frame_equal(out, df)  # reported, not altered


def test_unknown_hours_skip_the_hours_checks():
    df = fx_hours()
    shifted = df.set_axis(df.index + pd.Timedelta(hours=10))
    _, report = check_bars(shifted, symbol="QQQ", bar_minutes=60, hours=None)
    assert report.trading_hours is None and not codes(report)
