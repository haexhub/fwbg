"""Tests for the half-open date-window slice of the data loader (Plan 014).

`apply_date_window` restricts a DatetimeIndex-ed DataFrame to
[start_date, end_date) — start inclusive, end exclusive — so an in-sample
window ending at boundary date B and a holdout window starting at B never
share a row.
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import StrategyConfig
from fwbg.optimization import process
from fwbg.data.loader import apply_date_window
from fwbg.optimization.process import load_symbol_frame


def _make_df(n_days: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"C": np.arange(n_days, dtype=float)}, index=dates)


def test_half_open_slice_excludes_end_date():
    df = _make_df(10)
    # D0 = 2024-01-01 ... D5 = 2024-01-06
    out = apply_date_window(df, "2024-01-01", "2024-01-06")
    assert list(out.index) == list(pd.date_range("2024-01-01", "2024-01-05", freq="D"))


def test_only_start_date_is_inclusive():
    df = _make_df(10)
    out = apply_date_window(df, "2024-01-06", None)
    assert out.index.min() == pd.Timestamp("2024-01-06")
    assert len(out) == 5  # 01-06 .. 01-10


def test_only_end_date_is_exclusive():
    df = _make_df(10)
    out = apply_date_window(df, None, "2024-01-06")
    assert out.index.max() == pd.Timestamp("2024-01-05")
    assert len(out) == 5  # 01-01 .. 01-05


def test_adjacent_windows_share_no_row():
    """The exact scenario this plan fixes: an in-sample window ending at B
    and a holdout window starting at B must not both include B."""
    df = _make_df(10)
    boundary = "2024-01-06"
    in_sample = apply_date_window(df, None, boundary)
    holdout = apply_date_window(df, boundary, None)
    assert set(in_sample.index).isdisjoint(set(holdout.index))
    assert len(in_sample) + len(holdout) == len(df)


def test_malformed_start_date_raises_value_error_naming_value():
    df = _make_df(5)
    with pytest.raises(ValueError, match="not-a-date"):
        apply_date_window(df, "not-a-date", None)


def test_malformed_end_date_raises_value_error_naming_value():
    df = _make_df(5)
    with pytest.raises(ValueError, match="also-bad"):
        apply_date_window(df, None, "also-bad")


def _hourly(n: int = 48) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.linspace(1.0, 2.0, n)
    return pd.DataFrame({"O": close, "H": close + 0.1, "L": close - 0.1, "C": close}, index=index)


def test_load_symbol_frame_applies_the_half_open_window(monkeypatch):
    monkeypatch.setattr(process, "load_data_aligned", lambda path: _hourly())
    strategy = StrategyConfig(start_date="2024-01-01 06:00", end_date="2024-01-02")

    df = load_symbol_frame("EURUSD_HOUR_1.csv", strategy, "EURUSD")

    assert df.index[0] == pd.Timestamp("2024-01-01 06:00")
    assert df.index[-1] == pd.Timestamp("2024-01-01 23:00")


def test_load_symbol_frame_is_none_without_bars(monkeypatch):
    monkeypatch.setattr(process, "load_data_aligned", lambda path: _hourly())
    assert load_symbol_frame("EURUSD_HOUR_1.csv", StrategyConfig(start_date="2025-01-01"), "EURUSD") is None
    monkeypatch.setattr(process, "load_data_aligned", lambda path: None)
    assert load_symbol_frame("EURUSD_HOUR_1.csv", StrategyConfig(), "EURUSD") is None
