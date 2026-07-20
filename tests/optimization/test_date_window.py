"""Tests for the half-open date-window slice in process.py (Plan 014).

`_apply_date_window` restricts a DatetimeIndex-ed DataFrame to
[start_date, end_date) — start inclusive, end exclusive — so an in-sample
window ending at boundary date B and a holdout window starting at B never
share a row.
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.optimization.process import _apply_date_window


def _make_df(n_days: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"C": np.arange(n_days, dtype=float)}, index=dates)


def test_half_open_slice_excludes_end_date():
    df = _make_df(10)
    # D0 = 2024-01-01 ... D5 = 2024-01-06
    out = _apply_date_window(df, "2024-01-01", "2024-01-06")
    assert list(out.index) == list(pd.date_range("2024-01-01", "2024-01-05", freq="D"))


def test_only_start_date_is_inclusive():
    df = _make_df(10)
    out = _apply_date_window(df, "2024-01-06", None)
    assert out.index.min() == pd.Timestamp("2024-01-06")
    assert len(out) == 5  # 01-06 .. 01-10


def test_only_end_date_is_exclusive():
    df = _make_df(10)
    out = _apply_date_window(df, None, "2024-01-06")
    assert out.index.max() == pd.Timestamp("2024-01-05")
    assert len(out) == 5  # 01-01 .. 01-05


def test_adjacent_windows_share_no_row():
    """The exact scenario this plan fixes: an in-sample window ending at B
    and a holdout window starting at B must not both include B."""
    df = _make_df(10)
    boundary = "2024-01-06"
    in_sample = _apply_date_window(df, None, boundary)
    holdout = _apply_date_window(df, boundary, None)
    assert set(in_sample.index).isdisjoint(set(holdout.index))
    assert len(in_sample) + len(holdout) == len(df)


def test_malformed_start_date_raises_value_error_naming_value():
    df = _make_df(5)
    with pytest.raises(ValueError, match="not-a-date"):
        _apply_date_window(df, "not-a-date", None)


def test_malformed_end_date_raises_value_error_naming_value():
    df = _make_df(5)
    with pytest.raises(ValueError, match="also-bad"):
        _apply_date_window(df, None, "also-bad")
