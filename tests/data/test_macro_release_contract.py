"""Tests for the release-date contract used by the macro/COT merge layer."""
from __future__ import annotations

import pandas as pd
import pytest

from fwbg.data.macro_contract import (
    RELEASE_COL,
    assert_release_dates_present,
    merge_respect_release,
)


def _make_bars(start: str, periods: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"C": [1.0] * periods},
        index=pd.date_range(start, periods=periods, freq="D", name="T"),
    )


def test_bar_at_T_cannot_see_release_after_T():
    bars = _make_bars("2024-01-01", 3)
    macro = pd.DataFrame(
        {
            RELEASE_COL: pd.to_datetime(["2024-01-02", "2024-01-05"]),
            "value": [100.0, 200.0],
        }
    )
    merged = merge_respect_release(bars, macro, "test", ["value"])

    # 01-01: no release available yet → NaN.
    # 01-02: first release becomes visible same day → 100.
    # 01-03: still only the first release → 100.
    assert pd.isna(merged.loc["2024-01-01", "value"])
    assert merged.loc["2024-01-02", "value"] == 100.0
    assert merged.loc["2024-01-03", "value"] == 100.0


def test_second_release_supersedes_first():
    bars = _make_bars("2024-01-04", 4)
    macro = pd.DataFrame(
        {
            RELEASE_COL: pd.to_datetime(["2024-01-02", "2024-01-05"]),
            "value": [100.0, 200.0],
        }
    )
    merged = merge_respect_release(bars, macro, "test", ["value"])
    assert merged.loc["2024-01-04", "value"] == 100.0
    assert merged.loc["2024-01-05", "value"] == 200.0
    assert merged.loc["2024-01-06", "value"] == 200.0
    assert merged.loc["2024-01-07", "value"] == 200.0


def test_index_and_input_preserved():
    bars = _make_bars("2024-01-01", 3)
    macro = pd.DataFrame(
        {RELEASE_COL: pd.to_datetime(["2024-01-02"]), "value": [100.0]}
    )
    merged = merge_respect_release(bars, macro, "test", ["value"])
    # release_date must not leak through.
    assert RELEASE_COL not in merged.columns
    # Original bar column preserved.
    assert "C" in merged.columns
    # Index name preserved.
    assert merged.index.name == "T"


def test_missing_release_date_column_raises():
    bars = _make_bars("2024-01-01", 1)
    bad_macro = pd.DataFrame({"value": [1.0]}, index=[pd.Timestamp("2024-01-01")])
    with pytest.raises(ValueError, match="release_date"):
        merge_respect_release(bars, bad_macro, "bad", ["value"])


def test_nat_in_release_date_raises():
    bars = _make_bars("2024-01-01", 1)
    bad_macro = pd.DataFrame(
        {RELEASE_COL: [pd.NaT, pd.Timestamp("2024-01-02")], "value": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="NaT"):
        merge_respect_release(bars, bad_macro, "bad", ["value"])


def test_non_datetime_release_date_raises():
    bars = _make_bars("2024-01-01", 1)
    bad_macro = pd.DataFrame(
        {RELEASE_COL: ["2024-01-02"], "value": [1.0]}  # string, not datetime
    )
    with pytest.raises(ValueError, match="datetime"):
        merge_respect_release(bars, bad_macro, "bad", ["value"])


def test_value_col_missing_raises():
    bars = _make_bars("2024-01-01", 1)
    macro = pd.DataFrame(
        {RELEASE_COL: pd.to_datetime(["2024-01-02"]), "value": [1.0]}
    )
    with pytest.raises(ValueError, match="not found"):
        merge_respect_release(bars, macro, "bad", ["other"])


def test_assert_release_dates_present_happy_path():
    df = pd.DataFrame(
        {RELEASE_COL: pd.to_datetime(["2024-01-01"]), "value": [1.0]}
    )
    # Should not raise.
    assert_release_dates_present(df, "ok")
