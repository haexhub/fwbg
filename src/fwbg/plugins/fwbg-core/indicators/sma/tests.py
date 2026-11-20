"""Tests for SMA Indicator plugin."""

import numpy as np
import pandas as pd
import pytest


def create_ohlc(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {"O": close, "H": close + 1.0, "L": close - 1.0, "C": close},
        index=pd.date_range("2024-01-01", periods=len(close), freq="h"),
    )


@pytest.fixture
def indicator():
    from fwbg.plugins import import_plugin_module

    mod = import_plugin_module("fwbg-core", "indicators", "sma")
    return mod.SMAIndicator()


@pytest.fixture
def df():
    rng = np.random.default_rng(13)
    return create_ohlc(100 + np.cumsum(rng.normal(size=150)))


LINES = [
    {"period": 5, "source": "C"},
    {"period": 20, "source": "C"},
    {"period": 10, "source": "H"},
]


class TestSMAValues:
    def test_line_and_distance_match_rolling_mean(self, indicator, df):
        result = indicator.compute(df, lines=LINES)
        for period, source, key in ((5, "C", "5"), (20, "C", "20"), (10, "H", "10_h")):
            ref = df[source].rolling(period).mean()
            pd.testing.assert_series_equal(
                result[f"_sma_{key}"], ref.shift(1), check_names=False, atol=1e-12
            )
            pd.testing.assert_series_equal(
                result[f"sma_dist_{key}"],
                ((df["C"] - ref) / df["C"]).shift(1),
                check_names=False,
                atol=1e-12,
            )

    def test_warmup_rows_are_nan(self, indicator, df):
        result = indicator.compute(df, lines=LINES)
        # SMA(20) is valid from row 19; the shift moves it to row 20
        assert result["_sma_20"].iloc[:20].isna().all()
        assert result["_sma_20"].iloc[20:].notna().all()
        assert result["sma_dist_20"].iloc[:20].isna().all()

    def test_crossings_are_nan_until_both_averages_exist(self, indicator, df):
        result = indicator.compute(df, lines=LINES)
        above = result["sma_5_above_20"]
        assert above.iloc[:20].isna().all()  # SMA(20) not yet valid
        valid = above.iloc[20:]
        assert valid.notna().all()
        ref = (df["C"].rolling(5).mean() > df["C"].rolling(20).mean()).astype(float)
        pd.testing.assert_series_equal(valid, ref.shift(1).iloc[20:], check_names=False)

    def test_cross_source_crossing_column_exists(self, indicator, df):
        result = indicator.compute(df, lines=LINES)
        assert "sma_5_above_10_h" in result.columns
        assert result["sma_5_above_10_h"].iloc[:10].isna().all()

    def test_duplicate_lines_are_computed_once(self, indicator, df):
        lines = [
            {"period": 5},
            {"period": 5, "source": "C"},
            {"period": 20, "source": "c"},
            {"period": 20, "source": "C"},
        ]
        result = indicator.compute(df, lines=lines)
        assert not result.columns.duplicated().any()
        assert indicator.get_feature_columns({"lines": lines}) == [
            "sma_dist_5",
            "sma_dist_20",
            "sma_5_above_20",
        ]
        assert indicator.get_overlay_columns({"lines": lines}) == ["_sma_5", "_sma_20"]

    def test_period_below_two_raises(self, indicator, df):
        with pytest.raises(ValueError, match="period"):
            indicator.compute(df, lines=[{"period": 1}])

    def test_name(self, indicator):
        assert indicator.name == "sma"
