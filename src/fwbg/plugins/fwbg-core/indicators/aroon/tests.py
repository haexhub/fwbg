"""Tests for Aroon Indicator plugin."""

import numpy as np
import pandas as pd
import pytest


def create_ohlc(close, high_factor=1.005, low_factor=0.995):
    n = len(close)
    return pd.DataFrame(
        {
            "O": close * 0.999,
            "H": close * high_factor,
            "L": close * low_factor,
            "C": close,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


@pytest.fixture
def indicator():
    from fwbg.plugins import import_plugin_module

    mod = import_plugin_module("fwbg-core", "indicators", "aroon")
    return mod.AroonIndicator()


class TestAroon:
    def test_aroon_up_near_100_in_uptrend(self, indicator):
        n = 300
        close = np.linspace(100, 200, n)
        result = indicator.compute(create_ohlc(close))
        valid = result["aroon_up"].dropna()
        assert valid.iloc[-50:].mean() > 80

    def test_aroon_down_near_100_in_downtrend(self, indicator):
        n = 300
        close = np.linspace(200, 100, n)
        result = indicator.compute(create_ohlc(close))
        valid = result["aroon_down"].dropna()
        assert valid.iloc[-50:].mean() > 80

    def test_aroon_bounded_0_to_100(self, indicator):
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(300) * 0.5)
        result = indicator.compute(create_ohlc(close))
        for col in ["aroon_up", "aroon_down"]:
            valid = result[col].dropna()
            assert (valid >= 0).all() and (valid <= 100).all()

    def test_name(self, indicator):
        assert indicator.name == "aroon"


def _reference_aroon(high, low, period):
    """Aroon from the most recent extreme, written with explicit loops."""
    up = np.full(len(high), np.nan)
    down = np.full(len(high), np.nan)
    for i in range(period, len(high)):
        hw = list(high[i - period : i + 1])
        lw = list(low[i - period : i + 1])
        # latest index holding the extreme
        last_high = max(j for j, v in enumerate(hw) if v == max(hw))
        last_low = max(j for j, v in enumerate(lw) if v == min(lw))
        up[i] = 100.0 * last_high / period
        down[i] = 100.0 * last_low / period
    return up, down


class TestAroonValues:
    def _df(self, n=200, seed=5):
        rng = np.random.default_rng(seed)
        # Integer prices produce many tied highs/lows.
        close = np.round(100 + np.cumsum(rng.normal(size=n))).astype(float)
        high = close + rng.integers(0, 2, size=n)
        low = close - rng.integers(0, 2, size=n)
        return pd.DataFrame(
            {"O": close, "H": high, "L": low, "C": close},
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )

    def test_matches_most_recent_extreme_reference_with_ties(self, indicator):
        df = self._df()
        result = indicator.compute(df, period=10)
        up, down = _reference_aroon(df["H"].to_numpy(), df["L"].to_numpy(), 10)
        pd.testing.assert_series_equal(
            result["aroon_up"],
            pd.Series(up, index=df.index).shift(1),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result["aroon_down"],
            pd.Series(down, index=df.index).shift(1),
            check_names=False,
        )

    def test_tied_high_counts_from_latest_bar(self, indicator):
        # Highs: 5 5 5 5 5 -> the most recent bar holds the high -> Aroon Up 100.
        n = 12
        df = pd.DataFrame(
            {"O": 1.0, "H": 5.0, "L": 1.0, "C": 3.0},
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )
        result = indicator.compute(df, period=4)
        assert result["aroon_up"].iloc[-1] == 100.0
        assert result["aroon_down"].iloc[-1] == 100.0

    def test_warmup_is_nan(self, indicator):
        result = indicator.compute(self._df(), period=10)
        # needs period + 1 bars, one more NaN row from the shift
        assert result["aroon_up"].iloc[:11].isna().all()
        assert result["aroon_up"].iloc[11:].notna().all()
        assert result["aroon_down"].iloc[:11].isna().all()

    def test_short_input_is_all_nan(self, indicator):
        result = indicator.compute(self._df(n=10), period=10)
        assert result["aroon_up"].isna().all()
