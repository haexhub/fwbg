"""Tests for CCI Indicator plugin."""

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

    mod = import_plugin_module("fwbg-core", "indicators", "cci")
    return mod.CCIIndicator()


class TestCCI:
    def test_cci_high_in_uptrend(self, indicator):
        n = 300
        close = np.linspace(100, 200, n)
        result = indicator.compute(create_ohlc(close))
        cci_cols = [c for c in result.columns if c.startswith("cci_")]
        for col in cci_cols:
            valid = result[col].dropna()
            assert valid.iloc[-50:].mean() > 50

    def test_cci_low_in_downtrend(self, indicator):
        n = 300
        close = np.linspace(200, 100, n)
        result = indicator.compute(create_ohlc(close))
        cci_cols = [c for c in result.columns if c.startswith("cci_")]
        for col in cci_cols:
            valid = result[col].dropna()
            assert valid.iloc[-50:].mean() < -50

    def test_name(self, indicator):
        assert indicator.name == "cci"


class TestCCIValues:
    def test_matches_reference_and_warmup_is_nan(self, indicator):
        rng = np.random.default_rng(11)
        n, period = 120, 20
        close = 100 + np.cumsum(rng.normal(size=n))
        high = close + rng.random(n)
        low = close - rng.random(n)
        df = pd.DataFrame(
            {"O": close, "H": high, "L": low, "C": close},
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )
        result = indicator.compute(df, periods=[period])

        tp = (df["H"] + df["L"] + df["C"]) / 3
        expected = np.full(n, np.nan)
        for i in range(period - 1, n):
            window = tp.iloc[i - period + 1 : i + 1].to_numpy()
            mean_dev = np.abs(window - window.mean()).mean()
            expected[i] = (tp.iloc[i] - window.mean()) / (0.015 * mean_dev)
        pd.testing.assert_series_equal(
            result[f"cci_{period}"],
            pd.Series(expected, index=df.index).shift(1),
            check_names=False,
            atol=1e-9,
            rtol=0,
        )
        assert result[f"cci_{period}"].iloc[:period].isna().all()
