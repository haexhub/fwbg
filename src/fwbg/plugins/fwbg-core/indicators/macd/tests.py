"""Tests for MACD Indicator plugin."""

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

    mod = import_plugin_module("fwbg-core", "indicators", "macd")
    return mod.MACDIndicator()


class TestMACD:
    def test_macd_line_positive_in_uptrend(self, indicator):
        n = 300
        close = np.linspace(100, 200, n)
        result = indicator.compute(create_ohlc(close))
        valid = result["macd_line"].dropna()
        assert (valid.iloc[-50:] > 0).all()

    def test_macd_line_negative_in_downtrend(self, indicator):
        n = 300
        close = np.linspace(200, 100, n)
        result = indicator.compute(create_ohlc(close))
        valid = result["macd_line"].dropna()
        assert (valid.iloc[-50:] < 0).all()

    def test_macd_hist_flip_at_reversal(self, indicator):
        close = np.concatenate([np.linspace(100, 80, 200), np.linspace(80, 120, 200)])
        result = indicator.compute(create_ohlc(close))
        flips = result["macd_hist_flip"].dropna()
        assert flips.sum() > 0

    def test_macd_above_zero_in_uptrend(self, indicator):
        n = 300
        close = np.linspace(100, 200, n)
        result = indicator.compute(create_ohlc(close))
        valid = result["macd_above_zero"].dropna()
        assert (valid.iloc[-50:] == 1).all()

    def test_feature_columns(self, indicator):
        cols = indicator.get_feature_columns()
        assert "macd_line" in cols
        assert "macd_hist" in cols
        assert "macd_above_zero" in cols

    def test_name(self, indicator):
        assert indicator.name == "macd"


def _reference_macd(close, fast, slow, signal):
    """MACD via plain pandas EMAs (adjust=False, NaN until the span is filled)."""

    def ema(series, span):
        return series.ewm(span=span, adjust=False, min_periods=span).mean()

    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


class TestMACDValues:
    def _df(self, n=200, seed=3):
        rng = np.random.default_rng(seed)
        close = pd.Series(
            100 + np.cumsum(rng.normal(size=n)),
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )
        return create_ohlc(close.to_numpy())

    def test_matches_reference_after_one_bar_shift(self, indicator):
        df = self._df()
        result = indicator.compute(df)
        line, sig, hist = _reference_macd(df["C"], 12, 26, 9)
        for name, ref in (
            ("macd_line", line / df["C"]),
            ("macd_signal", sig / df["C"]),
            ("macd_hist", hist / df["C"]),
            ("macd_dist_zero", line.abs() / df["C"]),
        ):
            pd.testing.assert_series_equal(
                result[name], ref.shift(1), check_names=False, atol=1e-12, rtol=0
            )

    def test_custom_periods_match_reference(self, indicator):
        df = self._df()
        result = indicator.compute(df, fast_period=5, slow_period=13, signal_period=4)
        line, _, _ = _reference_macd(df["C"], 5, 13, 4)
        pd.testing.assert_series_equal(
            result["macd_line"],
            (line / df["C"]).shift(1),
            check_names=False,
            atol=1e-12,
            rtol=0,
        )

    def test_hist_flip_is_nan_during_warmup(self, indicator):
        df = self._df()
        result = indicator.compute(df)
        flip = result["macd_hist_flip"]
        # hist is valid from row slow + signal - 2 = 33, flip needs a valid
        # predecessor (row 34), and the one-bar shift moves it to row 35.
        assert flip.iloc[:35].isna().all()
        assert flip.iloc[35:].notna().all()
        assert flip.dropna().isin([0.0, 1.0]).all()

    def test_hist_flip_matches_sign_changes(self, indicator):
        df = self._df()
        flip = indicator.compute(df)["macd_hist_flip"]
        _, _, hist = _reference_macd(df["C"], 12, 26, 9)
        valid = hist.dropna()
        expected = (np.sign(valid) != np.sign(valid.shift(1))).astype(float)
        expected.iloc[0] = np.nan
        pd.testing.assert_series_equal(
            flip, expected.reindex(df.index).shift(1), check_names=False
        )

    @pytest.mark.parametrize("fast, slow", [(26, 12), (12, 12)])
    def test_fast_must_be_smaller_than_slow(self, indicator, fast, slow):
        with pytest.raises(ValueError, match="fast_period"):
            indicator.compute(self._df(), fast_period=fast, slow_period=slow)
