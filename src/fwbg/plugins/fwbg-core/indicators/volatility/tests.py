"""Tests for volatility indicator plugin - RV/IV features."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_vol = import_plugin_module("fwbg-core", "indicators", "volatility")
if _vol is None:
    pytest.skip("fwbg-core volatility plugin not available", allow_module_level=True)


def _make_ohlc_with_macro(n=2000):
    """Create OHLC DataFrame with macro columns for testing."""
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")

    df = pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.005,
        "L": close * 0.995,
        "C": close,
        "macro_vix": 20.0 + rng.normal(0, 0.5, n).cumsum() * 0.1,
    }, index=idx)
    return df


class TestRealizedVsImpliedVol:
    """Tests for RV/VIX features in volatility plugin."""

    def _get_indicator(self):
        return _vol.VolatilityIndicators()

    def test_rv_computed(self):
        """Realized vol should be computed from close-to-close returns."""
        ind = self._get_indicator()
        df = _make_ohlc_with_macro(n=2000)
        result = ind.compute(df)

        assert "vol_rv_20" in result.columns
        rv = result["vol_rv_20"].dropna()
        assert len(rv) > 0
        assert rv.mean() > 0

    def test_rv_iv_ratio_with_vix(self):
        """RV/VIX ratio should be computed when macro_vix present."""
        ind = self._get_indicator()
        df = _make_ohlc_with_macro(n=2000)
        result = ind.compute(df)

        assert "vol_rv_iv_ratio" in result.columns
        assert "vol_rv_iv_spread" in result.columns
        ratio = result["vol_rv_iv_ratio"].dropna()
        assert len(ratio) > 0
        assert ratio.mean() > 0

    def test_rv_iv_not_computed_without_vix(self):
        """RV/VIX should NOT be computed when macro_vix absent."""
        ind = self._get_indicator()
        n = 2000
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.005,
            "L": close * 0.995, "C": close,
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        result = ind.compute(df)

        assert "vol_rv_20" in result.columns
        assert "vol_rv_iv_ratio" not in result.columns
        assert "vol_rv_iv_spread" not in result.columns

    def test_rv_no_lookahead(self):
        """RV features should be shifted by 1 bar."""
        ind = self._get_indicator()
        df = _make_ohlc_with_macro(n=2000)
        result = ind.compute(df)

        assert pd.isna(result["vol_rv_20"].iloc[0])
        assert pd.isna(result["vol_rv_iv_ratio"].iloc[0])
