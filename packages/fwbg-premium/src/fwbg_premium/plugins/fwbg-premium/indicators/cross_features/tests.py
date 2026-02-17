"""Tests for cross_features indicator plugin - COT x Vol interaction."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_cross = import_plugin_module("fwbg-premium", "indicators", "cross_features")
if _cross is None:
    pytest.skip("fwbg-premium cross_features plugin not available", allow_module_level=True)


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
        "macro_cot_eurusd": (50000 + rng.normal(0, 5000, n).cumsum()).round(),
    }, index=idx)
    return df


class TestCOTVolInteraction:
    """Tests for COT x Vol interaction in cross_features plugin."""

    def _get_indicator(self):
        return _cross.CrossFeatureIndicators()

    def test_interaction_columns_present(self):
        """COT x Vol interaction columns should be created."""
        ind = self._get_indicator()
        df = _make_ohlc_with_macro(n=2000)
        result = ind.compute(df)

        assert "cross_cot_eurusd_vol_interaction" in result.columns
        assert "cross_cot_eurusd_price_divergence" in result.columns

    def test_no_cot_columns_no_crash(self):
        """Should work gracefully without COT columns."""
        ind = self._get_indicator()
        n = 500
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.005,
            "L": close * 0.995, "C": close,
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        result = ind.compute(df)

        # Standard cross features should still work
        assert "cross_vol_trend" in result.columns
        # No COT columns = no interaction features
        cot_cols = [c for c in result.columns if "cot_" in c]
        assert len(cot_cols) == 0

    def test_interaction_sign_logic(self):
        """Extreme COT + low vol should produce high absolute interaction."""
        ind = self._get_indicator()
        n = 5000
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
        idx = pd.date_range("2024-01-01", periods=n, freq="h")

        cot = np.full(n, 100000.0)
        cot[:4000] = rng.normal(50000, 5000, 4000)
        cot[4000:] = 100000

        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.005,
            "L": close * 0.995, "C": close,
            "macro_cot_eurusd": cot,
        }, index=idx)

        result = ind.compute(df)
        interaction = result["cross_cot_eurusd_vol_interaction"].dropna()
        assert interaction.std() > 0

    def test_divergence_detects_mismatch(self):
        """Price divergence should detect when price and COT move opposite."""
        ind = self._get_indicator()
        n = 5000
        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=n, freq="h")

        close = 100 * np.exp(np.cumsum(np.full(n, 0.001)))
        cot = 50000 - np.arange(n) * 10.0 + rng.normal(0, 100, n)

        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.005,
            "L": close * 0.995, "C": close,
            "macro_cot_eurusd": cot,
        }, index=idx)

        result = ind.compute(df)
        divergence = result["cross_cot_eurusd_price_divergence"].dropna()
        late_values = divergence.iloc[-500:]
        assert late_values.mean() > 0, f"Expected positive divergence, got {late_values.mean()}"

    def test_no_lookahead(self):
        """All interaction features should be shifted."""
        ind = self._get_indicator()
        df = _make_ohlc_with_macro(n=2000)
        result = ind.compute(df)

        assert pd.isna(result["cross_cot_eurusd_vol_interaction"].iloc[0])
        assert pd.isna(result["cross_cot_eurusd_price_divergence"].iloc[0])

    def test_feature_columns_list(self):
        """get_feature_columns should include COT interaction columns."""
        ind = self._get_indicator()
        cols = ind.get_feature_columns()

        assert "cross_cot_eurusd_vol_interaction" in cols
        assert "cross_cot_eurusd_price_divergence" in cols
        assert "cross_cot_usdjpy_vol_interaction" in cols
