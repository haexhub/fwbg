"""Tests for MarketRegimeIndicator (Risk-On/Off Composite)."""

import numpy as np
import pandas as pd

from fwbg.core.registry import get_indicator


def _make_df_with_macro(n=500):
    """Create OHLC DataFrame with mock macro columns."""
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")

    df = pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.005,
        "L": close * 0.995,
        "C": close,
        "macro_vix": 20 + rng.normal(0, 3, n).cumsum() * 0.1,
        "macro_hyg": 80 + rng.normal(0, 0.5, n).cumsum() * 0.01,
        "macro_lqd": 110 + rng.normal(0, 0.3, n).cumsum() * 0.01,
        "macro_spx": 4500 + rng.normal(0, 10, n).cumsum(),
        "macro_tlt": 90 + rng.normal(0, 0.5, n).cumsum() * 0.01,
    }, index=idx)
    return df


class TestMarketRegimeIndicator:
    def test_plugin_registered(self):
        """market_regime should be discoverable via registry."""
        cls = get_indicator("market_regime")
        assert cls is not None

    def test_features_computed(self):
        """All expected regime features should be present."""
        cls = get_indicator("market_regime")
        indicator = cls()
        df = _make_df_with_macro()
        result = indicator.compute(df)

        expected = [
            "regime_vix_zscore", "regime_credit_zscore",
            "regime_equity_momentum", "regime_treasury_flight",
            "regime_risk_composite", "regime_risk_on", "regime_risk_off",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_risk_on_off_are_binary(self):
        """risk_on and risk_off flags should be 0 or 1."""
        cls = get_indicator("market_regime")
        indicator = cls()
        df = _make_df_with_macro(n=1000)
        result = indicator.compute(df)

        for col in ["regime_risk_on", "regime_risk_off"]:
            vals = result[col].dropna().unique()
            assert set(vals).issubset({0.0, 1.0}), f"{col} has non-binary values: {vals}"

    def test_graceful_without_macro(self):
        """Should return original df if no macro columns present."""
        cls = get_indicator("market_regime")
        indicator = cls()
        n = 200
        close = np.full(n, 100.0)
        df = pd.DataFrame({
            "O": close, "H": close * 1.01, "L": close * 0.99, "C": close,
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        result = indicator.compute(df)
        # No crash, original columns preserved
        assert "O" in result.columns
        assert "C" in result.columns

    def test_features_shifted(self):
        """All regime features should be shifted by 1 bar (lookahead prevention)."""
        cls = get_indicator("market_regime")
        indicator = cls()
        df = _make_df_with_macro()
        result = indicator.compute(df)

        # First row should be NaN for all regime features (shifted)
        for col in ["regime_vix_zscore", "regime_risk_composite"]:
            if col in result.columns:
                assert pd.isna(result[col].iloc[0]), f"{col} first row should be NaN (shifted)"
