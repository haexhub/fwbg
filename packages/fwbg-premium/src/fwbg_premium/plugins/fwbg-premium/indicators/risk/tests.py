"""Tests for risk indicator plugin - Rolling Beta features."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_risk = import_plugin_module("fwbg-premium", "indicators", "risk")
if _risk is None:
    pytest.skip("fwbg-premium risk plugin not available", allow_module_level=True)


@pytest.fixture
def correlated_asset_df():
    """Asset das stark mit 'Benchmark' korreliert — für Beta-Tests."""
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    rng = np.random.default_rng(42)

    spx_returns = rng.normal(0.0005, 0.01, n)
    spx = 4000.0 * np.cumprod(1 + spx_returns)

    asset_returns = 2.0 * spx_returns + rng.normal(0, 0.005, n)
    close = 100.0 * np.cumprod(1 + asset_returns)

    return pd.DataFrame(
        {
            "O": close * 0.999,
            "H": close * 1.002,
            "L": close * 0.998,
            "C": close,
            "V": rng.integers(500, 2000, n).astype(float),
            "macro_spx": spx,
        },
        index=dates,
    )


@pytest.fixture
def trending_df():
    """Simple trending data without SPX."""
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    close = np.array([100.0 + i * 0.5 for i in range(n)])
    return pd.DataFrame(
        {
            "O": close - 0.3,
            "H": close + 0.5,
            "L": close - 0.5,
            "C": close,
            "V": np.random.default_rng(42).integers(500, 2000, n).astype(float),
        },
        index=dates,
    )


class TestRollingBeta:
    """Tests für Rolling Beta im Risk Plugin."""

    def _get_indicator(self):
        return _risk.RiskIndicators()

    def test_beta_columns_present(self, correlated_asset_df):
        """Beta-Spalten werden erzeugt wenn macro_spx vorhanden."""
        ind = self._get_indicator()
        result = ind.compute(correlated_asset_df)
        assert "beta_spx_50" in result.columns
        assert "beta_spx_100" in result.columns

    def test_beta_approximately_correct(self, correlated_asset_df):
        """Beta ~2.0 für ein Asset das mit Beta=2.0 zum SPX konstruiert wurde."""
        ind = self._get_indicator()
        result = ind.compute(correlated_asset_df)
        beta_100 = result["beta_spx_100"].dropna()
        last_values = beta_100.iloc[-50:]
        mean_beta = last_values.mean()
        assert 1.2 < mean_beta < 3.0, f"Expected beta ~2.0, got {mean_beta}"

    def test_no_beta_without_spx(self, trending_df):
        """Ohne macro_spx sollte kein Beta berechnet werden (keine Fehler)."""
        ind = self._get_indicator()
        result = ind.compute(trending_df)
        beta_cols = [c for c in result.columns if c.startswith("beta_spx")]
        assert len(beta_cols) == 0, f"Unexpected beta columns without SPX: {beta_cols}"

    def test_beta_no_lookahead(self, correlated_asset_df):
        """Beta-Features sollten um 1 Bar verschoben sein."""
        ind = self._get_indicator()
        result = ind.compute(correlated_asset_df)
        assert pd.isna(result["beta_spx_50"].iloc[0])
        assert pd.isna(result["beta_spx_100"].iloc[0])

    def test_beta_sign_positive_for_correlated(self, correlated_asset_df):
        """Beta sollte positiv sein für positiv korrelierte Assets."""
        ind = self._get_indicator()
        result = ind.compute(correlated_asset_df)
        beta_50 = result["beta_spx_50"].dropna()
        last_values = beta_50.iloc[-50:]
        assert last_values.mean() > 0, f"Beta should be positive, got {last_values.mean()}"
