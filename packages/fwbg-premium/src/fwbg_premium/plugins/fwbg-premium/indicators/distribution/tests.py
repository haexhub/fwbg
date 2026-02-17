"""Tests for distribution indicator plugin - Auto-Correlation features."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_dist = import_plugin_module("fwbg-premium", "indicators", "distribution")
if _dist is None:
    pytest.skip("fwbg-premium distribution plugin not available", allow_module_level=True)


@pytest.fixture
def trending_df():
    """Linearer Aufwärtstrend — bekannte Auto-Korrelation > 0."""
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


@pytest.fixture
def mean_reverting_df():
    """Oszillierender Preis — bekannte Auto-Korrelation < 0."""
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="h")
    rng = np.random.default_rng(42)
    base = 100.0
    prices = [base]
    for i in range(1, n):
        step = -np.sign(prices[-1] - base) * 1.0 + rng.normal(0, 0.3)
        prices.append(prices[-1] + step)
    close = np.array(prices)
    return pd.DataFrame(
        {
            "O": close - 0.2,
            "H": close + 0.3,
            "L": close - 0.3,
            "C": close,
            "V": rng.integers(500, 2000, n).astype(float),
        },
        index=dates,
    )


class TestAutoCorrelation:
    """Tests für Auto-Korrelation Features im Distribution Plugin."""

    def _get_indicator(self):
        return _dist.DistributionIndicators()

    def test_autocorr_columns_present(self, trending_df):
        """Alle Auto-Korrelation Spalten werden erzeugt."""
        ind = self._get_indicator()
        result = ind.compute(trending_df)
        for lag in [1, 5, 10, 20]:
            assert f"dist_autocorr_{lag}" in result.columns
        assert "dist_autocorr_1_change" in result.columns

    def test_trending_positive_autocorrelation(self, trending_df):
        """Trending-Daten sollten positive Auto-Korrelation auf Lag 1 haben."""
        ind = self._get_indicator()
        result = ind.compute(trending_df)
        autocorr_1 = result["dist_autocorr_1"].dropna()
        last_values = autocorr_1.iloc[-50:]
        mean_ac = last_values.mean()
        assert mean_ac > 0, f"Expected positive autocorr for trend, got {mean_ac}"

    def test_mean_reverting_negative_autocorrelation(self, mean_reverting_df):
        """Mean-Reverting-Daten sollten negative Auto-Korrelation auf Lag 1 haben."""
        ind = self._get_indicator()
        result = ind.compute(mean_reverting_df)
        autocorr_1 = result["dist_autocorr_1"].dropna()
        last_values = autocorr_1.iloc[-50:]
        mean_ac = last_values.mean()
        assert mean_ac < 0, f"Expected negative autocorr for mean-reversion, got {mean_ac}"

    def test_autocorr_bounded(self, trending_df):
        """Auto-Korrelation sollte zwischen -1 und 1 liegen."""
        ind = self._get_indicator()
        result = ind.compute(trending_df)
        for lag in [1, 5, 10, 20]:
            col = result[f"dist_autocorr_{lag}"].dropna()
            assert col.min() >= -1.01, f"Lag {lag}: min={col.min()}"
            assert col.max() <= 1.01, f"Lag {lag}: max={col.max()}"

    def test_autocorr_change_detects_regime_shift(self):
        """Auto-Korrelation Change sollte Regime-Wechsel erkennen."""
        ind = self._get_indicator()
        n = 400
        dates = pd.date_range("2023-01-01", periods=n, freq="h")
        rng = np.random.default_rng(42)

        prices = [100.0]
        for i in range(1, n):
            if i < 200:
                step = 0.5 + rng.normal(0, 0.1)
            else:
                step = -np.sign(prices[-1] - 200) * 1.0 + rng.normal(0, 0.2)
            prices.append(prices[-1] + step)

        df = pd.DataFrame(
            {
                "O": np.array(prices) - 0.2,
                "H": np.array(prices) + 0.3,
                "L": np.array(prices) - 0.3,
                "C": np.array(prices),
                "V": rng.integers(500, 2000, n).astype(float),
            },
            index=dates,
        )

        result = ind.compute(df)
        change_col = result["dist_autocorr_1_change"].dropna()
        assert not change_col.empty
        assert change_col.std() > 0, "Change should vary during regime shift"

    def test_no_lookahead_bias(self, trending_df):
        """Features sollten um 1 Bar verschoben sein (kein Lookahead)."""
        ind = self._get_indicator()
        result = ind.compute(trending_df)
        assert pd.isna(result["dist_autocorr_1"].iloc[0])
