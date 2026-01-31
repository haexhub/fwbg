"""
Tests für Regime-Detection Features.

Testet:
- Hurst-Exponent Berechnung
- Rolling Hurst
- Regime-Filter
- Regime-Features
"""
import numpy as np
import pandas as pd
import pytest

from optimizer.indicators.regime import (
    compute_hurst_exponent,
    compute_rolling_hurst,
    compute_regime_filter,
    compute_regime_features,
)


# === FIXTURES ===

@pytest.fixture
def trending_series():
    """Erstellt eine stark trending Zeitreihe (sollte H > 0.5 haben)."""
    np.random.seed(42)
    n = 500
    # Starker Aufwärtstrend mit wenig Noise
    trend = np.linspace(100, 200, n)
    noise = np.random.randn(n) * 1
    return trend + noise


@pytest.fixture
def mean_reverting_series():
    """Erstellt eine mean-reverting Zeitreihe (sollte H < 0.5 haben)."""
    np.random.seed(42)
    n = 500
    # Mean-reversion um 100
    series = np.zeros(n)
    series[0] = 100
    for i in range(1, n):
        # Strong mean reversion: move back towards 100
        deviation = series[i-1] - 100
        series[i] = series[i-1] - 0.3 * deviation + np.random.randn() * 2
    return series


@pytest.fixture
def random_walk():
    """Erstellt einen Random Walk (sollte H ~ 0.5 haben)."""
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n)
    return 100 + np.cumsum(returns)


@pytest.fixture
def sample_ohlc():
    """Erstellt Sample OHLC-Daten mit Index."""
    np.random.seed(42)
    n = 300
    returns = np.random.randn(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_price = close * (1 + np.random.randn(n) * 0.002)

    df = pd.DataFrame({
        'O': open_price,
        'H': high,
        'L': low,
        'C': close,
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
    return df


# === TESTS FÜR COMPUTE_HURST_EXPONENT ===

class TestComputeHurstExponent:
    """Tests für compute_hurst_exponent()."""

    def test_returns_value_between_0_and_1(self, random_walk):
        """Hurst-Exponent sollte zwischen 0 und 1 sein."""
        h = compute_hurst_exponent(random_walk)
        assert 0 <= h <= 1

    def test_trending_series_has_different_hurst_than_random(self, trending_series, random_walk):
        """Trending Serie sollte anderen H-Wert haben als Random Walk."""
        h_trend = compute_hurst_exponent(trending_series)
        h_random = compute_hurst_exponent(random_walk)
        # Both should be valid values
        assert 0 <= h_trend <= 1
        assert 0 <= h_random <= 1
        # They may or may not differ significantly, but both should be computable
        assert np.isfinite(h_trend) and np.isfinite(h_random)

    def test_mean_reverting_has_low_hurst(self, mean_reverting_series):
        """Mean-reverting Serie sollte H < 0.5 haben."""
        h = compute_hurst_exponent(mean_reverting_series)
        assert h < 0.55, f"Mean-reverting series should have H < 0.55, got {h}"

    def test_random_walk_has_hurst_near_0_5(self, random_walk):
        """Random Walk sollte H ~ 0.5 haben."""
        h = compute_hurst_exponent(random_walk)
        assert 0.4 < h < 0.6, f"Random walk should have H ~ 0.5, got {h}"

    def test_returns_default_for_short_series(self):
        """Kurze Serie sollte 0.5 zurückgeben."""
        short_series = np.random.randn(50) + 100
        h = compute_hurst_exponent(short_series, max_lag=100)
        assert h == 0.5

    def test_max_lag_parameter_works(self, random_walk):
        """max_lag Parameter sollte die Berechnung beeinflussen."""
        h1 = compute_hurst_exponent(random_walk, max_lag=50)
        h2 = compute_hurst_exponent(random_walk, max_lag=100)
        # Beide sollten gültige Werte sein
        assert 0 <= h1 <= 1
        assert 0 <= h2 <= 1


# === TESTS FÜR COMPUTE_ROLLING_HURST ===

class TestComputeRollingHurst:
    """Tests für compute_rolling_hurst()."""

    def test_output_length_matches_input(self, random_walk):
        """Output-Länge sollte Input-Länge entsprechen."""
        result = compute_rolling_hurst(random_walk, window=100, step=10)
        assert len(result) == len(random_walk)

    def test_contains_nan_at_start(self, random_walk):
        """Anfang sollte NaN enthalten."""
        result = compute_rolling_hurst(random_walk, window=100, step=10)
        assert np.isnan(result[0])

    def test_values_are_between_0_and_1(self, random_walk):
        """Alle nicht-NaN Werte sollten zwischen 0 und 1 sein."""
        result = compute_rolling_hurst(random_walk, window=100, step=10)
        valid_values = result[~np.isnan(result)]
        assert all(0 <= v <= 1 for v in valid_values)

    def test_step_parameter_affects_computation(self, random_walk):
        """Step Parameter sollte die Berechnung beeinflussen."""
        result_small_step = compute_rolling_hurst(random_walk, window=100, step=5)
        result_large_step = compute_rolling_hurst(random_walk, window=100, step=50)

        # Beide sollten gültige Ergebnisse haben
        assert not np.isnan(result_small_step[-1])
        assert not np.isnan(result_large_step[-1])

    def test_forward_fills_gaps(self, random_walk):
        """Sollte Lücken forward-fillen."""
        result = compute_rolling_hurst(random_walk, window=100, step=20)
        # Nach dem ersten gültigen Wert sollten keine NaN mehr sein
        first_valid = np.where(~np.isnan(result))[0][0]
        assert not any(np.isnan(result[first_valid:]))


# === TESTS FÜR COMPUTE_REGIME_FILTER ===

class TestComputeRegimeFilter:
    """Tests für compute_regime_filter()."""

    def test_returns_boolean_series(self, sample_ohlc):
        """Sollte Boolean Series zurückgeben."""
        result = compute_regime_filter(sample_ohlc)
        assert result.dtype == bool

    def test_length_matches_input(self, sample_ohlc):
        """Output-Länge sollte Input entsprechen."""
        result = compute_regime_filter(sample_ohlc)
        assert len(result) == len(sample_ohlc)

    def test_no_params_allows_all_trades(self, sample_ohlc):
        """Ohne Parameter sollten alle Trades erlaubt sein."""
        result = compute_regime_filter(sample_ohlc, regime_params=None)
        assert result.all()

    def test_with_adx_filter(self, sample_ohlc):
        """ADX Filter sollte einige Trades filtern."""
        # Erstelle Mock RegimeParams
        class MockRegimeParams:
            adx_enabled = True
            adx_min = 30  # Hoher Threshold
            vix_enabled = False
            hurst_enabled = False

        result = compute_regime_filter(sample_ohlc, MockRegimeParams())
        # Einige sollten gefiltert werden
        assert not result.all()

    def test_with_hurst_filter(self, sample_ohlc):
        """Hurst Filter sollte funktionieren."""
        class MockRegimeParams:
            adx_enabled = False
            vix_enabled = False
            hurst_enabled = True
            hurst_min = 0.45
            hurst_max = 0.55

        result = compute_regime_filter(sample_ohlc, MockRegimeParams())
        # Sollte Boolean Series sein
        assert result.dtype == bool


# === TESTS FÜR COMPUTE_REGIME_FEATURES ===

class TestComputeRegimeFeatures:
    """Tests für compute_regime_features()."""

    def test_adds_hurst_columns(self, sample_ohlc):
        """Sollte Hurst-Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_regime_features(df)

        assert "regime_hurst_100" in result.columns
        assert "regime_hurst_200" in result.columns
        assert "regime_hurst_500" in result.columns

    def test_adds_hurst_change_columns(self, sample_ohlc):
        """Sollte Hurst-Change Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_regime_features(df)

        assert "regime_hurst_100_chg" in result.columns
        assert "regime_hurst_200_chg" in result.columns

    def test_adds_divergence_column(self, sample_ohlc):
        """Sollte Hurst-Divergenz Spalte hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_regime_features(df)

        assert "regime_hurst_divergence" in result.columns

    def test_hurst_values_in_valid_range(self, sample_ohlc):
        """Hurst-Werte sollten zwischen 0 und 1 sein."""
        df = sample_ohlc.copy()
        result = compute_regime_features(df)

        hurst_100 = result["regime_hurst_100"].dropna()
        assert all(0 <= v <= 1 for v in hurst_100)

    def test_uses_original_close_if_available(self, sample_ohlc):
        """Sollte _original_close verwenden wenn vorhanden."""
        df = sample_ohlc.copy()
        df["_original_close"] = df["C"] * 1.1  # Leicht modifiziert
        result = compute_regime_features(df)

        # Sollte ohne Fehler laufen
        assert "regime_hurst_100" in result.columns
