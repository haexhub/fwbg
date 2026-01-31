"""
Tests für Preprocessing-Funktionen.

Testet:
- Fractional Differentiation (Gewichte, Transformation, Stationarität)
- Log-Returns Transformation
- Z-Score Normalisierung
"""
import numpy as np
import pandas as pd
import pytest

from optimizer.indicators.preprocessing import (
    get_frac_diff_weights,
    frac_diff,
    find_min_d_for_stationarity,
    apply_frac_diff_preprocessing,
    apply_log_returns_preprocessing,
    apply_normalize_preprocessing,
)


# === FIXTURES ===

@pytest.fixture
def sample_ohlc():
    """Erstellt Sample OHLC-Daten für Tests."""
    np.random.seed(42)
    n = 500
    # Random Walk für realistische Preise
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


@pytest.fixture
def trending_series():
    """Erstellt eine trending Zeitreihe (nicht stationär)."""
    np.random.seed(42)
    n = 500
    trend = np.linspace(100, 150, n)
    noise = np.random.randn(n) * 2
    return trend + noise


@pytest.fixture
def stationary_series():
    """Erstellt eine stationäre Zeitreihe."""
    np.random.seed(42)
    return np.random.randn(500) * 10 + 100


# === TESTS FÜR FRAC_DIFF_WEIGHTS ===

class TestFracDiffWeights:
    """Tests für get_frac_diff_weights()."""

    def test_weights_first_element_is_one(self):
        """Erstes Gewicht sollte 1 sein."""
        weights = get_frac_diff_weights(d=0.5, size=100)
        # Das letzte Element im Array ist das erste Gewicht (nach Umkehrung)
        assert weights[-1] == 1.0

    def test_weights_are_decreasing(self):
        """Gewichte sollten in Absolutwert abnehmen."""
        weights = get_frac_diff_weights(d=0.5, size=100)
        abs_weights = np.abs(weights)
        # Vom neuesten zum ältesten (reversed)
        for i in range(len(abs_weights) - 1):
            assert abs_weights[i] <= abs_weights[i + 1]

    def test_weights_have_expected_pattern(self):
        """Gewichte sollten ein vorhersagbares Muster haben."""
        weights = get_frac_diff_weights(d=0.5, size=100)
        # First weight (newest) should be 1.0
        assert weights[-1] == 1.0
        # All weights should be finite
        assert np.all(np.isfinite(weights))
        # Second weight for d=0.5: w_1 = -d/1 = -0.5
        assert abs(weights[-2] - (-0.5)) < 0.01

    def test_d_0_returns_single_weight(self):
        """d=0 sollte nur das erste Gewicht [1] zurückgeben."""
        weights = get_frac_diff_weights(d=0, size=100, threshold=1e-10)
        assert len(weights) == 1
        assert weights[0] == 1.0

    def test_threshold_affects_length(self):
        """Höherer threshold sollte weniger Gewichte geben."""
        weights_low = get_frac_diff_weights(d=0.5, size=1000, threshold=1e-6)
        weights_high = get_frac_diff_weights(d=0.5, size=1000, threshold=1e-2)
        assert len(weights_high) < len(weights_low)


# === TESTS FÜR FRAC_DIFF ===

class TestFracDiff:
    """Tests für frac_diff()."""

    def test_output_length_matches_input(self):
        """Output-Länge sollte Input-Länge entsprechen."""
        series = np.random.randn(100) + 100
        result = frac_diff(series, d=0.5)
        assert len(result) == len(series)

    def test_contains_nan_at_start(self):
        """Anfang sollte NaN enthalten (Warmup-Periode)."""
        series = np.random.randn(100) + 100
        result = frac_diff(series, d=0.5)
        assert np.isnan(result[0])

    def test_later_values_are_not_nan(self):
        """Spätere Werte sollten nicht NaN sein."""
        series = np.random.randn(100) + 100
        result = frac_diff(series, d=0.5, threshold=1e-2)
        # Nach Warmup sollten Werte existieren
        assert not np.isnan(result[-1])

    def test_d_1_reduces_memory(self):
        """d=1 sollte Stationarität erreichen (ähnlich erste Differenz)."""
        # Trending series
        np.random.seed(42)
        series = np.linspace(100, 150, 200) + np.random.randn(200) * 0.5
        result = frac_diff(series, d=1.0, threshold=1e-4)
        valid = result[~np.isnan(result)]
        # Result should be more stationary (lower std of mean over windows)
        assert len(valid) > 50

    def test_d_0_preserves_series(self):
        """d=0 sollte Serie unverändert lassen."""
        series = np.array([100, 102, 101, 105, 103], dtype=float)
        result = frac_diff(series, d=0, threshold=1e-10)
        np.testing.assert_array_almost_equal(result, series, decimal=5)


# === TESTS FÜR FIND_MIN_D ===

class TestFindMinD:
    """Tests für find_min_d_for_stationarity()."""

    def test_returns_d_between_0_and_1(self, trending_series):
        """Sollte d zwischen 0 und 1 zurückgeben."""
        d = find_min_d_for_stationarity(trending_series)
        assert 0 <= d <= 1

    def test_returns_small_d_for_nearly_stationary(self, stationary_series):
        """Für fast stationäre Daten sollte d klein sein."""
        d = find_min_d_for_stationarity(stationary_series)
        assert d <= 0.3

    def test_returns_default_for_short_series(self):
        """Kurze Serien sollten Default 0.5 zurückgeben."""
        short_series = np.random.randn(50)
        d = find_min_d_for_stationarity(short_series)
        assert d == 0.5


# === TESTS FÜR APPLY_FRAC_DIFF_PREPROCESSING ===

class TestApplyFracDiffPreprocessing:
    """Tests für apply_frac_diff_preprocessing()."""

    def test_preserves_original_close(self, sample_ohlc):
        """Sollte _original_close Spalte erstellen."""
        df = sample_ohlc.copy()
        result_df, d = apply_frac_diff_preprocessing(df, use_auto_d=False, default_d=0.4)
        assert "_original_close" in result_df.columns

    def test_transforms_ohlc_columns(self, sample_ohlc):
        """Alle OHLC-Spalten sollten transformiert werden."""
        df = sample_ohlc.copy()
        original_close = df["C"].copy()
        result_df, d = apply_frac_diff_preprocessing(df, use_auto_d=False, default_d=0.4)

        # Werte sollten sich unterscheiden
        assert not np.allclose(
            result_df["C"].dropna().values[:10],
            original_close.values[:10],
            rtol=0.1
        )

    def test_returns_d_value(self, sample_ohlc):
        """Sollte verwendeten d-Wert zurückgeben."""
        df = sample_ohlc.copy()
        result_df, d = apply_frac_diff_preprocessing(df, use_auto_d=False, default_d=0.35)
        assert d == 0.35


# === TESTS FÜR APPLY_LOG_RETURNS_PREPROCESSING ===

class TestApplyLogReturnsPreprocessing:
    """Tests für apply_log_returns_preprocessing()."""

    def test_preserves_original_close(self, sample_ohlc):
        """Sollte _original_close Spalte erstellen."""
        df = sample_ohlc.copy()
        result_df = apply_log_returns_preprocessing(df)
        assert "_original_close" in result_df.columns

    def test_output_is_log_returns(self, sample_ohlc):
        """Output sollte Log-Returns sein."""
        df = sample_ohlc.copy()
        original = df["C"].copy()
        result_df = apply_log_returns_preprocessing(df)

        # Manuell berechnete Log-Returns
        expected = np.log(original / original.shift(1)).dropna()
        actual = result_df["C"]

        np.testing.assert_array_almost_equal(
            actual.values[:10],
            expected.values[:10],
            decimal=10
        )

    def test_removes_first_row(self, sample_ohlc):
        """Erste Zeile sollte entfernt werden (NaN durch shift)."""
        df = sample_ohlc.copy()
        original_len = len(df)
        result_df = apply_log_returns_preprocessing(df)
        assert len(result_df) == original_len - 1


# === TESTS FÜR APPLY_NORMALIZE_PREPROCESSING ===

class TestApplyNormalizePreprocessing:
    """Tests für apply_normalize_preprocessing()."""

    def test_output_is_zscore_normalized(self, sample_ohlc):
        """Output sollte Z-Score normalisiert sein."""
        df = sample_ohlc.copy()
        window = 50
        result_df = apply_normalize_preprocessing(df, window=window)

        # Z-Score sollte mean ~0 und std ~1 haben
        close_values = result_df["C"].dropna()
        assert abs(close_values.mean()) < 0.5  # Nahe 0
        assert 0.5 < close_values.std() < 1.5  # Nahe 1

    def test_removes_warmup_period(self, sample_ohlc):
        """Warmup-Periode sollte entfernt werden."""
        df = sample_ohlc.copy()
        original_len = len(df)
        window = 100
        result_df = apply_normalize_preprocessing(df, window=window)
        assert len(result_df) == original_len - window

    def test_preserves_original_close_if_not_exists(self, sample_ohlc):
        """Sollte _original_close erstellen wenn nicht vorhanden."""
        df = sample_ohlc.copy()
        result_df = apply_normalize_preprocessing(df)
        assert "_original_close" in result_df.columns
