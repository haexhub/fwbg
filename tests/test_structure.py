"""
Tests für Struktur-bezogene Features.

Testet:
- FFT Features (Fourier-Analyse)
- Path Efficiency / Fractal Dimension
- Convexity Features
- Event Features (Time-Since-Event)
- VWAP Features

NOTE: Diese Tests müssen auf das neue Plugin-basierte Indicator-System angepasst werden.
"""
import numpy as np
import pandas as pd
import pytest

# Alte Imports - neue Struktur verwendet Plugin-System
pytest.skip("Tests need migration to new indicator plugin system", allow_module_level=True)

from fwbg.builtins.indicators.structure import (
    _bars_since_event,
    compute_fft_features,
    compute_event_features,
    compute_path_efficiency,
    compute_convexity_features,
    compute_vwap_features,
)


# === FIXTURES ===

@pytest.fixture
def sample_ohlc():
    """Erstellt Sample OHLC-Daten."""
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


@pytest.fixture
def sinusoidal_series():
    """Erstellt eine sinusförmige Zeitreihe für FFT-Tests."""
    n = 256
    t = np.arange(n)
    # Kombination aus zwei Frequenzen
    freq1, freq2 = 0.05, 0.02
    series = 100 + 5 * np.sin(2 * np.pi * freq1 * t) + 3 * np.sin(2 * np.pi * freq2 * t)
    return pd.DataFrame({
        'C': series,
        'H': series * 1.01,
        'L': series * 0.99,
        'O': series,
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))


@pytest.fixture
def trending_ohlc():
    """Erstellt stark trending OHLC-Daten (hohe Path Efficiency)."""
    n = 200
    # Starker Aufwärtstrend
    close = np.linspace(100, 150, n) + np.random.randn(n) * 0.5
    high = close * 1.005
    low = close * 0.995
    open_price = close - 0.1

    return pd.DataFrame({
        'O': open_price,
        'H': high,
        'L': low,
        'C': close,
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))


@pytest.fixture
def ranging_ohlc():
    """Erstellt ranging OHLC-Daten (niedrige Path Efficiency)."""
    np.random.seed(42)
    n = 200
    # Mean-reverting um 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5) * 0
    close = 100 + np.random.randn(n) * 5  # Einfach Noise um 100
    high = close + abs(np.random.randn(n) * 0.5)
    low = close - abs(np.random.randn(n) * 0.5)
    open_price = close + np.random.randn(n) * 0.1

    return pd.DataFrame({
        'O': open_price,
        'H': high,
        'L': low,
        'C': close,
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))


# === TESTS FÜR _BARS_SINCE_EVENT ===

class TestBarsSinceEvent:
    """Tests für _bars_since_event()."""

    def test_counts_correctly(self):
        """Sollte korrekt zählen."""
        events = pd.Series([0, 0, 1, 0, 0, 0, 1, 0])
        result = _bars_since_event(events)

        # Nach Event bei Index 2: 0, 1, 2, 3
        # Nach Event bei Index 6: 0, 1
        assert result.iloc[3] == 1  # 1 Bar nach Event bei Index 2
        assert result.iloc[4] == 2  # 2 Bars nach Event
        assert result.iloc[7] == 1  # 1 Bar nach Event bei Index 6

    def test_nan_before_first_event(self):
        """Vor erstem Event sollte NaN sein."""
        events = pd.Series([0, 0, 0, 1, 0, 0])
        result = _bars_since_event(events)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])

    def test_resets_at_each_event(self):
        """Zähler sollte bei jedem Event zurücksetzen."""
        events = pd.Series([1, 0, 0, 1, 0, 1, 0, 0, 0])
        result = _bars_since_event(events)

        assert result.iloc[1] == 1
        assert result.iloc[2] == 2
        assert result.iloc[4] == 1  # Reset nach Event bei Index 3
        assert result.iloc[6] == 1  # Reset nach Event bei Index 5

    def test_handles_empty_series(self):
        """Sollte leere Series behandeln."""
        events = pd.Series([], dtype=int)
        result = _bars_since_event(events)
        assert len(result) == 0

    def test_handles_no_events(self):
        """Sollte Series ohne Events behandeln."""
        events = pd.Series([0, 0, 0, 0])
        result = _bars_since_event(events)
        # Result should have same length
        assert len(result) == len(events)


# === TESTS FÜR COMPUTE_FFT_FEATURES ===

class TestComputeFftFeatures:
    """Tests für compute_fft_features()."""

    def test_adds_fft_columns(self, sample_ohlc):
        """Sollte FFT-Spalten hinzufügen."""
        df = sample_ohlc.copy()
        window = 64
        result = compute_fft_features(df, window)

        assert f"fft_dom_freq_{window}" in result.columns
        assert f"fft_dom_power_{window}" in result.columns
        assert f"fft_energy_{window}" in result.columns
        assert f"fft_entropy_{window}" in result.columns
        assert f"fft_lowfreq_{window}" in result.columns

    def test_values_in_valid_range(self, sample_ohlc):
        """FFT-Werte sollten in gültigem Bereich sein."""
        df = sample_ohlc.copy()
        result = compute_fft_features(df, 64)

        # Dominant power sollte zwischen 0 und 1 sein
        dom_power = result["fft_dom_power_64"].dropna()
        assert all(0 <= v <= 1 for v in dom_power)

        # Low freq ratio sollte zwischen 0 und 1 sein
        low_freq = result["fft_lowfreq_64"].dropna()
        assert all(0 <= v <= 1 for v in low_freq)

    def test_detects_dominant_frequency(self, sinusoidal_series):
        """Sollte dominante Frequenz in sinusförmiger Serie erkennen."""
        df = sinusoidal_series.copy()
        result = compute_fft_features(df, 128)

        # Dominante Frequenz sollte nahe 0.05 oder 0.02 sein
        dom_freq = result["fft_dom_freq_128"].dropna().mean()
        assert 0.01 < dom_freq < 0.1

    def test_nan_at_start(self, sample_ohlc):
        """Anfang sollte NaN haben (Warmup)."""
        df = sample_ohlc.copy()
        result = compute_fft_features(df, 64)

        # Erste 64 Werte sollten NaN sein
        assert pd.isna(result["fft_dom_freq_64"].iloc[0])
        assert pd.isna(result["fft_dom_freq_64"].iloc[63])
        assert not pd.isna(result["fft_dom_freq_64"].iloc[64])


# === TESTS FÜR COMPUTE_EVENT_FEATURES ===

class TestComputeEventFeatures:
    """Tests für compute_event_features()."""

    def test_adds_event_columns(self, sample_ohlc):
        """Sollte Event-Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_event_features(df)

        assert "event_bars_since_high_20" in result.columns
        assert "event_bars_since_low_20" in result.columns
        assert "event_bars_since_high_50" in result.columns
        assert "event_bars_since_low_50" in result.columns
        assert "event_bars_since_ema_cross" in result.columns
        assert "event_bars_since_rsi_extreme" in result.columns
        assert "event_bars_since_vol_spike" in result.columns

    def test_adds_log_transformed_columns(self, sample_ohlc):
        """Sollte log-transformierte Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_event_features(df)

        assert "event_bars_since_high_20_log" in result.columns
        assert "event_bars_since_low_20_log" in result.columns

    def test_values_are_non_negative(self, sample_ohlc):
        """Bars-since Werte sollten nicht-negativ sein."""
        df = sample_ohlc.copy()
        result = compute_event_features(df)

        for col in result.columns:
            if col.startswith("event_bars_since_"):
                values = result[col].dropna()
                assert all(v >= 0 for v in values)

    def test_log_values_are_correct(self, sample_ohlc):
        """Log-Werte sollten korrekt berechnet sein."""
        df = sample_ohlc.copy()
        result = compute_event_features(df)

        bars = result["event_bars_since_high_20"].dropna()
        bars_log = result["event_bars_since_high_20_log"].dropna()

        # log1p(x) sollte angewendet sein
        expected = np.log1p(bars)
        np.testing.assert_array_almost_equal(bars_log.values, expected.values)


# === TESTS FÜR COMPUTE_PATH_EFFICIENCY ===

class TestComputePathEfficiency:
    """Tests für compute_path_efficiency()."""

    def test_adds_path_efficiency_columns(self, sample_ohlc):
        """Sollte Path Efficiency Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_path_efficiency(df)

        assert "path_efficiency_10" in result.columns
        assert "path_efficiency_20" in result.columns
        assert "path_efficiency_50" in result.columns
        assert "path_efficiency_100" in result.columns

    def test_adds_fractal_dimension_columns(self, sample_ohlc):
        """Sollte Fractal Dimension Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_path_efficiency(df)

        assert "fractal_dim_10" in result.columns
        assert "fractal_dim_50" in result.columns

    def test_path_efficiency_between_0_and_1(self, sample_ohlc):
        """Path Efficiency sollte zwischen 0 und 1 sein."""
        df = sample_ohlc.copy()
        result = compute_path_efficiency(df)

        pe_20 = result["path_efficiency_20"].dropna()
        assert all(0 <= v <= 1 for v in pe_20)

    def test_trending_has_higher_efficiency_than_ranging(self, trending_ohlc, ranging_ohlc):
        """Trending Serie sollte höhere Path Efficiency haben als Ranging."""
        result_trend = compute_path_efficiency(trending_ohlc.copy())
        result_range = compute_path_efficiency(ranging_ohlc.copy())

        pe_trend = result_trend["path_efficiency_50"].dropna().mean()
        pe_range = result_range["path_efficiency_50"].dropna().mean()
        assert pe_trend > pe_range, f"Trend PE ({pe_trend:.2f}) should be > Range PE ({pe_range:.2f})"

    def test_path_efficiency_in_valid_range(self, ranging_ohlc):
        """Path Efficiency sollte zwischen 0 und 1 sein."""
        df = ranging_ohlc.copy()
        result = compute_path_efficiency(df)

        pe_50 = result["path_efficiency_50"].dropna()
        assert all(0 <= v <= 1 for v in pe_50), "PE should be between 0 and 1"

    def test_fractal_dim_formula_correct(self, sample_ohlc):
        """Fractal Dimension sollte 1 + (1 - PE) sein."""
        df = sample_ohlc.copy()
        result = compute_path_efficiency(df)

        pe = result["path_efficiency_20"].dropna()
        fd = result["fractal_dim_20"].dropna()

        expected_fd = 1 + (1 - pe)
        np.testing.assert_array_almost_equal(fd.values, expected_fd.values)

    def test_adds_change_columns(self, sample_ohlc):
        """Sollte Change-Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_path_efficiency(df)

        assert "path_efficiency_20_chg" in result.columns
        assert "path_efficiency_50_chg" in result.columns


# === TESTS FÜR COMPUTE_CONVEXITY_FEATURES ===

class TestComputeConvexityFeatures:
    """Tests für compute_convexity_features()."""

    def test_adds_convexity_columns(self, sample_ohlc):
        """Sollte Convexity Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_convexity_features(df)

        assert "convex_ema_21" in result.columns
        assert "convex_ema_50" in result.columns
        assert "convex_ema_21_smooth" in result.columns
        assert "convex_ema_50_smooth" in result.columns

    def test_adds_divergence_column(self, sample_ohlc):
        """Sollte Divergenz-Spalte hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_convexity_features(df)

        assert "convex_divergence" in result.columns

    def test_adds_zscore_column(self, sample_ohlc):
        """Sollte Z-Score Spalte hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_convexity_features(df)

        assert "convex_zscore" in result.columns

    def test_accelerating_trend_has_positive_convexity(self):
        """Beschleunigender Trend sollte positive Convexity haben."""
        n = 200
        # Parabolischer Anstieg (beschleunigend)
        t = np.arange(n)
        close = 100 + 0.01 * t ** 2

        df = pd.DataFrame({
            'C': close,
            'H': close * 1.01,
            'L': close * 0.99,
            'O': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))

        result = compute_convexity_features(df)
        convex = result["convex_ema_21"].dropna()

        # Sollte überwiegend positiv sein
        assert convex.mean() > 0


# === TESTS FÜR COMPUTE_VWAP_FEATURES ===

class TestComputeVwapFeatures:
    """Tests für compute_vwap_features()."""

    def test_adds_vwap_distance_columns(self, sample_ohlc):
        """Sollte VWAP Distance Spalten hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_vwap_features(df)

        assert "structure_vwap_dist_20" in result.columns
        assert "structure_vwap_dist_50" in result.columns
        assert "structure_vwap_dist_100" in result.columns

    def test_adds_time_above_column(self, sample_ohlc):
        """Sollte Time Above VWAP Spalte hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_vwap_features(df)

        assert "structure_vwap_time_above" in result.columns

    def test_adds_bars_since_cross_column(self, sample_ohlc):
        """Sollte Bars Since VWAP Cross Spalte hinzufügen."""
        df = sample_ohlc.copy()
        result = compute_vwap_features(df)

        assert "structure_bars_since_vwap_cross" in result.columns

    def test_time_above_between_0_and_1(self, sample_ohlc):
        """Time above VWAP sollte zwischen 0 und 1 sein."""
        df = sample_ohlc.copy()
        result = compute_vwap_features(df)

        time_above = result["structure_vwap_time_above"].dropna()
        assert all(0 <= v <= 1 for v in time_above)

    def test_trending_up_mostly_above_vwap(self, trending_ohlc):
        """Aufwärtstrend sollte meist über VWAP sein."""
        df = trending_ohlc.copy()
        result = compute_vwap_features(df)

        time_above = result["structure_vwap_time_above"].dropna().mean()
        assert time_above > 0.5, f"Uptrend should be mostly above VWAP, got {time_above}"

    def test_vwap_distance_formula_correct(self, sample_ohlc):
        """VWAP Distance Formel sollte korrekt sein."""
        df = sample_ohlc.copy()
        result = compute_vwap_features(df)

        # Manuell berechnen
        tp = (df["H"] + df["L"] + df["C"]) / 3
        vwap_20 = tp.rolling(20).mean()
        expected_dist = (df["C"] - vwap_20) / vwap_20

        actual_dist = result["structure_vwap_dist_20"]

        np.testing.assert_array_almost_equal(
            actual_dist.dropna().values[-10:],
            expected_dist.dropna().values[-10:],
            decimal=10
        )
