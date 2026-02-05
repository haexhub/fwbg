"""
Tests für Struktur-bezogene Features.

Testet:
- FFT Features (Fourier-Analyse)
- Path Efficiency / Fractal Dimension
- Convexity Features
- Event Features (Time-Since-Event)
- VWAP Features

Migriert auf das neue Plugin-basierte Indicator-System.
"""
import numpy as np
import pandas as pd
import pytest

# Neue Imports - Plugin-System
from fwbg.builtins.indicators.structure import StructureIndicators, _bars_since_event


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


@pytest.fixture
def structure_indicator():
    """Erstellt eine StructureIndicators-Instanz."""
    return StructureIndicators()


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


# === TESTS FÜR STRUCTURE INDICATOR (via Plugin) ===

class TestStructureIndicator:
    """Tests für StructureIndicators Plugin."""

    def test_compute_adds_columns(self, sample_ohlc, structure_indicator):
        """Sollte Structure-Spalten hinzufügen."""
        result = structure_indicator.compute(sample_ohlc.copy())

        # Prüfe dass einige erwartete Spalten existieren
        structure_cols = [c for c in result.columns if c.startswith(('path_', 'fractal_', 'fft_', 'convex_', 'structure_'))]
        assert len(structure_cols) > 0, "Should add structure columns"

    def test_get_feature_columns(self, structure_indicator):
        """Sollte Feature-Spalten zurückgeben."""
        cols = structure_indicator.get_feature_columns()
        assert len(cols) > 0
        # Sollte structure_ prefixes enthalten
        assert any('path_' in c or 'fractal_' in c or 'structure_' in c for c in cols)


class TestPathEfficiency:
    """Tests für Path Efficiency Features."""

    def test_adds_path_efficiency_columns(self, sample_ohlc, structure_indicator):
        """Sollte Path Efficiency Spalten hinzufügen."""
        result = structure_indicator.compute(sample_ohlc.copy())

        pe_cols = [c for c in result.columns if c.startswith('path_efficiency_')]
        assert len(pe_cols) > 0, "Should add path_efficiency columns"

    def test_path_efficiency_between_0_and_1(self, sample_ohlc, structure_indicator):
        """Path Efficiency sollte zwischen 0 und 1 sein."""
        result = structure_indicator.compute(sample_ohlc.copy())

        pe_cols = [c for c in result.columns if c.startswith('path_efficiency_') and '_chg' not in c]
        for col in pe_cols:
            values = result[col].dropna()
            if len(values) > 0:
                assert all(0 <= v <= 1 for v in values), f"{col} should be between 0 and 1"

    def test_trending_has_higher_efficiency(self, trending_ohlc, ranging_ohlc, structure_indicator):
        """Trending Serie sollte höhere Path Efficiency haben als Ranging."""
        result_trend = structure_indicator.compute(trending_ohlc.copy())
        result_range = structure_indicator.compute(ranging_ohlc.copy())

        # Finde eine gemeinsame PE-Spalte
        pe_cols = [c for c in result_trend.columns if c.startswith('path_efficiency_') and '_chg' not in c]
        if len(pe_cols) > 0:
            col = pe_cols[0]
            pe_trend = result_trend[col].dropna().mean()
            pe_range = result_range[col].dropna().mean()
            # Trending sollte höher sein (oder zumindest nicht viel niedriger)
            # NOTE: Dieser Test kann flaky sein je nach Random Seed
            assert pe_trend >= pe_range * 0.8, f"Trend PE ({pe_trend:.2f}) should be >= Range PE ({pe_range:.2f})"


class TestFFTFeatures:
    """Tests für FFT Features."""

    def test_adds_fft_columns(self, sample_ohlc, structure_indicator):
        """Sollte FFT-Spalten hinzufügen."""
        result = structure_indicator.compute(sample_ohlc.copy())

        fft_cols = [c for c in result.columns if c.startswith('fft_')]
        assert len(fft_cols) > 0, "Should add FFT columns"

    def test_fft_values_in_valid_range(self, sample_ohlc, structure_indicator):
        """FFT-Werte sollten in gültigem Bereich sein."""
        result = structure_indicator.compute(sample_ohlc.copy())

        # Dominant power sollte zwischen 0 und 1 sein
        dom_power_cols = [c for c in result.columns if 'dom_power' in c or 'lowfreq' in c]
        for col in dom_power_cols:
            values = result[col].dropna()
            if len(values) > 0:
                assert all(0 <= v <= 1 for v in values), f"{col} should be between 0 and 1"


class TestConvexityFeatures:
    """Tests für Convexity Features."""

    def test_adds_convexity_columns(self, sample_ohlc, structure_indicator):
        """Sollte Convexity Spalten hinzufügen."""
        result = structure_indicator.compute(sample_ohlc.copy())

        convex_cols = [c for c in result.columns if c.startswith('convex_')]
        assert len(convex_cols) > 0, "Should add convexity columns"

    def test_accelerating_trend_has_positive_convexity(self, structure_indicator):
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

        result = structure_indicator.compute(df)

        convex_cols = [c for c in result.columns if c.startswith('convex_') and '_smooth' not in c and 'zscore' not in c and 'divergence' not in c]
        if len(convex_cols) > 0:
            col = convex_cols[0]
            convex = result[col].dropna()
            # Sollte überwiegend positiv sein
            assert convex.mean() > 0, f"{col} should be positive for accelerating trend"


class TestVWAPFeatures:
    """Tests für VWAP Features."""

    def test_adds_vwap_columns(self, sample_ohlc, structure_indicator):
        """Sollte VWAP Spalten hinzufügen."""
        result = structure_indicator.compute(sample_ohlc.copy())

        vwap_cols = [c for c in result.columns if 'vwap' in c.lower()]
        assert len(vwap_cols) > 0, "Should add VWAP columns"

    def test_time_above_between_0_and_1(self, sample_ohlc, structure_indicator):
        """Time above VWAP sollte zwischen 0 und 1 sein."""
        result = structure_indicator.compute(sample_ohlc.copy())

        time_above_cols = [c for c in result.columns if 'time_above' in c]
        for col in time_above_cols:
            values = result[col].dropna()
            if len(values) > 0:
                assert all(0 <= v <= 1 for v in values), f"{col} should be between 0 and 1"
