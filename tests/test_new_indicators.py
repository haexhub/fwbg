"""
Tests für neue Indicator Plugins.

- MicrostructureIndicator
- MacroSurpriseIndicator

Fokus auf Edge Cases und realistisches Verhalten.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# === Test Fixtures ===


@pytest.fixture
def sample_ohlcv_df():
    """Standard OHLCV DataFrame für Tests."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    # Realistische Preis-Generierung
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.001)
    high = close + np.abs(np.random.randn(n)) * 0.0015
    low = close - np.abs(np.random.randn(n)) * 0.0015
    open_ = close + np.random.randn(n) * 0.0008

    # High muss >= max(O, C), Low <= min(O, C)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.randint(100, 10000, n).astype(float)

    return pd.DataFrame({
        "O": open_,
        "H": high,
        "L": low,
        "C": close,
        "V": volume,
    }, index=dates)


@pytest.fixture
def doji_df():
    """DataFrame mit Doji-Kerzen (O ≈ C)."""
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    close = np.full(n, 1.1000)
    open_ = close.copy()  # Doji: Open = Close
    high = close + 0.001
    low = close - 0.001

    return pd.DataFrame({
        "O": open_,
        "H": high,
        "L": low,
        "C": close,
        "V": np.full(n, 1000.0),
    }, index=dates)


@pytest.fixture
def gap_df():
    """DataFrame mit Gaps zwischen Sessions."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    close = np.zeros(n)
    open_ = np.zeros(n)
    close[0] = 1.1000

    for i in range(1, n):
        # Jede 24. Kerze hat einen Gap
        if i % 24 == 0:
            gap = np.random.choice([-0.005, 0.005])  # 50 pips gap
            open_[i] = close[i-1] + gap
        else:
            open_[i] = close[i-1] + np.random.randn() * 0.0002

        close[i] = open_[i] + np.random.randn() * 0.001

    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * 0.0005
    low = np.minimum(open_, close) - np.abs(np.random.randn(n)) * 0.0005

    return pd.DataFrame({
        "O": open_,
        "H": high,
        "L": low,
        "C": close,
        "V": np.full(n, 1000.0),
    }, index=dates)


@pytest.fixture
def zero_range_df():
    """DataFrame mit Zero-Range Bars (H = L)."""
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")

    price = np.linspace(1.1, 1.15, n)

    return pd.DataFrame({
        "O": price,
        "H": price,  # H = L = O = C
        "L": price,
        "C": price,
        "V": np.full(n, 1000.0),
    }, index=dates)


@pytest.fixture
def no_volume_df():
    """DataFrame ohne Volume-Daten."""
    n = 50
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    np.random.seed(42)

    close = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
    high = close + 0.001
    low = close - 0.001

    return pd.DataFrame({
        "O": close - 0.0005,
        "H": high,
        "L": low,
        "C": close,
        # Kein V
    }, index=dates)


# === Microstructure Indicator Tests ===


class TestMicrostructureIndicator:
    """Tests für MicrostructureIndicator."""

    def test_basic_computation(self, sample_ohlcv_df):
        """Basisberechnung sollte alle Features produzieren."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        indicator = MicrostructureIndicator()
        df = indicator.compute(sample_ohlcv_df.copy())

        # Alle Features sollten existieren
        for col in indicator.get_feature_columns():
            assert col in df.columns, f"Missing column: {col}"

    def test_wick_imbalance_range(self, sample_ohlcv_df):
        """Wick Imbalance sollte zwischen -1 und 1 liegen."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        indicator = MicrostructureIndicator()
        df = indicator.compute(sample_ohlcv_df.copy())

        imbalance = df["micro_wick_imbalance"].dropna()
        assert imbalance.min() >= -1.0001
        assert imbalance.max() <= 1.0001

    def test_intrabar_bias_range(self, sample_ohlcv_df):
        """Intrabar Bias sollte zwischen -1 und 1 liegen."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        indicator = MicrostructureIndicator()
        df = indicator.compute(sample_ohlcv_df.copy())

        bias = df["micro_intrabar_bias"].dropna()
        assert bias.min() >= -1.0001
        assert bias.max() <= 1.0001

    def test_body_ratio_range(self, sample_ohlcv_df):
        """Body Ratio sollte zwischen 0 und 1 liegen."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        indicator = MicrostructureIndicator()
        df = indicator.compute(sample_ohlcv_df.copy())

        ratio = df["micro_body_ratio"].dropna()
        assert ratio.min() >= -0.0001
        assert ratio.max() <= 1.0001

    def test_doji_handling(self, doji_df):
        """Doji-Kerzen (O=C) sollten Body Ratio = 0 haben."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        indicator = MicrostructureIndicator()
        df = indicator.compute(doji_df.copy())

        # Bei Doji: Body = 0, also Body Ratio = 0
        body_ratio = df["micro_body_ratio"].dropna()
        assert (body_ratio == 0).all()

    def test_zero_range_handling(self, zero_range_df):
        """Zero-Range Bars sollten NaN produzieren (Division durch 0)."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        indicator = MicrostructureIndicator()
        df = indicator.compute(zero_range_df.copy())

        # Bei H=L ist Range=0, Features sollten NaN sein
        assert df["micro_wick_imbalance"].isna().all()
        assert df["micro_intrabar_bias"].isna().all()

    def test_volume_weighted_with_volume(self, sample_ohlcv_df):
        """VWAP Pressure sollte mit Volume berechnet werden."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        indicator = MicrostructureIndicator()
        df = indicator.compute(sample_ohlcv_df.copy())

        # VWAP Pressure sollte existieren und nicht konstant sein
        vwap = df["micro_vwap_pressure"].dropna()
        assert len(vwap) > 0
        assert vwap.std() > 0  # Sollte variieren

    def test_volume_weighted_without_volume(self, no_volume_df):
        """Ohne Volume sollte Fallback greifen."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        # Füge leere V-Spalte hinzu
        no_volume_df["V"] = 0.0

        indicator = MicrostructureIndicator()
        df = indicator.compute(no_volume_df.copy())

        # Relative Volume sollte Fallback (1.0) sein
        assert (df["micro_relative_volume"] == 1.0).all()

    def test_custom_parameters(self, sample_ohlcv_df):
        """Benutzerdefinierte Parameter sollten funktionieren."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        indicator = MicrostructureIndicator()
        df = indicator.compute(
            sample_ohlcv_df.copy(),
            atr_period=7,
            rolling_window=10,
        )

        # Features sollten trotzdem existieren
        assert "micro_wick_imbalance_sum" in df.columns
        assert "micro_pressure_sum" in df.columns

    def test_pressure_score_sign(self, sample_ohlcv_df):
        """Pressure Score sollte Vorzeichen von (C-O) haben."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        indicator = MicrostructureIndicator()
        df = indicator.compute(sample_ohlcv_df.copy())

        # Bullish Bar (C > O) sollte positive Pressure haben
        bullish_mask = df["C"] > df["O"]
        if bullish_mask.any():
            bullish_pressure = df.loc[bullish_mask, "micro_pressure_score"].dropna()
            assert (bullish_pressure >= 0).all()

        # Bearish Bar (C < O) sollte negative Pressure haben
        bearish_mask = df["C"] < df["O"]
        if bearish_mask.any():
            bearish_pressure = df.loc[bearish_mask, "micro_pressure_score"].dropna()
            assert (bearish_pressure <= 0).all()

    def test_rolling_sum_accumulation(self, sample_ohlcv_df):
        """Rolling Sums sollten korrekt akkumulieren."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        indicator = MicrostructureIndicator()
        window = 5
        df = indicator.compute(sample_ohlcv_df.copy(), rolling_window=window)

        # Manueller Check für einen Punkt
        idx = 50
        wick_sum_computed = df.loc[df.index[idx], "micro_wick_imbalance_sum"]
        wick_values = df["micro_wick_imbalance"].iloc[idx-window+1:idx+1]
        wick_sum_expected = wick_values.sum()

        assert abs(wick_sum_computed - wick_sum_expected) < 0.0001

    def test_default_params(self):
        """Default-Parameter sollten gesetzt sein."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        params = MicrostructureIndicator.get_default_params()
        assert "atr_period" in params
        assert "rolling_window" in params
        assert params["atr_period"] == 14
        assert params["rolling_window"] == 5


# === Macro Surprise Indicator Tests ===


class TestMacroSurpriseIndicator:
    """Tests für MacroSurpriseIndicator."""

    def test_basic_computation(self, sample_ohlcv_df):
        """Basisberechnung sollte alle Features produzieren."""
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        indicator = MacroSurpriseIndicator()
        df = indicator.compute(sample_ohlcv_df.copy())

        # Alle Features sollten existieren
        for col in indicator.get_feature_columns():
            assert col in df.columns, f"Missing column: {col}"

    def test_gap_detection(self, gap_df):
        """Gaps sollten korrekt erkannt werden."""
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        indicator = MacroSurpriseIndicator()
        df = indicator.compute(gap_df.copy())

        # Es sollten Gaps erkannt werden
        gap_up = df["macro_gap_up"].sum()
        gap_down = df["macro_gap_down"].sum()
        assert gap_up > 0 or gap_down > 0

    def test_gap_filled_calculation(self, gap_df):
        """Gap Filled sollte korrekt berechnet werden."""
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        indicator = MacroSurpriseIndicator()
        df = indicator.compute(gap_df.copy())

        # Gap Filled sollte 0 oder 1 sein
        gap_filled = df["macro_gap_filled"].dropna()
        assert gap_filled.isin([0.0, 1.0]).all()

    def test_return_decomposition(self, sample_ohlcv_df):
        """Return-Zerlegung sollte konsistent sein."""
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        indicator = MacroSurpriseIndicator()
        df = indicator.compute(sample_ohlcv_df.copy())

        # Overnight + Intraday ≈ Total (mit Rundungsfehlern)
        overnight = df["macro_overnight_return"].dropna()
        intraday = df["macro_intraday_return"].dropna()
        total = df["macro_total_return"].dropna()

        # Indizes ausrichten
        common_idx = overnight.index.intersection(intraday.index).intersection(total.index)
        if len(common_idx) > 10:
            # Approximation: overnight + intraday ≈ total (nicht exakt wegen Compound)
            combined = overnight.loc[common_idx] + intraday.loc[common_idx]
            # Grobe Prüfung
            correlation = combined.corr(total.loc[common_idx])
            assert correlation > 0.9

    def test_surprise_detection(self, sample_ohlcv_df):
        """Surprise sollte bei extremen Moves = 1 sein."""
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        indicator = MacroSurpriseIndicator()
        df = indicator.compute(sample_ohlcv_df.copy(), surprise_threshold=2.0)

        # is_surprise sollte binär sein
        is_surprise = df["macro_is_surprise"].dropna()
        assert is_surprise.isin([0.0, 1.0]).all()

    def test_zscore_range(self, sample_ohlcv_df):
        """Return ZScore sollte plausible Werte haben."""
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        indicator = MacroSurpriseIndicator()
        df = indicator.compute(sample_ohlcv_df.copy())

        zscore = df["macro_return_zscore"].dropna()
        # ZScore sollte meistens zwischen -4 und 4 liegen
        assert zscore.median() < 4
        assert zscore.median() > -4

    def test_volatility_ratio(self, sample_ohlcv_df):
        """Vol Ratio sollte positive Werte haben."""
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        indicator = MacroSurpriseIndicator()
        df = indicator.compute(sample_ohlcv_df.copy())

        vol_ratio = df["macro_vol_ratio"].dropna()
        assert (vol_ratio >= 0).all()

    def test_streak_calculation(self, gap_df):
        """Gap Streak sollte korrekt zählen."""
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        indicator = MacroSurpriseIndicator()
        df = indicator.compute(gap_df.copy())

        # Streak sollte >= 0 sein
        streak = df["macro_gap_streak"].dropna()
        assert (streak >= 0).all()

    def test_custom_vol_lookback(self, sample_ohlcv_df):
        """Benutzerdefinierter Vol Lookback sollte funktionieren."""
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        indicator = MacroSurpriseIndicator()
        df = indicator.compute(sample_ohlcv_df.copy(), vol_lookback=10)

        # Sollte trotzdem funktionieren
        assert "macro_range_surprise" in df.columns

    def test_custom_surprise_threshold(self, sample_ohlcv_df):
        """Höherer Threshold sollte weniger Surprises produzieren."""
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        indicator = MacroSurpriseIndicator()

        df_low = indicator.compute(sample_ohlcv_df.copy(), surprise_threshold=1.0)
        df_high = indicator.compute(sample_ohlcv_df.copy(), surprise_threshold=3.0)

        surprises_low = df_low["macro_is_surprise"].sum()
        surprises_high = df_high["macro_is_surprise"].sum()

        assert surprises_low >= surprises_high

    def test_gap_normalized_values(self, gap_df):
        """Normalisierter Gap sollte relative Größe zeigen."""
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        indicator = MacroSurpriseIndicator()
        df = indicator.compute(gap_df.copy())

        gap_norm = df["macro_gap_normalized"].dropna()
        # Sollte existieren und finite Werte haben
        assert len(gap_norm) > 0
        assert np.isfinite(gap_norm).any()

    def test_default_params(self):
        """Default-Parameter sollten gesetzt sein."""
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        params = MacroSurpriseIndicator.get_default_params()
        assert "vol_lookback" in params
        assert "surprise_threshold" in params
        assert "gap_ma_period" in params
        assert params["surprise_threshold"] == 2.0


# === Integration Tests ===


class TestIndicatorIntegration:
    """Integration Tests für mehrere Indikatoren."""

    def test_all_new_indicators_load(self):
        """Alle neuen Indikatoren sollten importierbar sein."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        assert MicrostructureIndicator is not None
        assert MacroSurpriseIndicator is not None

    def test_indicators_chainable(self, sample_ohlcv_df):
        """Mehrere Indikatoren sollten hintereinander anwendbar sein."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        df = sample_ohlcv_df.copy()

        micro = MicrostructureIndicator()
        surprise = MacroSurpriseIndicator()

        df = micro.compute(df)
        df = surprise.compute(df)

        # Beide Feature-Sets sollten vorhanden sein
        assert "micro_wick_imbalance" in df.columns
        assert "macro_gap" in df.columns

    def test_indicators_with_existing(self, sample_ohlcv_df):
        """Neue Indikatoren sollten mit bestehenden kombinierbar sein."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator
        from fwbg.builtins.indicators.volatility import VolatilityIndicators

        df = sample_ohlcv_df.copy()

        vol = VolatilityIndicators()
        micro = MicrostructureIndicator()

        df = vol.compute(df)
        df = micro.compute(df)

        # Beide sollten da sein
        assert "vol_atr" in df.columns
        assert "micro_range_over_atr" in df.columns

    def test_feature_columns_unique(self, sample_ohlcv_df):
        """Feature-Spalten sollten eindeutige Prefixes haben."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        micro = MicrostructureIndicator()
        surprise = MacroSurpriseIndicator()

        micro_cols = set(micro.get_feature_columns())
        surprise_cols = set(surprise.get_feature_columns())

        # Keine Überlappung
        overlap = micro_cols & surprise_cols
        assert len(overlap) == 0, f"Overlapping columns: {overlap}"


# === Edge Case Tests ===


class TestEdgeCases:
    """Edge Cases für neue Indikatoren."""

    def test_single_row(self):
        """Single Row sollte nicht crashen."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        df = pd.DataFrame({
            "O": [1.1],
            "H": [1.11],
            "L": [1.09],
            "C": [1.105],
            "V": [1000.0],
        }, index=[pd.Timestamp("2024-01-01")])

        micro = MicrostructureIndicator()
        surprise = MacroSurpriseIndicator()

        # Sollte nicht crashen
        df_micro = micro.compute(df.copy())
        df_surprise = surprise.compute(df.copy())

        assert len(df_micro) == 1
        assert len(df_surprise) == 1

    def test_all_nan_volume(self):
        """DataFrame mit NaN Volume sollte Fallback nutzen."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        df = pd.DataFrame({
            "O": [1.1, 1.11, 1.12],
            "H": [1.12, 1.13, 1.14],
            "L": [1.08, 1.09, 1.10],
            "C": [1.11, 1.12, 1.13],
            "V": [np.nan, np.nan, np.nan],
        }, index=pd.date_range("2024-01-01", periods=3, freq="1h"))

        micro = MicrostructureIndicator()
        result = micro.compute(df.copy())

        assert "micro_relative_volume" in result.columns

    def test_extreme_values(self):
        """Extreme Werte sollten nicht zum Crash führen."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator
        from fwbg.builtins.indicators.macro_surprise import MacroSurpriseIndicator

        n = 50
        df = pd.DataFrame({
            "O": np.random.randn(n) * 1e10,
            "H": np.random.randn(n) * 1e10 + 1e9,
            "L": np.random.randn(n) * 1e10 - 1e9,
            "C": np.random.randn(n) * 1e10,
            "V": np.random.randn(n) * 1e6,
        }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))

        # High >= max(O, C), Low <= min(O, C)
        df["H"] = df[["O", "H", "C"]].max(axis=1)
        df["L"] = df[["O", "L", "C"]].min(axis=1)

        micro = MicrostructureIndicator()
        surprise = MacroSurpriseIndicator()

        # Sollte nicht crashen
        micro.compute(df.copy())
        surprise.compute(df.copy())

    def test_negative_prices(self):
        """Negative Preise (wie bei manchen Futures) sollten funktionieren."""
        from fwbg.builtins.indicators.microstructure import MicrostructureIndicator

        n = 50
        df = pd.DataFrame({
            "O": np.linspace(-10, -5, n),
            "H": np.linspace(-9, -4, n),
            "L": np.linspace(-11, -6, n),
            "C": np.linspace(-9.5, -4.5, n),
            "V": np.full(n, 1000.0),
        }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))

        micro = MicrostructureIndicator()
        result = micro.compute(df.copy())

        # Sollte funktionieren
        assert "micro_wick_imbalance" in result.columns


# === Registry Tests ===


class TestIndicatorRegistry:
    """Tests für Indicator Registry Integration."""

    def test_microstructure_registered(self):
        """Microstructure sollte im Registry sein."""
        from fwbg.core.registry import get_indicator, discover_plugins

        discover_plugins()

        cls = get_indicator("microstructure")
        assert cls is not None
        assert cls.group == "microstructure"

    def test_macro_surprise_registered(self):
        """Macro Surprise sollte im Registry sein."""
        from fwbg.core.registry import get_indicator, discover_plugins

        discover_plugins()

        cls = get_indicator("macro_surprise")
        assert cls is not None
        assert cls.group == "macro_surprise"

    def test_registry_lists_new_indicators(self):
        """Neue Indikatoren sollten in list_indicators erscheinen."""
        from fwbg.core.registry import list_indicators, discover_plugins

        discover_plugins()

        indicators = list_indicators()
        assert "microstructure" in indicators
        assert "macro_surprise" in indicators
