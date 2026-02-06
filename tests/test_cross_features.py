"""
Tests für Cross Features Indicator.

Diese Tests stellen sicher, dass:
1. Cross Features KEINE Double-Shift Probleme haben
2. Basis-Indikatoren immer neu berechnet werden
3. Verkettete Berechnung korrekt funktioniert

Das Double-Shift Problem tritt auf wenn:
- Momentum-Indikatoren berechnet werden (shiften um 1)
- Cross-Features diese bereits geshifteten Werte verwenden
- Cross-Features am Ende nochmal shiften = Double-Shift (um 2)
"""
import numpy as np
import pandas as pd
import pytest
import ta

from fwbg.builtins.indicators.cross_features import CrossFeatureIndicators
from fwbg.builtins.indicators.momentum import MomentumIndicators
from fwbg.builtins.indicators.volatility import VolatilityIndicators
from fwbg.builtins.indicators.trend import TrendIndicators


def create_test_ohlc(n_bars: int = 200) -> pd.DataFrame:
    """Erstellt Test-OHLC-Daten mit bekanntem Muster."""
    np.random.seed(42)
    idx = pd.date_range('2024-01-01', periods=n_bars, freq='h')

    # Erstelle deterministische Preise mit Trend und Volatilität
    base_price = 100 + np.arange(n_bars) * 0.05
    noise = np.random.randn(n_bars) * 0.5

    close = base_price + noise
    high = close + np.abs(np.random.randn(n_bars)) * 0.3
    low = close - np.abs(np.random.randn(n_bars)) * 0.3
    open_ = close + np.random.randn(n_bars) * 0.1

    return pd.DataFrame({
        'O': open_,
        'H': high,
        'L': low,
        'C': close,
    }, index=idx)


class TestCrossFeatureNoDoubleShift:
    """Tests für korrektes Single-Shift Verhalten."""

    def test_cross_features_shift_only_once(self):
        """Cross Features sollten nur einmal shiften, nicht zweimal."""
        df = create_test_ohlc()
        indicator = CrossFeatureIndicators()
        result = indicator.compute(df)

        # Berechne RSI roh (ohne Shift)
        rsi_raw = ta.momentum.rsi(df["C"], window=14)

        # cross_rsi_high_rising basiert auf RSI > 70 und RSI-Change > 0
        # Bei Bar i sollte das Feature den RSI-Wert von Bar i-1 verwenden

        # Finde Bars wo RSI > 70 war (für Cross-Feature-Aktivierung)
        for i in range(2, min(100, len(result))):
            # Wenn das Feature bei Bar i den Wert von Bar i-2 verwendet,
            # wäre das ein Double-Shift Bug

            rsi_at_i_minus_1 = rsi_raw.iloc[i-1]
            rsi_at_i_minus_2 = rsi_raw.iloc[i-2] if i >= 2 else np.nan

            # Prüfe RSI-basierte Features
            # Das Feature sollte Daten von i-1 reflektieren (Single-Shift)
            # nicht von i-2 (Double-Shift)

            if rsi_at_i_minus_1 > 70 and not pd.isna(rsi_at_i_minus_1):
                rsi_change_at_i_minus_1 = rsi_at_i_minus_1 - rsi_raw.iloc[i-5] if i >= 5 else np.nan
                if not pd.isna(rsi_change_at_i_minus_1) and rsi_change_at_i_minus_1 > 0:
                    # Feature sollte 1 sein bei Bar i
                    feature_val = result['cross_rsi_high_rising'].iloc[i]
                    assert feature_val == 1, \
                        f"At bar {i}: cross_rsi_high_rising should be 1 (RSI at i-1 was {rsi_at_i_minus_1} > 70 and rising)"

    def test_standalone_vs_chained_same_result(self):
        """Standalone und verkettete Berechnung sollten gleiche Ergebnisse geben."""
        df = create_test_ohlc()

        # Standalone: Nur Cross-Features
        standalone_result = CrossFeatureIndicators().compute(df.copy())

        # Chained: Erst andere Indikatoren, dann Cross-Features
        chained = MomentumIndicators().compute(df.copy())
        chained = VolatilityIndicators().compute(chained)
        chained = TrendIndicators().compute(chained)
        chained_result = CrossFeatureIndicators().compute(chained)

        # Cross-Feature Spalten sollten identisch sein
        cross_cols = [c for c in standalone_result.columns if c.startswith('cross_')]

        for col in cross_cols:
            standalone_vals = standalone_result[col]
            chained_vals = chained_result[col]

            # Beide sollten gleich sein (ignoriere NaN-Vergleiche)
            mask = ~(standalone_vals.isna() | chained_vals.isna())
            if mask.any():
                np.testing.assert_array_almost_equal(
                    standalone_vals[mask].values,
                    chained_vals[mask].values,
                    decimal=10,
                    err_msg=f"Column {col} differs between standalone and chained computation"
                )

    def test_first_bar_is_nan(self):
        """Cross Features sollten NaN bei Bar 0 haben (Single-Shift)."""
        df = create_test_ohlc()
        indicator = CrossFeatureIndicators()
        result = indicator.compute(df)

        cross_cols = [c for c in result.columns if c.startswith('cross_')]

        for col in cross_cols:
            assert pd.isna(result[col].iloc[0]), \
                f"Feature {col} should be NaN at bar 0"

    def test_second_bar_not_nan_if_data_available(self):
        """Bar 1 sollte Daten haben wenn genug Historie vorhanden.

        Bei Double-Shift wäre auch Bar 1 NaN.
        """
        df = create_test_ohlc()
        indicator = CrossFeatureIndicators()
        result = indicator.compute(df)

        # Einige einfache Features sollten bei Bar 1 schon Werte haben
        # (wenn sie nur RSI > 70 prüfen, und RSI bei Bar 0 schon berechnet war)

        # Prüfe ob mindestens einige Features bei Bar 1 definiert sind
        cross_cols = [c for c in result.columns if c.startswith('cross_')]

        # Zähle Features mit Werten bei Bar 1
        defined_at_bar1 = sum(
            1 for col in cross_cols
            if not pd.isna(result[col].iloc[1])
        )

        # Mindestens einige Features sollten bei Bar 1 definiert sein
        # (nicht alle wegen Rolling Windows in manchen Features)
        assert defined_at_bar1 > 0, \
            "Some cross features should be defined at bar 1 (no double-shift)"


class TestBaseIndicatorRecalculation:
    """Tests für Neuberechnung der Basis-Indikatoren."""

    def test_ignores_existing_features(self):
        """Cross Features sollten existierende Features ignorieren."""
        df = create_test_ohlc()

        # Füge "falsche" RSI-Werte hinzu
        df['mom_rsi_14'] = 999.0  # Absichtlich falscher Wert

        indicator = CrossFeatureIndicators()
        result = indicator.compute(df)

        # Cross Features sollten RSI neu berechnen, nicht den falschen Wert verwenden
        # Der falsche Wert von 999 würde RSI > 70 immer triggern

        # Berechne echten RSI
        true_rsi = ta.momentum.rsi(df["C"], window=14)

        # Bei Bars wo echter RSI < 30, sollte cross_rsi_low_falling möglich sein
        # nicht unmöglich wegen dem falschen 999-Wert
        for i in range(20, min(100, len(result))):
            true_rsi_at_prev = true_rsi.iloc[i-1]
            if true_rsi_at_prev < 30:
                rsi_change = true_rsi.iloc[i-1] - true_rsi.iloc[i-5] if i >= 5 else np.nan
                if not pd.isna(rsi_change) and rsi_change < 0:
                    feature_val = result['cross_rsi_low_falling'].iloc[i]
                    assert feature_val == 1, \
                        f"At bar {i}: should use recalculated RSI ({true_rsi_at_prev}), not fake value (999)"

    def test_compute_base_indicators_returns_unshifted(self):
        """_compute_base_indicators sollte nicht geshiftete Werte zurückgeben."""
        df = create_test_ohlc()
        indicator = CrossFeatureIndicators()

        base = indicator._compute_base_indicators(df)

        # Prüfe RSI gegen direkte Berechnung
        expected_rsi = ta.momentum.rsi(df["C"], window=14)

        # Sollten identisch sein (nicht geshiptet)
        np.testing.assert_array_almost_equal(
            base["rsi"].values,
            expected_rsi.values,
            decimal=10,
            err_msg="Base RSI should be unshifted (raw calculation)"
        )


class TestCrossFeatureIntegration:
    """Integration Tests für Cross Features."""

    def test_full_pipeline_no_double_shift(self):
        """Komplette Pipeline sollte korrekt funktionieren."""
        df = create_test_ohlc()

        # Führe komplette Pipeline aus
        result = MomentumIndicators().compute(df)
        result = VolatilityIndicators().compute(result)
        result = TrendIndicators().compute(result)
        result = CrossFeatureIndicators().compute(result)

        # Alle Features sollten Single-Shift haben
        all_feature_cols = [c for c in result.columns if c not in df.columns]

        # Bar 0 sollte NaN sein für alle
        for col in all_feature_cols:
            if not col.startswith('_'):
                assert pd.isna(result[col].iloc[0]), \
                    f"Feature {col} should be NaN at bar 0 after full pipeline"

        # Bar 2+ sollte Werte haben (wenn genug Historie)
        # Bei Double-Shift wären mehr Bars NaN
        defined_counts = {}
        for col in all_feature_cols:
            if not col.startswith('_'):
                # Zähle definierte Werte ab Bar 20 (genug Warmup)
                defined = (~result[col].iloc[20:50].isna()).sum()
                defined_counts[col] = defined

        # Die meisten Features sollten bei Bar 20-50 definiert sein
        avg_defined = np.mean(list(defined_counts.values()))
        assert avg_defined > 20, \
            f"Most features should be defined after warmup, but avg defined is {avg_defined}"

    def test_cross_features_with_various_indicators(self):
        """Cross Features sollten mit allen Indikator-Kombinationen funktionieren."""
        df = create_test_ohlc()

        # Verschiedene Reihenfolgen testen
        orders = [
            [MomentumIndicators(), CrossFeatureIndicators()],
            [VolatilityIndicators(), MomentumIndicators(), CrossFeatureIndicators()],
            [TrendIndicators(), VolatilityIndicators(), MomentumIndicators(), CrossFeatureIndicators()],
        ]

        results = []
        for order in orders:
            result = df.copy()
            for indicator in order:
                result = indicator.compute(result)
            results.append(result)

        # Cross-Feature Spalten sollten in allen Fällen gleich sein
        cross_cols = [c for c in results[0].columns if c.startswith('cross_')]

        for col in cross_cols:
            for i, result in enumerate(results[1:], start=2):
                mask = ~(results[0][col].isna() | result[col].isna())
                if mask.any():
                    np.testing.assert_array_almost_equal(
                        results[0][col][mask].values,
                        result[col][mask].values,
                        decimal=10,
                        err_msg=f"Column {col} differs with different indicator order (order {i})"
                    )
