"""
Tests für Indicator Utility-Funktionen.

Testet:
- safe_divide: Division-durch-Null Handling
- shift_features: Lookahead Bias Prevention
- EPSILON Konstante

Diese Tests stellen sicher, dass die Basis-Funktionen
korrekt arbeiten und alle Indikatoren konsistent implementiert sind.
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins.indicator import (
    safe_divide,
    shift_features,
    EPSILON,
)


# === SAFE_DIVIDE TESTS ===

class TestSafeDivide:
    """Tests für die safe_divide Funktion."""

    def test_normal_division(self):
        """Normale Division sollte korrekt funktionieren."""
        numerator = pd.Series([10, 20, 30, 40])
        denominator = pd.Series([2, 4, 5, 8])
        result = safe_divide(numerator, denominator)

        expected = pd.Series([5.0, 5.0, 6.0, 5.0])
        pd.testing.assert_series_equal(result, expected)

    def test_division_by_zero_returns_nan(self):
        """Division durch Null sollte NaN zurückgeben."""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 0, 5])
        result = safe_divide(numerator, denominator)

        assert result.iloc[0] == 5.0
        assert np.isnan(result.iloc[1]), "Division by zero should return NaN"
        assert result.iloc[2] == 6.0

    def test_division_by_very_small_number_returns_nan(self):
        """Division durch sehr kleine Werte sollte NaN zurückgeben."""
        numerator = pd.Series([10, 20])
        denominator = pd.Series([1e-11, 1e-12])  # Kleiner als EPSILON
        result = safe_divide(numerator, denominator)

        assert np.isnan(result.iloc[0]), "Division by value < EPSILON should return NaN"
        assert np.isnan(result.iloc[1]), "Division by value < EPSILON should return NaN"

    def test_negative_small_values_return_nan(self):
        """Division durch negative sehr kleine Werte sollte NaN zurückgeben."""
        numerator = pd.Series([10, 20])
        denominator = pd.Series([-1e-11, -1e-12])
        result = safe_divide(numerator, denominator)

        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])

    def test_numpy_array_input(self):
        """Sollte auch mit numpy arrays funktionieren."""
        numerator = np.array([10, 20, 30])
        denominator = np.array([2, 0, 5])
        result = safe_divide(numerator, denominator)

        assert result[0] == 5.0
        assert np.isnan(result[1])
        assert result[2] == 6.0

    def test_preserves_index(self):
        """Sollte den Index beibehalten."""
        idx = pd.date_range('2024-01-01', periods=3, freq='h')
        numerator = pd.Series([10, 20, 30], index=idx)
        denominator = pd.Series([2, 4, 5], index=idx)
        result = safe_divide(numerator, denominator)

        assert result.index.equals(idx)

    def test_mixed_zero_and_valid(self):
        """Gemischte Nullen und gültige Werte."""
        numerator = pd.Series([100, 200, 300, 400, 500])
        denominator = pd.Series([10, 0, 30, 0, 50])
        result = safe_divide(numerator, denominator)

        assert result.iloc[0] == 10.0
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 10.0
        assert np.isnan(result.iloc[3])
        assert result.iloc[4] == 10.0

    def test_all_zeros_denominator(self):
        """Alle Nullen im Nenner sollten alle NaN geben."""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([0, 0, 0])
        result = safe_divide(numerator, denominator)

        assert result.isna().all(), "All zeros should result in all NaN"

    def test_negative_division(self):
        """Negative Division sollte korrekt funktionieren."""
        numerator = pd.Series([-10, 20, -30])
        denominator = pd.Series([2, -4, -5])
        result = safe_divide(numerator, denominator)

        assert result.iloc[0] == -5.0
        assert result.iloc[1] == -5.0
        assert result.iloc[2] == 6.0


# === SHIFT_FEATURES TESTS ===

class TestShiftFeatures:
    """Tests für die shift_features Funktion."""

    def test_basic_shift(self):
        """Features sollten um 1 Bar geshiptet werden."""
        idx = pd.date_range('2024-01-01', periods=5, freq='h')
        features = {
            'feat_a': pd.Series([1, 2, 3, 4, 5], index=idx),
            'feat_b': pd.Series([10, 20, 30, 40, 50], index=idx),
        }
        result = shift_features(features, idx)

        # Erste Werte sollten NaN sein
        assert np.isnan(result['feat_a'].iloc[0])
        assert np.isnan(result['feat_b'].iloc[0])

        # Werte sollten um 1 verschoben sein
        assert result['feat_a'].iloc[1] == 1
        assert result['feat_a'].iloc[2] == 2
        assert result['feat_b'].iloc[1] == 10
        assert result['feat_b'].iloc[4] == 40

    def test_returns_dataframe(self):
        """Sollte einen DataFrame zurückgeben."""
        idx = pd.date_range('2024-01-01', periods=3, freq='h')
        features = {'a': pd.Series([1, 2, 3], index=idx)}
        result = shift_features(features, idx)

        assert isinstance(result, pd.DataFrame)

    def test_preserves_all_columns(self):
        """Alle Feature-Spalten sollten erhalten bleiben."""
        idx = pd.date_range('2024-01-01', periods=3, freq='h')
        features = {
            'feat_1': pd.Series([1, 2, 3], index=idx),
            'feat_2': pd.Series([4, 5, 6], index=idx),
            'feat_3': pd.Series([7, 8, 9], index=idx),
        }
        result = shift_features(features, idx)

        assert set(result.columns) == {'feat_1', 'feat_2', 'feat_3'}

    def test_correct_index(self):
        """Der Index sollte korrekt sein."""
        idx = pd.date_range('2024-01-01', periods=5, freq='h')
        features = {'a': pd.Series([1, 2, 3, 4, 5], index=idx)}
        result = shift_features(features, idx)

        assert result.index.equals(idx)

    def test_numpy_array_input(self):
        """Sollte auch mit numpy arrays als Input funktionieren."""
        idx = pd.date_range('2024-01-01', periods=4, freq='h')
        features = {'a': np.array([1, 2, 3, 4])}
        result = shift_features(features, idx)

        assert np.isnan(result['a'].iloc[0])
        assert result['a'].iloc[1] == 1
        assert result['a'].iloc[3] == 3

    def test_empty_features_dict(self):
        """Leeres Feature-Dict sollte leeren DataFrame geben."""
        idx = pd.date_range('2024-01-01', periods=3, freq='h')
        features = {}
        result = shift_features(features, idx)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 0

    def test_nan_values_preserved(self):
        """NaN-Werte im Input sollten erhalten bleiben."""
        idx = pd.date_range('2024-01-01', periods=5, freq='h')
        features = {'a': pd.Series([1, np.nan, 3, 4, 5], index=idx)}
        result = shift_features(features, idx)

        # Position 0 ist NaN wegen Shift
        assert np.isnan(result['a'].iloc[0])
        # Position 2 ist NaN aus dem Original
        assert np.isnan(result['a'].iloc[2])
        # Position 1 hat den Wert 1 (geshiftet von Position 0)
        assert result['a'].iloc[1] == 1


# === EPSILON TESTS ===

class TestEpsilonConstant:
    """Tests für die EPSILON Konstante."""

    def test_epsilon_is_small(self):
        """EPSILON sollte klein genug für numerische Stabilität sein."""
        assert EPSILON < 1e-8, "EPSILON should be small"

    def test_epsilon_is_positive(self):
        """EPSILON sollte positiv sein."""
        assert EPSILON > 0, "EPSILON should be positive"

    def test_epsilon_consistent_with_safe_divide(self):
        """EPSILON sollte konsistent mit safe_divide sein."""
        # Werte größer als EPSILON sollten okay sein
        numerator = pd.Series([10])
        denominator = pd.Series([EPSILON * 10])
        result = safe_divide(numerator, denominator)
        assert not np.isnan(result.iloc[0]), "Values > EPSILON should work"

        # Werte kleiner als EPSILON sollten NaN geben
        denominator_small = pd.Series([EPSILON / 10])
        result_small = safe_divide(numerator, denominator_small)
        assert np.isnan(result_small.iloc[0]), "Values < EPSILON should return NaN"


# === INTEGRATION TESTS ===

class TestIntegration:
    """Integration Tests für die Utility-Funktionen."""

    def test_safe_divide_in_ratio_calculation(self):
        """safe_divide sollte in typischen Ratio-Berechnungen funktionieren."""
        # Simuliere Range-Position Berechnung
        close = pd.Series([105, 102, 108, 100, 103])
        low = pd.Series([100, 100, 100, 100, 100])
        high = pd.Series([110, 110, 110, 100, 110])  # Bei Index 3: High = Low

        bar_range = high - low
        range_pos = safe_divide(close - low, bar_range)

        # Normale Berechnung
        assert range_pos.iloc[0] == 0.5  # (105-100)/(110-100)
        assert range_pos.iloc[2] == 0.8  # (108-100)/(110-100)

        # Division durch Null (Doji-Candle wo High=Low)
        assert np.isnan(range_pos.iloc[3]), "Doji candle should have NaN range position"

    def test_shift_features_prevents_lookahead(self):
        """shift_features sollte Lookahead Bias verhindern."""
        idx = pd.date_range('2024-01-01', periods=10, freq='h')

        # Simuliere einen Indikator der aktuelle Bar-Daten verwendet
        current_close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=idx)

        features = {'rsi': current_close}  # Vereinfachtes Beispiel
        shifted = shift_features(features, idx)

        # Bei Bar i sollte der Wert von Bar i-1 verfügbar sein
        for i in range(1, len(idx)):
            assert shifted['rsi'].iloc[i] == current_close.iloc[i - 1], \
                f"At bar {i}, should see value from bar {i-1}"

        # Bei Bar 0 sollte nichts verfügbar sein
        assert np.isnan(shifted['rsi'].iloc[0]), "First bar should have no data"

    def test_combined_usage(self):
        """Kombinierte Nutzung von safe_divide und shift_features."""
        idx = pd.date_range('2024-01-01', periods=5, freq='h')

        # Berechne Features mit safe_divide
        numerator = pd.Series([10, 20, 30, 0, 50], index=idx)
        denominator = pd.Series([2, 0, 5, 0, 10], index=idx)
        ratio = safe_divide(numerator, denominator)

        # Shifte Features
        features = {'ratio': ratio}
        result = shift_features(features, idx)

        # Erste Position ist NaN (Shift)
        assert np.isnan(result['ratio'].iloc[0])

        # Zweite Position hat den Wert von Position 0 (5.0)
        assert result['ratio'].iloc[1] == 5.0

        # Dritte Position hat NaN von Position 1 (Division durch 0)
        assert np.isnan(result['ratio'].iloc[2])

        # Vierte Position hat 6.0 von Position 2
        assert result['ratio'].iloc[3] == 6.0
