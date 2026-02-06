"""
Tests für Lookahead Bias Prevention in Indikatoren.

Diese Tests stellen sicher, dass:
1. Alle Indikatoren Features um 1 Bar verschieben
2. Bei Bar i nur Daten von Bar i-1 und früher verfügbar sind
3. Keine aktuellen Bar-Werte in Features einfließen

Das ist KRITISCH für die Integrität des Backtesting-Systems.
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.builtins.indicators.momentum import MomentumIndicators
from fwbg.builtins.indicators.volatility import VolatilityIndicators
from fwbg.builtins.indicators.trend import TrendIndicators
from fwbg.builtins.indicators.dynamics import DynamicsIndicators
from fwbg.builtins.indicators.time_season import TimeSeasonIndicators


def create_test_ohlc(n_bars: int = 100) -> pd.DataFrame:
    """Erstellt Test-OHLC-Daten mit bekanntem Muster."""
    np.random.seed(42)
    idx = pd.date_range('2024-01-01', periods=n_bars, freq='h')

    # Erstelle deterministische Preise mit leichtem Trend
    base_price = 100 + np.arange(n_bars) * 0.1  # Leichter Aufwärtstrend
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


class TestLookaheadBiasPrevention:
    """Tests für Lookahead Bias Prevention."""

    def test_momentum_first_bar_is_nan(self):
        """Momentum Features sollten NaN bei Bar 0 haben."""
        df = create_test_ohlc()
        indicator = MomentumIndicators()
        result = indicator.compute(df)

        # Alle Momentum-Feature-Spalten
        feature_cols = [c for c in result.columns if c.startswith('mom_')]

        for col in feature_cols:
            assert pd.isna(result[col].iloc[0]), \
                f"Feature {col} should be NaN at bar 0 to prevent lookahead"

    def test_volatility_first_bar_is_nan(self):
        """Volatility Features sollten NaN bei Bar 0 haben."""
        df = create_test_ohlc()
        indicator = VolatilityIndicators()
        result = indicator.compute(df)

        feature_cols = [c for c in result.columns if c.startswith('vol_')]

        for col in feature_cols:
            assert pd.isna(result[col].iloc[0]), \
                f"Feature {col} should be NaN at bar 0 to prevent lookahead"

    def test_trend_first_bar_is_nan(self):
        """Trend Features sollten NaN bei Bar 0 haben."""
        df = create_test_ohlc()
        indicator = TrendIndicators()
        result = indicator.compute(df)

        feature_cols = [c for c in result.columns if c.startswith('trend_')]

        for col in feature_cols:
            assert pd.isna(result[col].iloc[0]), \
                f"Feature {col} should be NaN at bar 0 to prevent lookahead"

    def test_time_season_first_bar_is_nan(self):
        """Time/Season Features sollten NaN bei Bar 0 haben."""
        df = create_test_ohlc()
        indicator = TimeSeasonIndicators()
        result = indicator.compute(df)

        # Alle Time/Season-Feature-Spalten
        feature_cols = [c for c in result.columns if c.startswith(('time_', 'season_'))]

        for col in feature_cols:
            assert pd.isna(result[col].iloc[0]), \
                f"Feature {col} should be NaN at bar 0 to prevent lookahead"

    def test_dynamics_first_bar_is_nan(self):
        """Dynamics Features sollten NaN bei Bar 0 haben."""
        df = create_test_ohlc(200)  # Mehr Daten für Dynamics
        indicator = DynamicsIndicators()
        result = indicator.compute(df)

        feature_cols = [c for c in result.columns if c.startswith(('dyn_', 'lag_', 'accel_'))]

        for col in feature_cols:
            assert pd.isna(result[col].iloc[0]), \
                f"Feature {col} should be NaN at bar 0 to prevent lookahead"


class TestFeatureShiftCorrectness:
    """Tests für korrekte Feature-Verschiebung."""

    def test_feature_at_bar_i_uses_bar_i_minus_1(self):
        """Feature bei Bar i sollte Wert von Bar i-1 haben."""
        df = create_test_ohlc()
        indicator = MomentumIndicators()
        result = indicator.compute(df)

        # RSI ist ein guter Test-Kandidat
        import ta

        # Berechne RSI ohne Shift
        rsi_raw = ta.momentum.rsi(df["C"], window=14)

        # Das Feature 'mom_rsi_14' sollte um 1 geshiptet sein
        for i in range(1, min(50, len(result))):
            if not pd.isna(rsi_raw.iloc[i-1]) and not pd.isna(result['mom_rsi_14'].iloc[i]):
                assert np.isclose(
                    result['mom_rsi_14'].iloc[i],
                    rsi_raw.iloc[i-1],
                    rtol=1e-10
                ), f"At bar {i}: feature should equal raw value from bar {i-1}"

    def test_no_current_bar_data_in_features(self):
        """Features dürfen keine aktuellen Bar-Daten enthalten."""
        # Erstelle spezielle Testdaten mit variierenden Werten
        n_bars = 50
        idx = pd.date_range('2024-01-01', periods=n_bars, freq='h')

        # Preise mit leichter Variation, letzte Bar hat extremen Spike
        close = 100.0 + np.sin(np.arange(n_bars) * 0.1) * 2  # Variiert 98-102
        close[-1] = 200.0  # Letzte Bar: extremer Spike auf 200

        df = pd.DataFrame({
            'O': close,
            'H': close + 1,
            'L': close - 1,
            'C': close,
        }, index=idx)

        indicator = MomentumIndicators()
        result = indicator.compute(df)

        # Berechne erwarteten RSI bei Bar n-1 (vor dem Spike)
        import ta
        rsi_raw = ta.momentum.rsi(df["C"], window=14)

        # Feature bei letzter Bar sollte RSI von vorletzter Bar haben
        last_feature = result['mom_rsi_14'].iloc[-1]
        expected_from_prev_bar = rsi_raw.iloc[-2]

        # Wenn beide nicht NaN sind, sollten sie gleich sein
        if not pd.isna(last_feature) and not pd.isna(expected_from_prev_bar):
            assert np.isclose(last_feature, expected_from_prev_bar, rtol=1e-10), \
                f"Last bar feature ({last_feature}) should equal prev bar's raw value ({expected_from_prev_bar})"


class TestMultipleIndicatorConsistency:
    """Tests für Konsistenz über mehrere Indikatoren."""

    def test_all_indicators_shift_consistently(self):
        """Alle Indikatoren sollten konsistent shiften."""
        df = create_test_ohlc(200)

        indicators = [
            MomentumIndicators(),
            VolatilityIndicators(),
            TrendIndicators(),
        ]

        for indicator in indicators:
            result = indicator.compute(df)

            # Alle neuen Feature-Spalten finden
            original_cols = set(df.columns)
            feature_cols = [c for c in result.columns if c not in original_cols]

            # Erste Zeile muss NaN sein für alle Features
            for col in feature_cols:
                if not col.startswith('_'):  # Interne Spalten ignorieren
                    assert pd.isna(result[col].iloc[0]), \
                        f"Indicator {indicator.__class__.__name__}: {col} should be NaN at bar 0"

    def test_chained_indicators_maintain_shift(self):
        """Verkettete Indikatoren sollten Shift beibehalten."""
        df = create_test_ohlc(200)

        # Berechne nacheinander
        result = MomentumIndicators().compute(df)
        result = VolatilityIndicators().compute(result)
        result = TrendIndicators().compute(result)

        # Alle Feature-Spalten
        feature_cols = [c for c in result.columns if c not in df.columns]

        # Erste Zeile muss immer noch NaN sein
        for col in feature_cols:
            if not col.startswith('_'):
                assert pd.isna(result[col].iloc[0]), \
                    f"Chained indicators: {col} should still be NaN at bar 0"


class TestEdgeCases:
    """Tests für Randfälle."""

    def test_short_data_series(self):
        """Kurze Datenreihen sollten ohne Fehler verarbeitet werden."""
        df = create_test_ohlc(30)  # Nur 30 Bars

        indicator = MomentumIndicators()
        result = indicator.compute(df)

        # Sollte nicht abstürzen und DataFrame zurückgeben
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 30

    def test_data_with_gaps(self):
        """Daten mit NaN-Lücken sollten korrekt verarbeitet werden."""
        df = create_test_ohlc()

        # Füge einige NaN-Lücken ein
        df.loc[df.index[20:25], 'C'] = np.nan

        indicator = MomentumIndicators()
        result = indicator.compute(df)

        # Erste Zeile muss immer noch NaN sein
        feature_cols = [c for c in result.columns if c.startswith('mom_')]
        for col in feature_cols:
            assert pd.isna(result[col].iloc[0])
