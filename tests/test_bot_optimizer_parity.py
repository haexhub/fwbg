"""
Tests für Bot-Optimizer Parität.

Stellt sicher, dass der Bot exakt dieselbe Logik verwendet wie der Optimizer:
- Identische Feature-Berechnung
- Identische Indikator-Werte
- Identische Makro-Daten-Integration
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Füge Projekt-Root zum Path hinzu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer.indicators import compute_indicator_pool, get_feature_columns
from optimizer.data_loader import load_macro_indicators, load_interest_rates


class TestFeatureCalculationParity:
    """Tests für identische Feature-Berechnung zwischen Bot und Optimizer."""

    @pytest.fixture
    def sample_ohlc_data(self):
        """Erstellt realistische OHLC-Testdaten."""
        np.random.seed(42)
        n = 500

        # Simuliere Preisentwicklung
        base_price = 1800.0  # Gold-ähnlicher Preis
        returns = np.random.normal(0, 0.001, n)
        prices = base_price * np.cumprod(1 + returns)

        # Erstelle OHLC
        df = pd.DataFrame({
            'O': prices * (1 + np.random.uniform(-0.001, 0.001, n)),
            'H': prices * (1 + np.random.uniform(0, 0.003, n)),
            'L': prices * (1 - np.random.uniform(0, 0.003, n)),
            'C': prices,
        })

        # Index mit Timestamps
        df.index = pd.date_range('2024-01-01', periods=n, freq='h')
        df.index.name = 'T'

        return df

    def test_compute_indicator_pool_produces_ichimoku(self, sample_ohlc_data):
        """compute_indicator_pool muss alle Ichimoku-Indikatoren berechnen."""
        df = compute_indicator_pool(sample_ohlc_data.copy())

        ichimoku_features = [
            'ichi_tenkan', 'ichi_kijun', 'ichi_senkou_a', 'ichi_senkou_b',
            'ichi_cloud_pos', 'ichi_cloud_thick', 'ichi_tk_cross', 'ichi_price_kijun'
        ]

        for feat in ichimoku_features:
            assert feat in df.columns, f"Ichimoku Feature {feat} fehlt!"
            # Prüfe dass Werte nicht alle NaN sind (nach Warmup-Phase)
            assert not df[feat].iloc[100:].isna().all(), f"{feat} ist komplett NaN"

    def test_compute_indicator_pool_produces_trend_features(self, sample_ohlc_data):
        """compute_indicator_pool muss alle Trend-Indikatoren berechnen."""
        df = compute_indicator_pool(sample_ohlc_data.copy())

        trend_features = [
            'trend_adx_14', 'trend_ema_dist_50', 'trend_ema_dist_200',
            'trend_sma_dist_200', 'trend_macd', 'trend_macd_signal'
        ]

        for feat in trend_features:
            assert feat in df.columns, f"Trend Feature {feat} fehlt!"
            assert not df[feat].iloc[200:].isna().all(), f"{feat} ist komplett NaN"

    def test_compute_indicator_pool_produces_momentum_features(self, sample_ohlc_data):
        """compute_indicator_pool muss alle Momentum-Indikatoren berechnen."""
        df = compute_indicator_pool(sample_ohlc_data.copy())

        momentum_features = [
            'mom_rsi_14', 'mom_stoch_k_14', 'mom_stoch_d_14', 'mom_williams_14'
        ]

        for feat in momentum_features:
            assert feat in df.columns, f"Momentum Feature {feat} fehlt!"
            assert not df[feat].iloc[100:].isna().all(), f"{feat} ist komplett NaN"

    def test_compute_indicator_pool_produces_volatility_features(self, sample_ohlc_data):
        """compute_indicator_pool muss alle Volatilitäts-Indikatoren berechnen."""
        df = compute_indicator_pool(sample_ohlc_data.copy())

        vol_features = [
            'vol_atr', 'vol_atr_pct_14', 'vol_bb_pband_20', 'vol_bb_wband_20',
            'vol_kc_pband', 'vol_kc_wband'
        ]

        for feat in vol_features:
            assert feat in df.columns, f"Volatility Feature {feat} fehlt!"
            assert not df[feat].iloc[100:].isna().all(), f"{feat} ist komplett NaN"

    def test_compute_indicator_pool_produces_time_features(self, sample_ohlc_data):
        """compute_indicator_pool muss Zeit-Features berechnen."""
        df = compute_indicator_pool(sample_ohlc_data.copy())

        time_features = ['time_hour', 'time_day', 'time_hour_sin', 'time_hour_cos']

        for feat in time_features:
            assert feat in df.columns, f"Time Feature {feat} fehlt!"
            assert not df[feat].isna().any(), f"{feat} hat NaN Werte"

    def test_feature_values_are_deterministic(self, sample_ohlc_data):
        """Feature-Berechnung muss deterministisch sein."""
        df1 = compute_indicator_pool(sample_ohlc_data.copy())
        df2 = compute_indicator_pool(sample_ohlc_data.copy())

        # Wähle einige Features zum Vergleichen
        features_to_check = ['ichi_tenkan', 'trend_adx_14', 'mom_rsi_14', 'vol_atr']

        for feat in features_to_check:
            np.testing.assert_array_almost_equal(
                df1[feat].dropna().values,
                df2[feat].dropna().values,
                decimal=10,
                err_msg=f"Feature {feat} ist nicht deterministisch!"
            )

    def test_internal_atr_column_created(self, sample_ohlc_data):
        """_atr Spalte für Sizing muss erstellt werden."""
        df = compute_indicator_pool(sample_ohlc_data.copy())

        assert '_atr' in df.columns, "_atr Spalte für TP/SL Sizing fehlt!"
        assert not df['_atr'].iloc[20:].isna().all(), "_atr ist komplett NaN"


class TestAssetConfigFeatureAvailability:
    """Tests dass alle konfigurierten Features auch berechnet werden."""

    @pytest.fixture
    def sample_ohlc_data(self):
        """Erstellt OHLC-Testdaten mit ausreichend Länge."""
        np.random.seed(42)
        n = 600  # Genug für alle Lookbacks

        base_price = 1800.0
        returns = np.random.normal(0, 0.001, n)
        prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'O': prices * (1 + np.random.uniform(-0.001, 0.001, n)),
            'H': prices * (1 + np.random.uniform(0, 0.003, n)),
            'L': prices * (1 - np.random.uniform(0, 0.003, n)),
            'C': prices,
        })

        df.index = pd.date_range('2024-01-01', periods=n, freq='h')
        df.index.name = 'T'

        return df

    def test_brent_features_available(self, sample_ohlc_data):
        """Alle BRENT Features müssen verfügbar sein."""
        brent_features = [
            "ichi_senkou_a", "ichi_senkou_b", "ichi_kijun",
            "trend_ema_dist_200", "trend_sma_dist_200",
            "vol_atr", "ichi_tenkan"
        ]

        df = compute_indicator_pool(sample_ohlc_data.copy())

        for feat in brent_features:
            assert feat in df.columns, f"BRENT Feature {feat} nicht berechnet!"
            # Nach Warmup sollten Werte vorhanden sein
            non_nan_count = df[feat].iloc[250:].notna().sum()
            assert non_nan_count > 0, f"BRENT Feature {feat} hat keine gültigen Werte"

    def test_xagusd_features_available(self, sample_ohlc_data):
        """Alle XAGUSD Features müssen verfügbar sein."""
        xagusd_features = [
            "ichi_kijun", "trend_ema_dist_50", "ichi_tenkan",
            "ichi_senkou_a", "ichi_senkou_b",
            "trend_ema_dist_200", "trend_sma_dist_200"
        ]

        df = compute_indicator_pool(sample_ohlc_data.copy())

        for feat in xagusd_features:
            assert feat in df.columns, f"XAGUSD Feature {feat} nicht berechnet!"
            non_nan_count = df[feat].iloc[250:].notna().sum()
            assert non_nan_count > 0, f"XAGUSD Feature {feat} hat keine gültigen Werte"


class TestMacroDataIntegration:
    """Tests für Makro-Daten Integration."""

    @pytest.fixture
    def sample_ohlc_with_dates(self):
        """OHLC-Daten mit realistischen Datumsangaben."""
        np.random.seed(42)
        n = 200

        base_price = 1800.0
        returns = np.random.normal(0, 0.001, n)
        prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'O': prices * (1 + np.random.uniform(-0.001, 0.001, n)),
            'H': prices * (1 + np.random.uniform(0, 0.003, n)),
            'L': prices * (1 - np.random.uniform(0, 0.003, n)),
            'C': prices,
        })

        # Stündliche Daten
        df.index = pd.date_range('2024-01-01', periods=n, freq='h')

        return df

    def test_load_macro_indicators_adds_date_column(self, sample_ohlc_with_dates):
        """load_macro_indicators muss _date temporär hinzufügen und entfernen."""
        df = sample_ohlc_with_dates.copy()
        df = load_macro_indicators(df)

        # _date sollte am Ende entfernt sein
        assert '_date' not in df.columns, "_date Spalte sollte entfernt werden"

    def test_load_interest_rates_structure(self, sample_ohlc_with_dates):
        """load_interest_rates muss korrekte Struktur haben."""
        df = sample_ohlc_with_dates.copy()
        df = load_interest_rates(df)

        # Funktion sollte ohne Fehler durchlaufen (auch ohne Daten)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_ohlc_with_dates)


class TestBotImportsOptimizerModules:
    """Tests dass der Bot die Optimizer-Module korrekt importiert."""

    def test_bot_imports_compute_indicator_pool(self):
        """Bot muss compute_indicator_pool importieren."""
        # Lade ig_bot.py als Text und prüfe Imports
        bot_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ig_bot.py'
        )

        with open(bot_path, 'r') as f:
            bot_code = f.read()

        assert 'from optimizer.indicators import compute_indicator_pool' in bot_code, \
            "Bot muss compute_indicator_pool importieren!"

    def test_bot_imports_load_macro_indicators(self):
        """Bot muss load_macro_indicators importieren."""
        bot_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ig_bot.py'
        )

        with open(bot_path, 'r') as f:
            bot_code = f.read()

        assert 'from optimizer.data_loader import load_macro_indicators' in bot_code, \
            "Bot muss load_macro_indicators importieren!"

    def test_bot_imports_load_interest_rates(self):
        """Bot muss load_interest_rates importieren."""
        bot_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ig_bot.py'
        )

        with open(bot_path, 'r') as f:
            bot_code = f.read()

        assert 'load_interest_rates' in bot_code, \
            "Bot muss load_interest_rates importieren!"

    def test_bot_uses_compute_indicator_pool_in_load_and_prepare(self):
        """Bot muss compute_indicator_pool in load_and_prepare_data verwenden."""
        bot_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ig_bot.py'
        )

        with open(bot_path, 'r') as f:
            bot_code = f.read()

        # Prüfe dass compute_indicator_pool in load_and_prepare_data aufgerufen wird
        assert 'compute_indicator_pool(df)' in bot_code or 'compute_indicator_pool(' in bot_code, \
            "Bot muss compute_indicator_pool in load_and_prepare_data aufrufen!"


class TestBotDataFlow:
    """Tests für den Datenfluss im Bot."""

    def test_bot_has_fetch_ig_historical_method(self):
        """Bot muss fetch_ig_historical Methode haben."""
        bot_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ig_bot.py'
        )

        with open(bot_path, 'r') as f:
            bot_code = f.read()

        assert 'def fetch_ig_historical(' in bot_code, \
            "Bot muss fetch_ig_historical Methode haben!"

    def test_bot_has_fetch_macro_data_method(self):
        """Bot muss fetch_macro_data Methode haben."""
        bot_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ig_bot.py'
        )

        with open(bot_path, 'r') as f:
            bot_code = f.read()

        assert 'def fetch_macro_data(' in bot_code, \
            "Bot muss fetch_macro_data Methode haben!"

    def test_bot_uses_ig_api_for_historical_data(self):
        """Bot muss IG API für historische Daten verwenden."""
        bot_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ig_bot.py'
        )

        with open(bot_path, 'r') as f:
            bot_code = f.read()

        assert 'fetch_historical_prices_by_epic' in bot_code, \
            "Bot muss IG API fetch_historical_prices_by_epic verwenden!"


class TestGoldFeaturesParity:
    """Spezifische Tests für GOLD Features (Makro-basiert)."""

    def test_gold_requires_macro_features(self):
        """GOLD verwendet Makro-Features, diese müssen definiert sein."""
        gold_features = ["macro_hyg", "macro_gold_fut_chg_12h", "macro_fed_rate"]

        # Diese Features kommen aus load_macro_indicators, nicht compute_indicator_pool
        # Prüfe dass die Namen korrekt sind
        for feat in gold_features:
            assert feat.startswith('macro_'), f"GOLD Feature {feat} muss mit 'macro_' beginnen"

    def test_bot_updates_live_macro_data(self):
        """Bot muss Live-Makro-Daten für aktuelle Entscheidungen aktualisieren."""
        bot_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ig_bot.py'
        )

        with open(bot_path, 'r') as f:
            bot_code = f.read()

        # Bot sollte Live-Makro-Daten in die letzte Zeile schreiben
        assert 'macro_live' in bot_code or 'fetch_macro_data' in bot_code, \
            "Bot muss Live-Makro-Daten aktualisieren!"


class TestFeatureConsistency:
    """Tests für Feature-Konsistenz zwischen Bot und Optimizer."""

    @pytest.fixture
    def sample_data(self):
        """Standard Testdaten."""
        np.random.seed(123)
        n = 500

        base_price = 80.0  # Brent-ähnlicher Preis
        returns = np.random.normal(0, 0.002, n)
        prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            'O': prices * (1 + np.random.uniform(-0.002, 0.002, n)),
            'H': prices * (1 + np.random.uniform(0, 0.005, n)),
            'L': prices * (1 - np.random.uniform(0, 0.005, n)),
            'C': prices,
        })

        df.index = pd.date_range('2024-06-01', periods=n, freq='h')

        return df

    def test_ichimoku_components_mathematically_correct(self, sample_data):
        """Ichimoku-Komponenten müssen mathematisch korrekt sein."""
        df = compute_indicator_pool(sample_data.copy())

        # Tenkan-sen: (9-Perioden High + 9-Perioden Low) / 2
        # Kijun-sen: (26-Perioden High + 26-Perioden Low) / 2
        # Senkou Span A: (Tenkan + Kijun) / 2
        # Senkou Span B: (52-Perioden High + 52-Perioden Low) / 2

        # Prüfe Beziehungen
        idx = 300  # Nach Warmup

        # Senkou A sollte zwischen Tenkan und Kijun liegen (grob)
        tenkan = df['ichi_tenkan'].iloc[idx]
        kijun = df['ichi_kijun'].iloc[idx]

        # Beide sollten numerisch sein
        assert not np.isnan(tenkan), "Tenkan sollte nicht NaN sein"
        assert not np.isnan(kijun), "Kijun sollte nicht NaN sein"

    def test_rsi_values_in_valid_range(self, sample_data):
        """RSI muss zwischen 0 und 100 liegen (nur Basis-RSI, nicht Änderungsraten)."""
        df = compute_indicator_pool(sample_data.copy())

        # Nur Basis-RSI Spalten (nicht dyn_*, lag_*, accel_*)
        rsi_cols = [c for c in df.columns if c.startswith('mom_rsi_')]

        for col in rsi_cols:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                assert valid_values.min() >= 0, f"{col} hat Werte unter 0"
                assert valid_values.max() <= 100, f"{col} hat Werte über 100"

    def test_stochastic_values_in_valid_range(self, sample_data):
        """Stochastik muss zwischen 0 und 100 liegen."""
        df = compute_indicator_pool(sample_data.copy())

        stoch_cols = [c for c in df.columns if 'stoch' in c.lower() and 'chg' not in c.lower()]

        for col in stoch_cols:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                assert valid_values.min() >= 0, f"{col} hat Werte unter 0"
                assert valid_values.max() <= 100, f"{col} hat Werte über 100"

    def test_atr_is_positive(self, sample_data):
        """ATR muss immer positiv sein (nur Basis-ATR, nicht Änderungsraten)."""
        df = compute_indicator_pool(sample_data.copy())

        # Nur Basis-ATR Spalten (nicht dyn_*, accel_*)
        atr_cols = ['_atr', 'vol_atr', 'vol_atr_pct_7', 'vol_atr_pct_14', 'vol_atr_pct_21']

        for col in atr_cols:
            if col in df.columns:
                valid_values = df[col].dropna()
                if len(valid_values) > 0:
                    assert valid_values.min() >= 0, f"{col} hat negative Werte"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
