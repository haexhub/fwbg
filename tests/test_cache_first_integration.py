"""
Integration-Tests für die Cache-First Architektur des EliteBot.
Nutzt den Demo-Account für echte API-Calls.

Diese Tests benötigen einen echten Demo-Account mit Credentials.
"""
import os
import sys
import pytest

# Skip if demo account does not exist
DEMO_ACCOUNT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "accounts", "main_demo"
)
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DEMO_ACCOUNT_PATH, "account_info.json")),
    reason="Demo account not found - integration tests require real account"
)

import pandas as pd
import numpy as np

# Projekt-Root zum Path hinzufügen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.ig import EliteBot


@pytest.fixture(scope="module")
def bot_instance():
    """
    Erstellt eine Bot-Instanz mit Demo-Account.
    Scope=module damit der Bot nur einmal initialisiert wird (spart API-Calls).
    """
    if not os.path.exists(DEMO_ACCOUNT_PATH):
        pytest.skip("Demo account not found")

    # Erstelle Bot (lädt Daten und trainiert Modelle)
    bot = EliteBot(DEMO_ACCOUNT_PATH)
    yield bot
    # Cleanup - keine spezielle Aktion nötig


class TestCacheInitialization:
    """Tests für die Cache-Initialisierung beim Bot-Start."""

    def test_ohlc_cache_filled_after_init(self, bot_instance):
        """OHLC-Cache sollte für alle trainierten Modelle gefüllt sein."""
        for symbol in bot_instance.models.keys():
            assert symbol in bot_instance.ohlc_cache, f"OHLC-Cache fehlt für {symbol}"
            df = bot_instance.ohlc_cache[symbol]
            assert df is not None, f"OHLC-Cache ist None für {symbol}"
            assert len(df) > 0, f"OHLC-Cache ist leer für {symbol}"

    def test_features_cache_filled_after_init(self, bot_instance):
        """Feature-Cache sollte für alle trainierten Modelle gefüllt sein."""
        for symbol in bot_instance.models.keys():
            assert symbol in bot_instance.features_cache, f"Feature-Cache fehlt für {symbol}"
            df = bot_instance.features_cache[symbol]
            assert df is not None, f"Feature-Cache ist None für {symbol}"
            assert len(df) > 0, f"Feature-Cache ist leer für {symbol}"

    def test_last_bar_time_tracked(self, bot_instance):
        """last_bar_time sollte für alle Symbole gesetzt sein."""
        for symbol in bot_instance.models.keys():
            assert symbol in bot_instance.last_bar_time, f"last_bar_time fehlt für {symbol}"
            assert bot_instance.last_bar_time[symbol] is not None

    def test_ohlc_cache_has_required_columns(self, bot_instance):
        """OHLC-Cache sollte O, H, L, C Spalten haben."""
        required_cols = ["O", "H", "L", "C"]
        for symbol in bot_instance.models.keys():
            df = bot_instance.ohlc_cache[symbol]
            for col in required_cols:
                assert col in df.columns, f"Spalte {col} fehlt in OHLC-Cache für {symbol}"


class TestFeatureCacheQuality:
    """Tests für die Qualität der gecachten Features."""

    def test_features_cache_has_required_features(self, bot_instance):
        """Feature-Cache sollte alle konfigurierten Features enthalten."""
        for symbol, cfg in bot_instance.assets.items():
            if symbol not in bot_instance.models:
                continue

            df = bot_instance.features_cache[symbol]
            for feature in cfg["features"]:
                assert feature in df.columns, f"Feature {feature} fehlt für {symbol}"

    def test_features_have_valid_values_at_last_row(self, bot_instance):
        """Die letzte Zeile sollte keine NaN-Werte in den Features haben."""
        for symbol, cfg in bot_instance.assets.items():
            if symbol not in bot_instance.models:
                continue

            df = bot_instance.features_cache[symbol]
            last_row = df[cfg["features"]].iloc[-1]

            # Mindestens 80% der Features sollten nicht NaN sein
            non_nan_ratio = last_row.notna().sum() / len(last_row)
            assert non_nan_ratio >= 0.8, f"Zu viele NaN-Werte in letzter Zeile für {symbol}: {non_nan_ratio:.1%}"

    def test_atr_available_for_order_execution(self, bot_instance):
        """ATR sollte im Cache verfügbar sein für schnelle Order-Ausführung."""
        import ta

        for symbol in bot_instance.models.keys():
            ohlc_df = bot_instance.ohlc_cache[symbol]
            assert len(ohlc_df) >= 14, f"Nicht genug Daten für ATR-Berechnung: {symbol}"

            atr = ta.volatility.average_true_range(
                ohlc_df["H"], ohlc_df["L"], ohlc_df["C"]
            ).iloc[-1]

            assert not np.isnan(atr), f"ATR ist NaN für {symbol}"
            assert atr > 0, f"ATR ist <= 0 für {symbol}"


class TestIncrementalCacheUpdate:
    """Tests für das inkrementelle Cache-Update."""

    def test_update_ohlc_cache_returns_true_when_current(self, bot_instance):
        """update_ohlc_cache sollte True zurückgeben wenn Cache aktuell ist."""
        for symbol in bot_instance.models.keys():
            # Cache wurde gerade gefüllt, sollte aktuell sein
            result = bot_instance.update_ohlc_cache(symbol)
            assert result is True, f"Cache-Update fehlgeschlagen für {symbol}"

    def test_cache_preserves_data_after_update(self, bot_instance):
        """Cache sollte nach Update mindestens so viele Daten haben wie vorher."""
        for symbol in bot_instance.models.keys():
            len_before = len(bot_instance.ohlc_cache[symbol])
            bot_instance.update_ohlc_cache(symbol)
            len_after = len(bot_instance.ohlc_cache[symbol])

            assert len_after >= len_before, f"Datenverlust nach Update für {symbol}"


class TestPredictionFromCache:
    """Tests für die Prediction aus dem Cache."""

    def test_can_predict_from_cache(self, bot_instance):
        """Prediction sollte aus gecachten Features funktionieren."""
        for symbol, cfg in bot_instance.assets.items():
            if symbol not in bot_instance.models:
                continue

            df = bot_instance.features_cache[symbol]
            model = bot_instance.models[symbol]

            # Prediction auf letzter Zeile
            try:
                prob = model.predict_proba(
                    df[cfg["features"]].iloc[[-1]]
                )[0, 1]

                assert 0 <= prob <= 1, f"Ungültige Probability für {symbol}: {prob}"
            except Exception as e:
                pytest.fail(f"Prediction fehlgeschlagen für {symbol}: {e}")

    def test_prediction_is_deterministic(self, bot_instance):
        """Prediction sollte bei gleichen Daten gleich sein."""
        for symbol, cfg in bot_instance.assets.items():
            if symbol not in bot_instance.models:
                continue

            df = bot_instance.features_cache[symbol]
            model = bot_instance.models[symbol]

            prob1 = model.predict_proba(df[cfg["features"]].iloc[[-1]])[0, 1]
            prob2 = model.predict_proba(df[cfg["features"]].iloc[[-1]])[0, 1]

            assert prob1 == prob2, f"Nicht-deterministische Prediction für {symbol}"


class TestExecuteOrderFast:
    """Tests für die schnelle Order-Ausführung (ohne echte Orders)."""

    def test_execute_order_fast_method_exists(self, bot_instance):
        """execute_order_fast Methode sollte existieren."""
        assert hasattr(bot_instance, "execute_order_fast")
        assert callable(bot_instance.execute_order_fast)

    def test_cached_atr_can_be_computed(self, bot_instance):
        """ATR kann aus dem Cache berechnet werden."""
        import ta

        for symbol in bot_instance.models.keys():
            ohlc_df = bot_instance.ohlc_cache[symbol]

            cached_atr = ta.volatility.average_true_range(
                ohlc_df["H"], ohlc_df["L"], ohlc_df["C"]
            ).iloc[-1]

            assert cached_atr is not None
            assert not np.isnan(cached_atr)


class TestGetFeaturesForPrediction:
    """Tests für get_features_for_prediction Methode."""

    def test_get_features_returns_cached_when_current(self, bot_instance):
        """get_features_for_prediction sollte Cache zurückgeben wenn aktuell."""
        for symbol in bot_instance.models.keys():
            # Speichere Cache-Referenz
            original_cache = bot_instance.features_cache.get(symbol)

            # Hole Features
            result = bot_instance.get_features_for_prediction(symbol)

            # Sollte gleiche Referenz sein (kein Neuberechnen)
            assert result is original_cache, f"Cache wurde unnötig neu berechnet für {symbol}"

    def test_get_features_returns_valid_dataframe(self, bot_instance):
        """get_features_for_prediction sollte validen DataFrame zurückgeben."""
        for symbol in bot_instance.models.keys():
            df = bot_instance.get_features_for_prediction(symbol)

            assert df is not None
            assert isinstance(df, pd.DataFrame)
            assert len(df) >= 100


class TestBotStatusAndHeartbeat:
    """Tests für Status-Tracking."""

    def test_write_status_creates_file(self, bot_instance):
        """write_status sollte Status-Datei erstellen."""
        import json

        bot_instance.write_status("TESTING")

        status_dir = os.path.join("stats_export", bot_instance.account_id)
        status_file = os.path.join(status_dir, "bot_status.json")

        assert os.path.exists(status_file), "Status-Datei wurde nicht erstellt"

        with open(status_file) as f:
            status = json.load(f)

        assert status["status"] == "TESTING"
        assert "last_heartbeat" in status
        assert "active_pairs_count" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
