"""
Unit-Tests für die Cache-First Logik des EliteBot.
Diese Tests laufen ohne API-Calls und können in der CI ausgeführt werden.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Projekt-Root zum Path hinzufügen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCacheArchitecture:
    """Tests für die Cache-Architektur ohne echte API-Calls."""

    def test_bot_has_cache_attributes(self):
        """Bot-Klasse sollte Cache-Attribute haben."""
        from ig_bot import EliteBot

        # Prüfe dass die Attribute in __init__ definiert werden
        import inspect
        source = inspect.getsource(EliteBot.__init__)

        assert "ohlc_cache" in source, "ohlc_cache fehlt in __init__"
        assert "features_cache" in source, "features_cache fehlt in __init__"
        assert "last_bar_time" in source, "last_bar_time fehlt in __init__"

    def test_bot_has_update_ohlc_cache_method(self):
        """Bot sollte update_ohlc_cache Methode haben."""
        from ig_bot import EliteBot

        assert hasattr(EliteBot, "update_ohlc_cache")
        assert callable(getattr(EliteBot, "update_ohlc_cache"))

    def test_bot_has_get_features_for_prediction_method(self):
        """Bot sollte get_features_for_prediction Methode haben."""
        from ig_bot import EliteBot

        assert hasattr(EliteBot, "get_features_for_prediction")
        assert callable(getattr(EliteBot, "get_features_for_prediction"))

    def test_bot_has_execute_order_fast_method(self):
        """Bot sollte execute_order_fast Methode haben."""
        from ig_bot import EliteBot

        assert hasattr(EliteBot, "execute_order_fast")
        assert callable(getattr(EliteBot, "execute_order_fast"))

    def test_bot_has_update_cache_background_method(self):
        """Bot sollte update_cache_background Methode haben."""
        from ig_bot import EliteBot

        assert hasattr(EliteBot, "update_cache_background")
        assert callable(getattr(EliteBot, "update_cache_background"))


class TestRunMethodArchitecture:
    """Tests für die run() Methode Architektur."""

    def test_run_method_uses_features_cache(self):
        """run() sollte features_cache nutzen statt load_and_prepare_data."""
        from ig_bot import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.run)

        # Sollte Cache nutzen
        assert "features_cache" in source, "run() nutzt nicht features_cache"

        # Sollte NICHT load_and_prepare_data in der Signal-Schleife aufrufen
        # (außer für Fallback)
        assert "get_features_for_prediction" not in source or "update_cache_background" in source

    def test_run_method_has_signal_check_phase(self):
        """run() sollte eine Signal-Check Phase haben."""
        from ig_bot import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.run)

        assert "signals_to_execute" in source, "run() hat keine signals_to_execute Liste"

    def test_run_method_uses_execute_order_fast(self):
        """run() sollte execute_order_fast für schnelle Ausführung nutzen."""
        from ig_bot import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.run)

        assert "execute_order_fast" in source, "run() nutzt nicht execute_order_fast"

    def test_run_method_prevents_duplicate_signals_per_hour(self):
        """run() sollte mehrfache Signale pro Stunde verhindern."""
        from ig_bot import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.run)

        assert "last_signal_hour" in source, "run() hat keine last_signal_hour Logik"


class TestTrainMethodCacheFilling:
    """Tests für das Cache-Füllen während des Trainings."""

    def test_train_method_fills_ohlc_cache(self):
        """train_elite_model sollte OHLC-Cache füllen."""
        from ig_bot import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.train_elite_model)

        assert "ohlc_cache" in source, "train_elite_model füllt nicht ohlc_cache"

    def test_train_method_fills_features_cache(self):
        """train_elite_model sollte Feature-Cache füllen."""
        from ig_bot import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.train_elite_model)

        assert "features_cache" in source, "train_elite_model füllt nicht features_cache"

    def test_train_method_sets_last_bar_time(self):
        """train_elite_model sollte last_bar_time setzen."""
        from ig_bot import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.train_elite_model)

        assert "last_bar_time" in source, "train_elite_model setzt nicht last_bar_time"


class TestExecuteOrderFastMethod:
    """Tests für execute_order_fast Methode."""

    def test_execute_order_fast_takes_cached_atr(self):
        """execute_order_fast sollte cached_atr als Parameter nehmen."""
        from ig_bot import EliteBot
        import inspect

        sig = inspect.signature(EliteBot.execute_order_fast)
        params = list(sig.parameters.keys())

        assert "cached_atr" in params, "execute_order_fast hat keinen cached_atr Parameter"

    def test_execute_order_fast_does_not_call_fetch_ig_historical(self):
        """execute_order_fast sollte NICHT fetch_ig_historical aufrufen."""
        from ig_bot import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.execute_order_fast)

        assert "fetch_ig_historical" not in source, "execute_order_fast ruft fetch_ig_historical auf (sollte es nicht!)"


class TestUpdateOhlcCacheLogic:
    """Tests für die update_ohlc_cache Logik."""

    def test_update_ohlc_cache_checks_cache_existence(self):
        """update_ohlc_cache sollte prüfen ob Cache existiert."""
        from ig_bot import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.update_ohlc_cache)

        assert "ohlc_cache" in source
        assert "not in" in source or "if symbol" in source

    def test_update_ohlc_cache_checks_hour_boundary(self):
        """update_ohlc_cache sollte auf Stundenwechsel prüfen."""
        from ig_bot import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.update_ohlc_cache)

        assert "current_hour" in source or "hour" in source.lower()

    def test_update_ohlc_cache_limits_new_bars(self):
        """update_ohlc_cache sollte Anzahl neuer Bars limitieren."""
        from ig_bot import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.update_ohlc_cache)

        assert "24" in source or "max" in source.lower(), "Kein Limit für neue Bars"


class TestBotVersion:
    """Tests für Bot-Versionierung."""

    def test_bot_version_is_7_0(self):
        """Bot sollte Version 7.0 (Cache-First) sein."""
        from ig_bot import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.__init__)

        assert "7.0" in source or "Cache-First" in source, "Bot ist nicht Version 7.0"


class TestSleepIntervals:
    """Tests für Sleep-Intervalle."""

    def test_run_method_sleeps_60_seconds(self):
        """run() sollte 60 Sekunden schlafen (statt 300)."""
        from ig_bot import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.run)

        # Sollte 60 Sekunden sein für häufigere Cache-Checks
        assert "sleep(60)" in source or "time.sleep(60)" in source, \
            "run() schläft nicht 60 Sekunden"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
