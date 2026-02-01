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
        from bots.ig import EliteBot

        # Prüfe dass die Attribute in __init__ definiert werden
        import inspect
        source = inspect.getsource(EliteBot.__init__)

        assert "ohlc_cache" in source, "ohlc_cache fehlt in __init__"
        assert "features_cache" in source, "features_cache fehlt in __init__"
        assert "last_bar_time" in source, "last_bar_time fehlt in __init__"

    def test_bot_has_update_ohlc_cache_method(self):
        """Bot sollte update_ohlc_cache Methode haben."""
        from bots.ig import EliteBot

        assert hasattr(EliteBot, "update_ohlc_cache")
        assert callable(getattr(EliteBot, "update_ohlc_cache"))

    def test_bot_has_get_features_for_prediction_method(self):
        """Bot sollte get_features_for_prediction Methode haben."""
        from bots.ig import EliteBot

        assert hasattr(EliteBot, "get_features_for_prediction")
        assert callable(getattr(EliteBot, "get_features_for_prediction"))

    def test_bot_has_execute_order_fast_method(self):
        """Bot sollte execute_order_fast Methode haben."""
        from bots.ig import EliteBot

        assert hasattr(EliteBot, "execute_order_fast")
        assert callable(getattr(EliteBot, "execute_order_fast"))

    def test_bot_has_update_cache_background_method(self):
        """Bot sollte update_cache_background Methode haben."""
        from bots.ig import EliteBot

        assert hasattr(EliteBot, "update_cache_background")
        assert callable(getattr(EliteBot, "update_cache_background"))


class TestRunMethodArchitecture:
    """Tests für die run() Methode Architektur (Polling-Modus)."""

    def test_run_method_uses_features_cache(self):
        """_run_polling() sollte features_cache nutzen statt load_and_prepare_data."""
        from bots.ig import EliteBot
        import inspect

        # Prüfe _run_polling (der Legacy-Polling-Loop)
        source = inspect.getsource(EliteBot._run_polling)

        # Sollte Cache nutzen
        assert "features_cache" in source, "_run_polling() nutzt nicht features_cache"

    def test_run_method_has_signal_check_phase(self):
        """_run_polling() sollte eine Signal-Check Phase haben."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot._run_polling)

        assert "signals_to_execute" in source, "_run_polling() hat keine signals_to_execute Liste"

    def test_run_method_uses_execute_order_fast(self):
        """_run_polling() sollte execute_order_fast für schnelle Ausführung nutzen."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot._run_polling)

        assert "execute_order_fast" in source, "_run_polling() nutzt nicht execute_order_fast"

    def test_run_method_prevents_duplicate_signals_per_hour(self):
        """_run_polling() sollte mehrfache Signale pro Stunde verhindern."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot._run_polling)

        assert "last_signal_hour" in source, "_run_polling() hat keine last_signal_hour Logik"

    def test_run_dispatches_to_streaming_or_polling(self):
        """run() sollte zu streaming oder polling dispatchen."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.run)

        assert "use_streaming" in source, "run() prüft nicht use_streaming"
        assert "run_streaming" in source or "_run_polling" in source


class TestTrainMethodCacheFilling:
    """Tests für das Cache-Füllen während des Trainings."""

    def test_train_method_fills_ohlc_cache(self):
        """train_elite_model sollte OHLC-Cache füllen."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.train_elite_model)

        assert "ohlc_cache" in source, "train_elite_model füllt nicht ohlc_cache"

    def test_train_method_fills_features_cache(self):
        """train_elite_model sollte Feature-Cache füllen."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.train_elite_model)

        assert "features_cache" in source, "train_elite_model füllt nicht features_cache"

    def test_train_method_sets_last_bar_time(self):
        """train_elite_model sollte last_bar_time setzen."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.train_elite_model)

        assert "last_bar_time" in source, "train_elite_model setzt nicht last_bar_time"


class TestExecuteOrderFastMethod:
    """Tests für execute_order_fast Methode."""

    def test_execute_order_fast_takes_cached_atr(self):
        """execute_order_fast sollte cached_atr als Parameter nehmen."""
        from bots.ig import EliteBot
        import inspect

        sig = inspect.signature(EliteBot.execute_order_fast)
        params = list(sig.parameters.keys())

        assert "cached_atr" in params, "execute_order_fast hat keinen cached_atr Parameter"

    def test_execute_order_fast_does_not_call_fetch_ig_historical(self):
        """execute_order_fast sollte NICHT fetch_ig_historical aufrufen."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.execute_order_fast)

        assert "fetch_ig_historical" not in source, "execute_order_fast ruft fetch_ig_historical auf (sollte es nicht!)"


class TestUpdateOhlcCacheLogic:
    """Tests für die update_ohlc_cache Logik."""

    def test_update_ohlc_cache_checks_cache_existence(self):
        """update_ohlc_cache sollte prüfen ob Cache existiert."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.update_ohlc_cache)

        assert "ohlc_cache" in source
        assert "not in" in source or "if symbol" in source

    def test_update_ohlc_cache_checks_hour_boundary(self):
        """update_ohlc_cache sollte auf Stundenwechsel prüfen."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.update_ohlc_cache)

        assert "current_hour" in source or "hour" in source.lower()

    def test_update_ohlc_cache_limits_new_bars(self):
        """update_ohlc_cache sollte Anzahl neuer Bars limitieren."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.update_ohlc_cache)

        assert "24" in source or "max" in source.lower(), "Kein Limit für neue Bars"


class TestBotVersion:
    """Tests für Bot-Versionierung."""

    def test_bot_version_is_8_0(self):
        """Bot sollte Version 8.0 (Streaming) sein."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.__init__)

        assert "8.0" in source or "Streaming" in source, "Bot ist nicht Version 8.0"


class TestSleepIntervals:
    """Tests für Sleep-Intervalle."""

    def test_polling_method_sleeps_60_seconds(self):
        """_run_polling() sollte 60 Sekunden schlafen (statt 300)."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot._run_polling)

        # Sollte 60 Sekunden sein für häufigere Cache-Checks
        assert "sleep(60)" in source or "time.sleep(60)" in source, \
            "_run_polling() schläft nicht 60 Sekunden"

    def test_streaming_method_sleeps_60_seconds(self):
        """run_streaming() sollte auch 60 Sekunden im Health-Loop schlafen."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.run_streaming)

        assert "sleep(60)" in source or "time.sleep(60)" in source, \
            "run_streaming() schläft nicht 60 Sekunden"


class TestSlippageVerification:
    """Tests für die Slippage-Prüfung nach Order-Ausführung."""

    def test_verify_execution_price_method_exists(self):
        """verify_execution_price Methode sollte existieren."""
        from bots.ig import EliteBot

        assert hasattr(EliteBot, "verify_execution_price")
        assert callable(getattr(EliteBot, "verify_execution_price"))

    def test_verify_execution_price_signature(self):
        """verify_execution_price sollte die richtigen Parameter haben."""
        from bots.ig import EliteBot
        import inspect

        sig = inspect.signature(EliteBot.verify_execution_price)
        params = list(sig.parameters.keys())

        assert "symbol" in params, "Parameter 'symbol' fehlt"
        assert "deal_reference" in params, "Parameter 'deal_reference' fehlt"
        assert "expected_price" in params, "Parameter 'expected_price' fehlt"
        assert "max_slippage_pct" in params, "Parameter 'max_slippage_pct' fehlt"

    def test_execute_order_fast_calls_verify_execution(self):
        """execute_order_fast sollte verify_execution_price aufrufen."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.execute_order_fast)

        assert "verify_execution_price" in source, \
            "execute_order_fast ruft nicht verify_execution_price auf"

    def test_execute_order_fast_logs_expected_price(self):
        """execute_order_fast sollte erwarteten Preis loggen."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.execute_order_fast)

        assert "expected_price" in source, \
            "execute_order_fast nutzt keinen expected_price"

    def test_slippage_check_uses_cache_price(self):
        """Slippage-Check sollte Preis aus Cache nutzen."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.execute_order_fast)

        assert "ohlc_cache" in source, \
            "execute_order_fast holt expected_price nicht aus ohlc_cache"

    def test_verify_returns_tuple(self):
        """verify_execution_price sollte Tuple zurückgeben."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.verify_execution_price)

        # Sollte (is_ok, actual_price, slippage_pct) zurückgeben
        assert "return" in source
        assert "False" in source and "True" in source, \
            "verify_execution_price gibt keine boolean Werte zurück"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
