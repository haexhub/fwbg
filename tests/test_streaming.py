"""
Unit-Tests für die Streaming-Funktionalität des EliteBot.
Diese Tests laufen ohne echte WebSocket-Verbindungen.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

# Projekt-Root zum Path hinzufügen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip all tests if dependencies not available
pytest.importorskip("yfinance", reason="yfinance not installed (optional ig dependency)")
pytest.importorskip("trading_ig", reason="trading-ig not installed (optional ig dependency)")

# Prüfe ob Streaming-Module verfügbar sind
try:
    from trading_ig.stream import IGStreamService
    from lightstreamer.client import Subscription, SubscriptionListener
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

# Skip alle Tests wenn Streaming nicht verfügbar
pytestmark = pytest.mark.skipif(
    not STREAMING_AVAILABLE,
    reason="Streaming not available (trading-ig streaming module not installed)"
)


class TestStreamingImports:
    """Tests für die Streaming-Module Imports."""

    def test_streaming_module_exists(self):
        """ig_streaming.py sollte existieren und importierbar sein."""
        from bots.ig.streaming import StreamingManager, StreamingCacheManager, CandleListener
        assert StreamingManager is not None
        assert StreamingCacheManager is not None
        assert CandleListener is not None

    def test_bot_has_streaming_flag(self):
        """Bot sollte STREAMING_AVAILABLE Flag haben."""
        import bots.ig as ig_bot
        assert hasattr(ig_bot, "STREAMING_AVAILABLE")

    def test_bot_init_accepts_use_streaming_param(self):
        """EliteBot.__init__ sollte use_streaming Parameter akzeptieren."""
        from bots.ig import EliteBot
        import inspect

        sig = inspect.signature(EliteBot.__init__)
        params = list(sig.parameters.keys())
        assert "use_streaming" in params


class TestStreamingManagerArchitecture:
    """Tests für StreamingManager Klasse."""

    def test_streaming_manager_has_required_methods(self):
        """StreamingManager sollte alle erforderlichen Methoden haben."""
        from bots.ig.streaming import StreamingManager

        required_methods = [
            "start",
            "stop",
            "_connect",
            "_disconnect",
            "_reconnect",
            "_subscribe_all",
            "_subscribe_symbol",
            "_start_health_monitor",
            "_schedule_reconnect",
            "update_heartbeat",
        ]

        for method in required_methods:
            assert hasattr(StreamingManager, method), f"Methode '{method}' fehlt"
            assert callable(getattr(StreamingManager, method))

    def test_streaming_manager_has_is_connected_property(self):
        """StreamingManager sollte is_connected Property haben."""
        from bots.ig.streaming import StreamingManager

        assert hasattr(StreamingManager, "is_connected")

    def test_streaming_manager_has_candle_fields(self):
        """StreamingManager sollte CANDLE_FIELDS definieren."""
        from bots.ig.streaming import StreamingManager

        assert hasattr(StreamingManager, "CANDLE_FIELDS")
        fields = StreamingManager.CANDLE_FIELDS

        # Erforderliche Felder prüfen
        required = ["UTM", "BID_OPEN", "BID_HIGH", "BID_LOW", "BID_CLOSE",
                    "OFR_OPEN", "OFR_HIGH", "OFR_LOW", "OFR_CLOSE", "CONS_END"]
        for field in required:
            assert field in fields, f"Feld '{field}' fehlt in CANDLE_FIELDS"

    def test_streaming_manager_init_signature(self):
        """StreamingManager.__init__ sollte richtige Parameter haben."""
        from bots.ig.streaming import StreamingManager
        import inspect

        sig = inspect.signature(StreamingManager.__init__)
        params = list(sig.parameters.keys())

        assert "ig_service" in params
        assert "cache_manager" in params
        assert "epics_map" in params
        assert "timeframe" in params

    def test_streaming_manager_has_valid_timeframes(self):
        """StreamingManager sollte gültige Timeframes definieren."""
        from bots.ig.streaming import StreamingManager

        assert hasattr(StreamingManager, "VALID_TIMEFRAMES")
        valid = StreamingManager.VALID_TIMEFRAMES
        assert "HOUR" in valid
        assert "1MINUTE" in valid
        assert "5MINUTE" in valid
        assert "SECOND" in valid

    def test_streaming_manager_default_timeframe_is_hour(self):
        """Standard-Timeframe sollte HOUR sein."""
        from bots.ig.streaming import StreamingManager, StreamingCacheManager

        mock_ig_service = MagicMock()
        mock_bot = MagicMock()
        mock_bot.ohlc_cache = {}
        mock_bot.last_bar_time = {}

        cache_manager = StreamingCacheManager(mock_bot)
        epics_map = {"EURUSD": "CS.D.EURUSD.TODAY.IP"}

        manager = StreamingManager(mock_ig_service, cache_manager, epics_map)
        assert manager.timeframe == "HOUR"

    def test_streaming_manager_accepts_minute_timeframe(self):
        """StreamingManager sollte 1MINUTE Timeframe akzeptieren."""
        from bots.ig.streaming import StreamingManager, StreamingCacheManager

        mock_ig_service = MagicMock()
        mock_bot = MagicMock()
        mock_bot.ohlc_cache = {}
        mock_bot.last_bar_time = {}

        cache_manager = StreamingCacheManager(mock_bot)
        epics_map = {"EURUSD": "CS.D.EURUSD.TODAY.IP"}

        manager = StreamingManager(mock_ig_service, cache_manager, epics_map, timeframe="1MINUTE")
        assert manager.timeframe == "1MINUTE"

    def test_streaming_manager_falls_back_to_hour_for_invalid_timeframe(self):
        """Bei ungültiger Timeframe sollte HOUR verwendet werden."""
        from bots.ig.streaming import StreamingManager, StreamingCacheManager

        mock_ig_service = MagicMock()
        mock_bot = MagicMock()
        mock_bot.ohlc_cache = {}
        mock_bot.last_bar_time = {}

        cache_manager = StreamingCacheManager(mock_bot)
        epics_map = {"EURUSD": "CS.D.EURUSD.TODAY.IP"}

        manager = StreamingManager(mock_ig_service, cache_manager, epics_map, timeframe="INVALID")
        assert manager.timeframe == "HOUR"


class TestStreamingCacheManager:
    """Tests für StreamingCacheManager Klasse."""

    def test_cache_manager_has_required_methods(self):
        """StreamingCacheManager sollte alle erforderlichen Methoden haben."""
        from bots.ig.streaming import StreamingCacheManager

        required_methods = [
            "on_candle_complete",
            "get_cache_stats",
        ]

        for method in required_methods:
            assert hasattr(StreamingCacheManager, method), f"Methode '{method}' fehlt"

    def test_cache_manager_on_candle_complete_updates_bot_cache(self):
        """on_candle_complete sollte Bot-Cache aktualisieren."""
        from bots.ig.streaming import StreamingCacheManager

        # Mock Bot
        mock_bot = MagicMock()
        mock_bot.ohlc_cache = {}
        mock_bot.last_bar_time = {}
        mock_bot.on_streaming_candle = MagicMock()

        cache_manager = StreamingCacheManager(mock_bot, max_candles=100)

        # Simuliere Candle
        candle_time = datetime(2024, 1, 15, 14, 0, 0)
        candle_data = {"O": 1.1000, "H": 1.1050, "L": 1.0950, "C": 1.1020, "UTM": 1705327200000}

        cache_manager.on_candle_complete("EURUSD", candle_time, candle_data)

        # Prüfe dass Cache aktualisiert wurde
        assert "EURUSD" in mock_bot.ohlc_cache
        assert "EURUSD" in mock_bot.last_bar_time
        assert mock_bot.last_bar_time["EURUSD"] == candle_time

        # Prüfe dass Callback aufgerufen wurde
        mock_bot.on_streaming_candle.assert_called_once_with("EURUSD", candle_time, candle_data)

    def test_cache_manager_limits_candle_count(self):
        """Cache sollte auf max_candles begrenzt sein."""
        from bots.ig.streaming import StreamingCacheManager

        mock_bot = MagicMock()
        mock_bot.ohlc_cache = {}
        mock_bot.last_bar_time = {}
        mock_bot.on_streaming_candle = MagicMock()

        cache_manager = StreamingCacheManager(mock_bot, max_candles=5)

        # Füge 10 Candles hinzu
        for i in range(10):
            candle_time = datetime(2024, 1, 15, i, 0, 0)
            candle_data = {"O": 1.1 + i*0.001, "H": 1.11, "L": 1.09, "C": 1.10, "UTM": int(candle_time.timestamp() * 1000)}
            cache_manager.on_candle_complete("EURUSD", candle_time, candle_data)

        # Cache sollte nur 5 Candles haben
        assert len(mock_bot.ohlc_cache["EURUSD"]) == 5

    def test_cache_manager_avoids_duplicates(self):
        """Cache sollte keine Duplikate enthalten."""
        from bots.ig.streaming import StreamingCacheManager

        mock_bot = MagicMock()
        mock_bot.ohlc_cache = {}
        mock_bot.last_bar_time = {}
        mock_bot.on_streaming_candle = MagicMock()

        cache_manager = StreamingCacheManager(mock_bot, max_candles=100)

        # Gleiche Candle zweimal
        candle_time = datetime(2024, 1, 15, 14, 0, 0)
        candle_data = {"O": 1.1000, "H": 1.1050, "L": 1.0950, "C": 1.1020, "UTM": 1705327200000}

        cache_manager.on_candle_complete("EURUSD", candle_time, candle_data)
        cache_manager.on_candle_complete("EURUSD", candle_time, candle_data)

        # Sollte nur eine Candle haben
        assert len(mock_bot.ohlc_cache["EURUSD"]) == 1


class TestCandleListener:
    """Tests für CandleListener Klasse."""

    def test_candle_listener_has_required_methods(self):
        """CandleListener sollte Lightstreamer Listener-Methoden haben."""
        from bots.ig.streaming import CandleListener

        required_methods = [
            "onSubscription",
            "onSubscriptionError",
            "onItemUpdate",
            "onUnsubscription",
        ]

        for method in required_methods:
            assert hasattr(CandleListener, method), f"Methode '{method}' fehlt"

    def test_candle_listener_processes_complete_candle(self):
        """CandleListener sollte abgeschlossene Candles verarbeiten."""
        from bots.ig.streaming import CandleListener

        mock_cache_manager = MagicMock()
        mock_cache_manager.streaming_manager = MagicMock()

        listener = CandleListener(mock_cache_manager, "EURUSD")

        # Simuliere Update mit CONS_END=1
        mock_update = MagicMock()
        mock_update.getValue = MagicMock(side_effect=lambda field: {
            "UTM": "1705327200000",
            "CONS_END": "1",
            "BID_OPEN": "1.1000",
            "BID_HIGH": "1.1050",
            "BID_LOW": "1.0950",
            "BID_CLOSE": "1.1020",
            "OFR_OPEN": "1.1002",
            "OFR_HIGH": "1.1052",
            "OFR_LOW": "1.0952",
            "OFR_CLOSE": "1.1022",
        }.get(field))

        listener.onItemUpdate(mock_update)

        # Prüfe dass on_candle_complete aufgerufen wurde
        mock_cache_manager.on_candle_complete.assert_called_once()

    def test_candle_listener_ignores_incomplete_candle(self):
        """CandleListener sollte unvollständige Candles ignorieren."""
        from bots.ig.streaming import CandleListener

        mock_cache_manager = MagicMock()
        mock_cache_manager.streaming_manager = MagicMock()

        listener = CandleListener(mock_cache_manager, "EURUSD")

        # Simuliere Update ohne CONS_END=1
        mock_update = MagicMock()
        mock_update.getValue = MagicMock(side_effect=lambda field: {
            "UTM": "1705327200000",
            "CONS_END": "0",  # Nicht abgeschlossen
            "BID_CLOSE": "1.1020",
            "OFR_CLOSE": "1.1022",
        }.get(field))

        listener.onItemUpdate(mock_update)

        # on_candle_complete sollte NICHT aufgerufen werden
        mock_cache_manager.on_candle_complete.assert_not_called()


class TestBotStreamingIntegration:
    """Tests für die Bot-Streaming Integration."""

    def test_bot_has_on_streaming_candle_method(self):
        """EliteBot sollte on_streaming_candle Methode haben."""
        from bots.ig import EliteBot

        assert hasattr(EliteBot, "on_streaming_candle")
        assert callable(getattr(EliteBot, "on_streaming_candle"))

    def test_bot_has_run_streaming_method(self):
        """EliteBot sollte run_streaming Methode haben."""
        from bots.ig import EliteBot

        assert hasattr(EliteBot, "run_streaming")
        assert callable(getattr(EliteBot, "run_streaming"))

    def test_bot_run_calls_streaming_when_enabled(self):
        """run() sollte run_streaming aufrufen wenn use_streaming=True."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.run)

        assert "use_streaming" in source or "run_streaming" in source

    def test_on_streaming_candle_checks_for_signals(self):
        """on_streaming_candle sollte auf Signale prüfen."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.on_streaming_candle)

        # Sollte Prediction machen
        assert "predict_proba" in source
        # Sollte auf Threshold prüfen
        assert "conf_thresh" in source
        # Sollte Order ausführen können
        assert "execute_order" in source or "execute_order_fast" in source

    def test_on_streaming_candle_prevents_duplicate_signals(self):
        """on_streaming_candle sollte mehrfache Signale pro Stunde verhindern."""
        from bots.ig import EliteBot
        import inspect

        source = inspect.getsource(EliteBot.on_streaming_candle)

        assert "last_signal_hour" in source


class TestReconnectLogic:
    """Tests für die Reconnect-Logik."""

    def test_streaming_manager_has_reconnect_attributes(self):
        """StreamingManager sollte Reconnect-Attribute haben."""
        from bots.ig.streaming import StreamingManager
        import inspect

        source = inspect.getsource(StreamingManager.__init__)

        assert "_reconnect_attempts" in source
        assert "_max_reconnect_attempts" in source
        assert "_reconnect_delay" in source

    def test_reconnect_resets_attempt_counter(self):
        """Erfolgreicher Connect sollte Attempt-Counter zurücksetzen."""
        from bots.ig.streaming import StreamingManager
        import inspect

        source = inspect.getsource(StreamingManager._connect)

        assert "_reconnect_attempts" in source
        assert "= 0" in source  # Reset auf 0

    def test_health_monitor_checks_heartbeat(self):
        """Health Monitor sollte Heartbeat prüfen."""
        from bots.ig.streaming import StreamingManager
        import inspect

        source = inspect.getsource(StreamingManager._start_health_monitor)

        assert "_last_heartbeat" in source
        assert "_reconnect" in source


class TestStreamingManagerMocked:
    """Tests mit gemocktem StreamingManager."""

    def test_start_connects_and_subscribes(self):
        """start() sollte verbinden und subscriben."""
        from bots.ig.streaming import StreamingManager, StreamingCacheManager

        mock_ig_service = MagicMock()
        mock_bot = MagicMock()
        mock_bot.ohlc_cache = {}
        mock_bot.last_bar_time = {}

        cache_manager = StreamingCacheManager(mock_bot)
        epics_map = {"EURUSD": "CS.D.EURUSD.TODAY.IP"}

        with patch.object(StreamingManager, '_connect') as mock_connect, \
             patch.object(StreamingManager, '_subscribe_all') as mock_subscribe, \
             patch.object(StreamingManager, '_start_health_monitor') as mock_monitor:

            manager = StreamingManager(mock_ig_service, cache_manager, epics_map)
            manager.start()

            mock_connect.assert_called_once()
            mock_subscribe.assert_called_once()
            mock_monitor.assert_called_once()

    def test_stop_disconnects(self):
        """stop() sollte Verbindung trennen."""
        from bots.ig.streaming import StreamingManager, StreamingCacheManager

        mock_ig_service = MagicMock()
        mock_bot = MagicMock()
        mock_bot.ohlc_cache = {}
        mock_bot.last_bar_time = {}

        cache_manager = StreamingCacheManager(mock_bot)
        epics_map = {"EURUSD": "CS.D.EURUSD.TODAY.IP"}

        with patch.object(StreamingManager, '_disconnect') as mock_disconnect:
            manager = StreamingManager(mock_ig_service, cache_manager, epics_map)
            manager.stop()

            mock_disconnect.assert_called_once()


class TestCacheStatsMethod:
    """Tests für get_cache_stats Methode."""

    def test_get_cache_stats_returns_dict(self):
        """get_cache_stats sollte Dict zurückgeben."""
        from bots.ig.streaming import StreamingCacheManager

        mock_bot = MagicMock()
        mock_bot.ohlc_cache = {
            "EURUSD": pd.DataFrame({
                "O": [1.1, 1.2],
                "H": [1.15, 1.25],
                "L": [1.05, 1.15],
                "C": [1.12, 1.22]
            }, index=[datetime(2024, 1, 15, 14, 0), datetime(2024, 1, 15, 15, 0)])
        }

        cache_manager = StreamingCacheManager(mock_bot)
        stats = cache_manager.get_cache_stats()

        assert isinstance(stats, dict)
        assert "EURUSD" in stats
        assert stats["EURUSD"]["count"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
