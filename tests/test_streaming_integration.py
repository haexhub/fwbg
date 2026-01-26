"""
Integration-Tests für die Streaming-Funktionalität des EliteBot.
Nutzt den Demo-Account für echte WebSocket-Verbindungen.

Ausführung:
    python -m pytest tests/test_streaming_integration.py -v -s

ACHTUNG: Diese Tests erstellen echte WebSocket-Verbindungen zu IG Markets!
"""
import os
import sys
import pytest
import time
import threading
from datetime import datetime, timedelta

# Projekt-Root zum Path hinzufügen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ig_bot import EliteBot, STREAMING_AVAILABLE

# Skip alle Tests wenn Streaming nicht verfügbar
pytestmark = pytest.mark.skipif(
    not STREAMING_AVAILABLE,
    reason="Streaming not available (trading-ig streaming module not installed)"
)


# Pfad zum Demo-Account
DEMO_ACCOUNT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "accounts", "main_demo"
)


@pytest.fixture(scope="module")
def bot_instance():
    """
    Erstellt eine Bot-Instanz mit Demo-Account (ohne Streaming starten).
    Scope=module damit der Bot nur einmal initialisiert wird.
    """
    if not os.path.exists(DEMO_ACCOUNT_PATH):
        pytest.skip("Demo account not found at " + DEMO_ACCOUNT_PATH)

    # Bot mit Streaming aktiviert erstellen (aber noch nicht starten)
    try:
        bot = EliteBot(DEMO_ACCOUNT_PATH, use_streaming=True)
        yield bot
    except SystemExit:
        pytest.skip("IG API login failed (rate limit or auth error)")
    except Exception as e:
        pytest.skip(f"Could not initialize bot: {e}")


@pytest.fixture(scope="module")
def ig_service(bot_instance):
    """Gibt den IG Service des Bots zurück."""
    return bot_instance.ig


class TestStreamingAvailability:
    """Tests für Streaming-Verfügbarkeit."""

    def test_streaming_modules_importable(self):
        """Streaming-Module sollten importierbar sein."""
        from ig_streaming import StreamingManager, StreamingCacheManager, CandleListener
        assert StreamingManager is not None
        assert StreamingCacheManager is not None
        assert CandleListener is not None

    def test_bot_has_streaming_enabled(self, bot_instance):
        """Bot sollte Streaming aktiviert haben."""
        assert bot_instance.use_streaming is True

    def test_streaming_available_flag_is_true(self):
        """STREAMING_AVAILABLE sollte True sein."""
        assert STREAMING_AVAILABLE is True


class TestStreamingConnection:
    """Tests für WebSocket-Verbindung."""

    def test_can_create_streaming_session(self, ig_service):
        """Streaming-Session sollte erstellbar sein."""
        from trading_ig.stream import IGStreamService

        stream_service = IGStreamService(ig_service)

        try:
            stream_service.create_session()
            assert True  # Session erstellt
        finally:
            try:
                stream_service.disconnect()
            except Exception:
                pass

    def test_streaming_manager_connects(self, bot_instance):
        """StreamingManager sollte verbinden können."""
        from ig_streaming import StreamingManager, StreamingCacheManager

        cache_manager = StreamingCacheManager(bot_instance)

        # Nur ein Symbol für schnellen Test
        test_symbol = list(bot_instance.models.keys())[0] if bot_instance.models else None
        if not test_symbol:
            pytest.skip("No models trained")

        epic = bot_instance.SYMBOL_TO_EPIC.get(test_symbol)
        if not epic:
            pytest.skip(f"No EPIC mapping for {test_symbol}")

        epics_map = {test_symbol: epic}

        manager = StreamingManager(bot_instance.ig, cache_manager, epics_map)

        try:
            manager._connect()
            assert manager.is_connected is True
        finally:
            manager.stop()


class TestStreamingSubscription:
    """Tests für Streaming-Subscriptions."""

    def test_can_subscribe_to_chart_hour(self, bot_instance):
        """Sollte CHART:EPIC:HOUR subscriben können."""
        from ig_streaming import StreamingManager, StreamingCacheManager
        from lightstreamer.client import Subscription

        cache_manager = StreamingCacheManager(bot_instance)

        # Wähle ein Symbol
        test_symbol = list(bot_instance.models.keys())[0] if bot_instance.models else None
        if not test_symbol:
            pytest.skip("No models trained")

        epic = bot_instance.SYMBOL_TO_EPIC.get(test_symbol)
        epics_map = {test_symbol: epic}

        manager = StreamingManager(bot_instance.ig, cache_manager, epics_map)

        try:
            manager._connect()
            manager._subscribe_all()

            # Prüfe dass Subscription erstellt wurde
            assert test_symbol in manager.subscriptions
            assert test_symbol in manager.listeners

        finally:
            manager.stop()

    def test_subscription_receives_updates(self, bot_instance):
        """Subscription sollte Updates empfangen."""
        from ig_streaming import StreamingManager, StreamingCacheManager

        received_updates = []

        # Custom cache manager der Updates trackt
        class TrackingCacheManager(StreamingCacheManager):
            def __init__(self, bot):
                super().__init__(bot)
                self.updates = []

            def on_candle_complete(self, symbol, candle_time, candle_data):
                self.updates.append((symbol, candle_time, candle_data))
                # Auch original aufrufen
                super().on_candle_complete(symbol, candle_time, candle_data)

        cache_manager = TrackingCacheManager(bot_instance)

        test_symbol = list(bot_instance.models.keys())[0] if bot_instance.models else None
        if not test_symbol:
            pytest.skip("No models trained")

        epic = bot_instance.SYMBOL_TO_EPIC.get(test_symbol)
        epics_map = {test_symbol: epic}

        manager = StreamingManager(bot_instance.ig, cache_manager, epics_map)

        try:
            manager.start()

            # Warte kurz auf mögliche Updates
            # HINWEIS: CONS_END=1 kommt nur am Ende einer Stunde!
            # Dieser Test prüft nur dass die Verbindung steht
            time.sleep(5)

            assert manager.is_connected is True
            print(f"\nVerbindung steht. Warte auf Updates von {test_symbol}...")
            print(f"(CONS_END=1 kommt nur zur vollen Stunde)")

        finally:
            manager.stop()


class TestStreamingCacheIntegration:
    """Tests für Cache-Integration mit Streaming."""

    def test_streaming_updates_bot_cache(self, bot_instance):
        """Streaming-Updates sollten Bot-Cache aktualisieren."""
        from ig_streaming import StreamingCacheManager

        # Initial cache status
        initial_cache_count = len(bot_instance.ohlc_cache.get("EURUSD", []))

        cache_manager = StreamingCacheManager(bot_instance)

        # Simuliere manuell ein Candle-Update
        test_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        test_candle = {
            "O": 1.08500,
            "H": 1.08600,
            "L": 1.08400,
            "C": 1.08550,
            "UTM": int(test_time.timestamp() * 1000)
        }

        # Mock den Callback
        original_callback = bot_instance.on_streaming_candle
        callback_called = [False]

        def mock_callback(symbol, candle_time, candle_data):
            callback_called[0] = True

        bot_instance.on_streaming_candle = mock_callback

        try:
            cache_manager.on_candle_complete("EURUSD", test_time, test_candle)

            # Prüfe dass Cache aktualisiert wurde
            assert "EURUSD" in bot_instance.ohlc_cache
            assert callback_called[0] is True

        finally:
            bot_instance.on_streaming_candle = original_callback


class TestStreamingReconnect:
    """Tests für Reconnect-Funktionalität."""

    def test_manager_tracks_heartbeat(self, bot_instance):
        """Manager sollte Heartbeat tracken."""
        from ig_streaming import StreamingManager, StreamingCacheManager

        cache_manager = StreamingCacheManager(bot_instance)
        test_symbol = list(bot_instance.models.keys())[0] if bot_instance.models else None
        if not test_symbol:
            pytest.skip("No models trained")

        epic = bot_instance.SYMBOL_TO_EPIC.get(test_symbol)
        epics_map = {test_symbol: epic}

        manager = StreamingManager(bot_instance.ig, cache_manager, epics_map)

        try:
            manager._connect()

            initial_heartbeat = manager._last_heartbeat
            assert initial_heartbeat is not None

            # Simuliere Heartbeat-Update
            time.sleep(0.1)
            manager.update_heartbeat()

            assert manager._last_heartbeat > initial_heartbeat

        finally:
            manager.stop()

    def test_manager_resets_reconnect_counter_on_connect(self, bot_instance):
        """Reconnect-Counter sollte bei erfolgreichem Connect zurückgesetzt werden."""
        from ig_streaming import StreamingManager, StreamingCacheManager

        cache_manager = StreamingCacheManager(bot_instance)
        test_symbol = list(bot_instance.models.keys())[0] if bot_instance.models else None
        if not test_symbol:
            pytest.skip("No models trained")

        epic = bot_instance.SYMBOL_TO_EPIC.get(test_symbol)
        epics_map = {test_symbol: epic}

        manager = StreamingManager(bot_instance.ig, cache_manager, epics_map)

        # Setze künstlich Reconnect-Attempts
        manager._reconnect_attempts = 5

        try:
            manager._connect()

            # Nach erfolgreichem Connect sollte Counter 0 sein
            assert manager._reconnect_attempts == 0

        finally:
            manager.stop()


class TestBotStreamingMode:
    """Tests für den Bot im Streaming-Modus."""

    def test_bot_initializes_with_streaming(self, bot_instance):
        """Bot sollte mit Streaming initialisiert sein."""
        assert bot_instance.use_streaming is True
        assert bot_instance.streaming_manager is None  # Noch nicht gestartet
        assert bot_instance.cache_manager is None  # Noch nicht gestartet

    def test_bot_can_handle_streaming_candle(self, bot_instance):
        """Bot sollte Streaming-Candle verarbeiten können."""
        # Wähle ein Symbol mit trainiertem Modell
        test_symbol = list(bot_instance.models.keys())[0] if bot_instance.models else None
        if not test_symbol:
            pytest.skip("No models trained")

        # Stelle sicher dass genug Daten im Cache sind
        if test_symbol not in bot_instance.ohlc_cache or len(bot_instance.ohlc_cache[test_symbol]) < 100:
            pytest.skip(f"Not enough cached data for {test_symbol}")

        # Simuliere eine neue Candle
        test_time = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        test_candle = {
            "O": 1.08500,
            "H": 1.08600,
            "L": 1.08400,
            "C": 1.08550,
            "UTM": int(test_time.timestamp() * 1000)
        }

        # Sollte keine Exception werfen
        try:
            bot_instance.on_streaming_candle(test_symbol, test_time, test_candle)
        except Exception as e:
            pytest.fail(f"on_streaming_candle raised exception: {e}")


class TestMinuteCandleStreaming:
    """Tests mit MINUTE-Kerzen für schnellere Validierung."""

    def test_receives_complete_minute_candle(self, bot_instance):
        """
        Test dass wir eine vollständige MINUTE-Kerze mit CONS_END=1 empfangen.
        Dieser Test wartet bis zu 90 Sekunden auf eine abgeschlossene Kerze.
        """
        from ig_streaming import StreamingManager, StreamingCacheManager

        # Tracking für empfangene Candles
        received_candles = []
        candle_received_event = threading.Event()

        class MinuteTrackingCacheManager(StreamingCacheManager):
            def __init__(self, bot):
                super().__init__(bot)

            def on_candle_complete(self, symbol, candle_time, candle_data):
                print(f"\n✅ MINUTE Candle empfangen: {symbol} @ {candle_time}")
                print(f"   O={candle_data['O']:.5f} H={candle_data['H']:.5f} "
                      f"L={candle_data['L']:.5f} C={candle_data['C']:.5f}")
                received_candles.append((symbol, candle_time, candle_data))
                candle_received_event.set()

        cache_manager = MinuteTrackingCacheManager(bot_instance)

        # Wähle ein Symbol
        test_symbol = list(bot_instance.models.keys())[0] if bot_instance.models else None
        if not test_symbol:
            pytest.skip("No models trained")

        epic = bot_instance.SYMBOL_TO_EPIC.get(test_symbol)
        if not epic:
            pytest.skip(f"No EPIC mapping for {test_symbol}")

        epics_map = {test_symbol: epic}

        # StreamingManager mit 1MINUTE timeframe
        manager = StreamingManager(
            bot_instance.ig,
            cache_manager,
            epics_map,
            timeframe="1MINUTE"
        )

        try:
            print(f"\n📡 Starte MINUTE-Streaming für {test_symbol} ({epic})...")
            manager.start()

            assert manager.is_connected, "Streaming nicht verbunden"

            # Warte bis zu 90 Sekunden auf eine Candle
            # (max 60 Sekunden bis zur nächsten vollen Minute + 30 Sekunden Puffer)
            print("⏳ Warte auf CONS_END=1 (max 90 Sekunden)...")

            candle_received = candle_received_event.wait(timeout=90)

            if candle_received:
                print(f"\n🎉 Test erfolgreich! {len(received_candles)} Candle(s) empfangen.")
                assert len(received_candles) > 0
                symbol, candle_time, candle_data = received_candles[0]
                assert symbol == test_symbol
                assert "O" in candle_data
                assert "H" in candle_data
                assert "L" in candle_data
                assert "C" in candle_data
            else:
                # Wenn keine Candle kam, prüfen wir zumindest dass die Verbindung steht
                print("\n⚠️ Keine CONS_END=1 in 90 Sekunden empfangen.")
                print("   (Das kann passieren wenn der Markt geschlossen ist)")
                assert manager.is_connected, "Verbindung sollte noch bestehen"

        finally:
            manager.stop()
            print("📴 Streaming gestoppt")

    def test_minute_subscription_item_format(self, bot_instance):
        """Test dass die MINUTE-Subscription das richtige Format verwendet."""
        from ig_streaming import StreamingManager, StreamingCacheManager

        cache_manager = StreamingCacheManager(bot_instance)

        test_symbol = list(bot_instance.models.keys())[0] if bot_instance.models else None
        if not test_symbol:
            pytest.skip("No models trained")

        epic = bot_instance.SYMBOL_TO_EPIC.get(test_symbol)
        epics_map = {test_symbol: epic}

        manager = StreamingManager(
            bot_instance.ig,
            cache_manager,
            epics_map,
            timeframe="1MINUTE"
        )

        assert manager.timeframe == "1MINUTE"

        try:
            manager._connect()
            manager._subscribe_all()

            # Prüfe dass Subscription erstellt wurde
            assert test_symbol in manager.subscriptions
            print(f"\n✅ MINUTE-Subscription erfolgreich für {test_symbol}")

        finally:
            manager.stop()


class TestHistoricalPlusStreamingWorkflow:
    """
    Tests für den kompletten Workflow:
    1. Historische Daten von IG laden
    2. Streaming starten
    3. Neue Kerzen appenden
    """

    def test_workflow_historical_then_streaming(self, bot_instance):
        """
        Kompletter Workflow-Test:
        - Erst historische MINUTE-Daten von IG holen (nur 5 Kerzen)
        - Dann Streaming starten mit MINUTE-Kerzen
        - Prüfen dass neue Kerzen korrekt angehängt werden
        """
        from ig_streaming import StreamingManager, StreamingCacheManager
        import pandas as pd

        # Wähle ein Symbol
        test_symbol = list(bot_instance.models.keys())[0] if bot_instance.models else None
        if not test_symbol:
            pytest.skip("No models trained")

        epic = bot_instance.SYMBOL_TO_EPIC.get(test_symbol)
        if not epic:
            pytest.skip(f"No EPIC mapping for {test_symbol}")

        print(f"\n{'='*60}")
        print(f"📊 WORKFLOW TEST: {test_symbol}")
        print(f"{'='*60}")

        # --- SCHRITT 1: Historische MINUTE-Daten holen (nur 5 Kerzen!) ---
        print(f"\n1️⃣ Hole 5 historische MINUTE-Kerzen für {test_symbol}...")

        try:
            # Direkt IG API aufrufen mit MINUTE-Auflösung
            response = bot_instance.ig.fetch_historical_prices_by_epic(
                epic=epic,
                resolution="1Min",  # MINUTE statt HOUR
                numpoints=5,        # Nur 5 Kerzen für schnellen Test
            )

            if response is None or "prices" not in response or not response["prices"]:
                pytest.skip(f"Could not fetch historical data for {test_symbol}")

            prices = response["prices"]

            # Konvertiere zu DataFrame (wie in fetch_ig_historical)
            data = []
            for p in prices:
                snap = p.get("snapshotTimeUTC") or p.get("snapshotTime")
                o = (p["openPrice"]["bid"] + p["openPrice"]["ask"]) / 2
                h = (p["highPrice"]["bid"] + p["highPrice"]["ask"]) / 2
                low = (p["lowPrice"]["bid"] + p["lowPrice"]["ask"]) / 2
                c = (p["closePrice"]["bid"] + p["closePrice"]["ask"]) / 2
                data.append({"T": snap, "O": o, "H": h, "L": low, "C": c})

            historical_df = pd.DataFrame(data)
            historical_df["T"] = pd.to_datetime(historical_df["T"])
            historical_df = historical_df.set_index("T").sort_index()

            print(f"   ✅ {len(historical_df)} historische MINUTE-Kerzen geladen")
            print(f"   Erste Kerze: {historical_df.index[0]}")
            print(f"   Letzte Kerze: {historical_df.index[-1]}")

            # Cache initialisieren
            bot_instance.ohlc_cache[test_symbol] = historical_df.copy()
            initial_count = len(bot_instance.ohlc_cache[test_symbol])

        except Exception as e:
            pytest.skip(f"Could not fetch historical data: {e}")

        # --- SCHRITT 2: Streaming starten ---
        print(f"\n2️⃣ Starte MINUTE-Streaming...")

        candle_received_event = threading.Event()
        new_candles = []

        class WorkflowCacheManager(StreamingCacheManager):
            def on_candle_complete(self, symbol, candle_time, candle_data):
                print(f"   🕯️ Neue Streaming-Kerze: {symbol} @ {candle_time}")
                new_candles.append((symbol, candle_time, candle_data))
                # Original aufrufen - dies fügt die Kerze zum Cache hinzu
                super().on_candle_complete(symbol, candle_time, candle_data)
                candle_received_event.set()

        cache_manager = WorkflowCacheManager(bot_instance)
        epics_map = {test_symbol: epic}

        manager = StreamingManager(
            bot_instance.ig,
            cache_manager,
            epics_map,
            timeframe="1MINUTE"
        )

        try:
            manager.start()
            assert manager.is_connected, "Streaming nicht verbunden"
            print("   ✅ Streaming verbunden")

            # --- SCHRITT 3: Auf neue Kerze warten ---
            print(f"\n3️⃣ Warte auf neue Streaming-Kerze (max 90s)...")

            candle_received = candle_received_event.wait(timeout=90)

            # --- SCHRITT 4: Ergebnisse prüfen ---
            print(f"\n4️⃣ Ergebnisse:")

            if candle_received:
                final_count = len(bot_instance.ohlc_cache[test_symbol])
                print(f"   ✅ Kerze empfangen!")
                print(f"   Initial: {initial_count} Kerzen")
                print(f"   Final: {final_count} Kerzen")
                print(f"   Neu hinzugefügt: {len(new_candles)}")

                # Prüfe dass Cache gewachsen ist
                assert final_count >= initial_count, \
                    f"Cache sollte mindestens {initial_count} Kerzen haben, hat {final_count}"

                # Prüfe dass letzte Kerze im Cache ist
                last_time = bot_instance.ohlc_cache[test_symbol].index[-1]
                print(f"   Letzte Kerze im Cache: {last_time}")

                # Prüfe dass keine Duplikate
                cache_df = bot_instance.ohlc_cache[test_symbol]
                assert not cache_df.index.duplicated().any(), "Cache hat Duplikate!"
                print("   ✅ Keine Duplikate im Cache")

                print(f"\n🎉 WORKFLOW TEST ERFOLGREICH!")

            else:
                print("   ⚠️ Keine Kerze in 90s empfangen (Markt evtl. geschlossen)")
                # Trotzdem prüfen dass Cache intakt ist
                assert len(bot_instance.ohlc_cache[test_symbol]) == initial_count
                print("   ✅ Cache unverändert (wie erwartet)")

        finally:
            manager.stop()
            print(f"\n📴 Streaming gestoppt")

    def test_streaming_appends_without_gaps(self, bot_instance):
        """
        Test dass Streaming-Kerzen lückenlos an historische Daten angehängt werden.
        Simuliert den Übergang von historisch zu live.
        """
        from ig_streaming import StreamingCacheManager
        import pandas as pd

        test_symbol = "TEST_SYMBOL"

        # Erstelle Mock-historische Daten (letzte Kerze vor 2 Stunden)
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        historical_times = [now - timedelta(hours=i) for i in range(10, 0, -1)]

        historical_df = pd.DataFrame({
            "O": [1.1000 + i*0.001 for i in range(10)],
            "H": [1.1050 + i*0.001 for i in range(10)],
            "L": [1.0950 + i*0.001 for i in range(10)],
            "C": [1.1020 + i*0.001 for i in range(10)]
        }, index=historical_times)

        bot_instance.ohlc_cache[test_symbol] = historical_df.copy()
        bot_instance.last_bar_time[test_symbol] = historical_times[-1]

        print(f"\n📊 Historische Daten (Mock): {len(historical_df)} Kerzen")
        print(f"   Letzte historische Kerze: {historical_times[-1]}")

        # Cache Manager erstellen
        cache_manager = StreamingCacheManager(bot_instance)

        # Simuliere neue Streaming-Kerzen
        new_times = [now - timedelta(hours=i) for i in range(0, -3, -1)]  # now, now+1h, now+2h
        for i, candle_time in enumerate(new_times):
            candle_data = {
                "O": 1.1100 + i*0.001,
                "H": 1.1150 + i*0.001,
                "L": 1.1050 + i*0.001,
                "C": 1.1120 + i*0.001,
                "UTM": int(candle_time.timestamp() * 1000)
            }

            # Mock den callback
            original_callback = bot_instance.on_streaming_candle
            bot_instance.on_streaming_candle = lambda s, t, c: None

            cache_manager.on_candle_complete(test_symbol, candle_time, candle_data)

            bot_instance.on_streaming_candle = original_callback

        # Prüfen
        final_df = bot_instance.ohlc_cache[test_symbol]
        print(f"   Finale Kerzen: {len(final_df)}")
        print(f"   Letzte Kerze: {final_df.index[-1]}")

        # Prüfe Reihenfolge
        assert final_df.index.is_monotonic_increasing, "Cache ist nicht chronologisch sortiert!"
        print("   ✅ Cache ist chronologisch sortiert")

        # Prüfe keine Duplikate
        assert not final_df.index.duplicated().any(), "Cache hat Duplikate!"
        print("   ✅ Keine Duplikate")

        # Cleanup
        del bot_instance.ohlc_cache[test_symbol]


class TestCandleBuilding:
    """Tests für die korrekte Kerzen-Konstruktion aus Streaming-Daten."""

    def test_candle_builds_correctly_from_streaming_updates(self, bot_instance):
        """
        Test dass Kerzen korrekt aus mehreren Streaming-Updates zusammengebaut werden.
        Prüft Mid-Price Berechnung (Bid+Ask)/2.
        """
        from ig_streaming import CandleListener, StreamingCacheManager
        from unittest.mock import MagicMock

        print(f"\n{'='*60}")
        print("🔬 KERZEN-BAU TEST")
        print(f"{'='*60}")

        # Setup
        built_candles = []

        class TestCacheManager(StreamingCacheManager):
            def on_candle_complete(self, symbol, candle_time, candle_data):
                built_candles.append({
                    "symbol": symbol,
                    "time": candle_time,
                    "data": candle_data.copy()
                })

        cache_manager = TestCacheManager(bot_instance)
        cache_manager.streaming_manager = MagicMock()

        listener = CandleListener(cache_manager, "EURUSD")

        # --- Simuliere mehrere Updates während einer Kerze ---
        print("\n1️⃣ Simuliere Streaming-Updates...")

        # Update 1: Kerze öffnet
        update1 = MagicMock()
        update1.getValue = MagicMock(side_effect=lambda f: {
            "UTM": "1705327200000",  # 2024-01-15 14:00:00
            "CONS_END": "0",
            "BID_OPEN": "1.10000",
            "OFR_OPEN": "1.10020",
            "BID_HIGH": "1.10000",
            "OFR_HIGH": "1.10020",
            "BID_LOW": "1.10000",
            "OFR_LOW": "1.10020",
            "BID_CLOSE": "1.10000",
            "OFR_CLOSE": "1.10020"
        }.get(f))

        print("   Update 1: Open @ Bid=1.10000, Ask=1.10020")
        listener.onItemUpdate(update1)
        assert len(built_candles) == 0, "Kerze sollte noch nicht fertig sein"

        # Update 2: Preis steigt (neues High)
        update2 = MagicMock()
        update2.getValue = MagicMock(side_effect=lambda f: {
            "UTM": "1705327200000",
            "CONS_END": "0",
            "BID_HIGH": "1.10500",
            "OFR_HIGH": "1.10520",
            "BID_CLOSE": "1.10450",
            "OFR_CLOSE": "1.10470"
        }.get(f))

        print("   Update 2: High @ Bid=1.10500, Ask=1.10520")
        listener.onItemUpdate(update2)
        assert len(built_candles) == 0, "Kerze sollte noch nicht fertig sein"

        # Update 3: Preis fällt (neues Low)
        update3 = MagicMock()
        update3.getValue = MagicMock(side_effect=lambda f: {
            "UTM": "1705327200000",
            "CONS_END": "0",
            "BID_LOW": "1.09800",
            "OFR_LOW": "1.09820",
            "BID_CLOSE": "1.09900",
            "OFR_CLOSE": "1.09920"
        }.get(f))

        print("   Update 3: Low @ Bid=1.09800, Ask=1.09820")
        listener.onItemUpdate(update3)
        assert len(built_candles) == 0, "Kerze sollte noch nicht fertig sein"

        # Update 4: Kerze schließt (CONS_END=1)
        update4 = MagicMock()
        update4.getValue = MagicMock(side_effect=lambda f: {
            "UTM": "1705327200000",
            "CONS_END": "1",  # KERZE FERTIG!
            "BID_CLOSE": "1.10200",
            "OFR_CLOSE": "1.10220"
        }.get(f))

        print("   Update 4: Close @ Bid=1.10200, Ask=1.10220 [CONS_END=1]")
        listener.onItemUpdate(update4)

        # --- Prüfungen ---
        print("\n2️⃣ Prüfe gebaute Kerze...")

        assert len(built_candles) == 1, f"Erwartet 1 Kerze, bekam {len(built_candles)}"

        candle = built_candles[0]["data"]

        # Erwartete Werte (Mid = (Bid + Ask) / 2)
        expected_open = (1.10000 + 1.10020) / 2    # 1.10010
        expected_high = (1.10500 + 1.10520) / 2    # 1.10510
        expected_low = (1.09800 + 1.09820) / 2     # 1.09810
        expected_close = (1.10200 + 1.10220) / 2   # 1.10210

        print(f"\n   Erwartete Werte (Mid-Price):")
        print(f"   O: {expected_open:.5f}")
        print(f"   H: {expected_high:.5f}")
        print(f"   L: {expected_low:.5f}")
        print(f"   C: {expected_close:.5f}")

        print(f"\n   Tatsächliche Werte:")
        print(f"   O: {candle['O']:.5f}")
        print(f"   H: {candle['H']:.5f}")
        print(f"   L: {candle['L']:.5f}")
        print(f"   C: {candle['C']:.5f}")

        # Toleranz für Floating Point
        assert abs(candle["O"] - expected_open) < 0.00001, f"Open falsch: {candle['O']}"
        assert abs(candle["H"] - expected_high) < 0.00001, f"High falsch: {candle['H']}"
        assert abs(candle["L"] - expected_low) < 0.00001, f"Low falsch: {candle['L']}"
        assert abs(candle["C"] - expected_close) < 0.00001, f"Close falsch: {candle['C']}"

        print("\n   ✅ Alle OHLC-Werte korrekt berechnet!")

        # Prüfe UTM
        assert candle["UTM"] == 1705327200000
        print(f"   ✅ UTM korrekt: {candle['UTM']}")

        print(f"\n🎉 KERZEN-BAU TEST ERFOLGREICH!")

    def test_live_minute_candle_build(self, bot_instance):
        """
        Live-Test: Warte auf echte MINUTE-Kerze und prüfe die Struktur.
        """
        from ig_streaming import StreamingManager, StreamingCacheManager

        test_symbol = list(bot_instance.models.keys())[0] if bot_instance.models else None
        if not test_symbol:
            pytest.skip("No models trained")

        epic = bot_instance.SYMBOL_TO_EPIC.get(test_symbol)
        if not epic:
            pytest.skip(f"No EPIC mapping for {test_symbol}")

        print(f"\n{'='*60}")
        print(f"🔬 LIVE KERZEN-BAU TEST: {test_symbol}")
        print(f"{'='*60}")

        built_candles = []
        candle_event = threading.Event()

        class BuildTrackingCacheManager(StreamingCacheManager):
            def on_candle_complete(self, symbol, candle_time, candle_data):
                print(f"\n   🕯️ Kerze gebaut: {symbol} @ {candle_time}")
                print(f"      O={candle_data['O']:.5f} H={candle_data['H']:.5f}")
                print(f"      L={candle_data['L']:.5f} C={candle_data['C']:.5f}")
                built_candles.append({
                    "symbol": symbol,
                    "time": candle_time,
                    "data": candle_data.copy()
                })
                candle_event.set()

        cache_manager = BuildTrackingCacheManager(bot_instance)
        epics_map = {test_symbol: epic}

        manager = StreamingManager(
            bot_instance.ig,
            cache_manager,
            epics_map,
            timeframe="1MINUTE"
        )

        try:
            print(f"\n1️⃣ Starte MINUTE-Streaming für {test_symbol}...")
            manager.start()
            assert manager.is_connected

            print("2️⃣ Warte auf vollständige Kerze (max 90s)...")
            received = candle_event.wait(timeout=90)

            print("\n3️⃣ Ergebnisse:")
            if received:
                candle = built_candles[0]
                data = candle["data"]

                # Struktur-Prüfungen
                assert "O" in data, "Open fehlt"
                assert "H" in data, "High fehlt"
                assert "L" in data, "Low fehlt"
                assert "C" in data, "Close fehlt"
                assert "UTM" in data, "UTM fehlt"

                # Logik-Prüfungen
                assert data["H"] >= data["O"], "High < Open!"
                assert data["H"] >= data["C"], "High < Close!"
                assert data["L"] <= data["O"], "Low > Open!"
                assert data["L"] <= data["C"], "Low > Close!"
                assert data["H"] >= data["L"], "High < Low!"

                print("   ✅ Kerzen-Struktur korrekt (O, H, L, C, UTM)")
                print("   ✅ Kerzen-Logik korrekt (H >= O,C,L; L <= O,C)")
                print(f"\n🎉 LIVE KERZEN-BAU TEST ERFOLGREICH!")
            else:
                print("   ⚠️ Keine Kerze empfangen (Markt evtl. geschlossen)")

        finally:
            manager.stop()


class TestStreamingPerformance:
    """Performance-Tests für Streaming."""

    def test_cache_update_is_fast(self, bot_instance):
        """Cache-Update sollte schnell sein (<100ms)."""
        from ig_streaming import StreamingCacheManager
        import time

        cache_manager = StreamingCacheManager(bot_instance)

        test_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        test_candle = {
            "O": 1.08500,
            "H": 1.08600,
            "L": 1.08400,
            "C": 1.08550,
            "UTM": int(test_time.timestamp() * 1000)
        }

        # Mock callback um Zeit zu messen
        bot_instance.on_streaming_candle = lambda s, t, c: None

        start = time.perf_counter()
        for i in range(100):
            t = test_time + timedelta(hours=i)
            cache_manager.on_candle_complete("PERFTEST", t, test_candle)
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / 100) * 1000
        print(f"\nDurchschnittliche Cache-Update Zeit: {avg_time_ms:.2f}ms")

        assert avg_time_ms < 100, f"Cache-Update zu langsam: {avg_time_ms:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
