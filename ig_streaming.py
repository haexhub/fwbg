"""
IG Markets WebSocket Streaming Module.
Handles real-time candle data via Lightstreamer with automatic reconnection.
"""

import logging
import threading
import time
import pandas as pd
from datetime import datetime
from trading_ig.stream import IGStreamService
from lightstreamer.client import Subscription, SubscriptionListener

logger = logging.getLogger("FortressBot")


class CandleListener(SubscriptionListener):
    """
    Lightstreamer Listener for candle updates.
    Collects candle data and signals when a new candle is complete.
    """

    def __init__(self, cache_manager, symbol):
        self.cache_manager = cache_manager
        self.symbol = symbol
        self.current_candle = {}
        self.last_utm = None

    def onSubscription(self):
        logger.info(f"📡 {self.symbol}: Streaming subscription active")

    def onSubscriptionError(self, code, message):
        logger.error(f"❌ {self.symbol}: Subscription error {code}: {message}")

    def onItemUpdate(self, update):
        """Called on each candle update from Lightstreamer."""
        try:
            # Update heartbeat in streaming manager
            if self.cache_manager.streaming_manager:
                self.cache_manager.streaming_manager.update_heartbeat()

            utm = update.getValue("UTM")
            cons_end = update.getValue("CONS_END")

            bid_open = update.getValue("BID_OPEN")
            bid_high = update.getValue("BID_HIGH")
            bid_low = update.getValue("BID_LOW")
            bid_close = update.getValue("BID_CLOSE")
            ofr_open = update.getValue("OFR_OPEN")
            ofr_high = update.getValue("OFR_HIGH")
            ofr_low = update.getValue("OFR_LOW")
            ofr_close = update.getValue("OFR_CLOSE")

            if utm:
                self.current_candle["UTM"] = int(utm)
            if bid_open and ofr_open:
                self.current_candle["O"] = (float(bid_open) + float(ofr_open)) / 2
            if bid_high and ofr_high:
                self.current_candle["H"] = (float(bid_high) + float(ofr_high)) / 2
            if bid_low and ofr_low:
                self.current_candle["L"] = (float(bid_low) + float(ofr_low)) / 2
            if bid_close and ofr_close:
                self.current_candle["C"] = (float(bid_close) + float(ofr_close)) / 2

            # Check if candle is complete
            if cons_end == "1" and all(k in self.current_candle for k in ["UTM", "O", "H", "L", "C"]):
                candle_time = datetime.fromtimestamp(self.current_candle["UTM"] / 1000)

                if self.last_utm is None or self.current_candle["UTM"] > self.last_utm:
                    self.last_utm = self.current_candle["UTM"]
                    self.cache_manager.on_candle_complete(self.symbol, candle_time, self.current_candle.copy())

                self.current_candle = {}

        except Exception as e:
            logger.warning(f"⚠️ {self.symbol}: Candle update error: {e}")

    def onUnsubscription(self):
        logger.info(f"📴 {self.symbol}: Streaming subscription ended")


class StreamingManager:
    """
    Manages WebSocket streaming with automatic reconnection.
    Handles subscription lifecycle and connection health monitoring.
    """

    CANDLE_FIELDS = [
        "UTM", "BID_OPEN", "BID_HIGH", "BID_LOW", "BID_CLOSE",
        "OFR_OPEN", "OFR_HIGH", "OFR_LOW", "OFR_CLOSE", "CONS_END"
    ]

    # Valid timeframes for IG streaming
    VALID_TIMEFRAMES = ["SECOND", "1MINUTE", "5MINUTE", "HOUR"]

    def __init__(self, ig_service, cache_manager, epics_map, timeframe="HOUR"):
        """
        Args:
            ig_service: Authenticated IGService instance
            cache_manager: StreamingCacheManager for candle storage
            epics_map: Dict mapping symbol -> epic
            timeframe: Candle timeframe (SECOND, 1MINUTE, 5MINUTE, HOUR)
        """
        self.ig_service = ig_service
        self.cache_manager = cache_manager
        self.epics_map = epics_map
        self.timeframe = timeframe if timeframe in self.VALID_TIMEFRAMES else "HOUR"
        self.stream_service = None
        self.subscriptions = {}
        self.listeners = {}
        self._stop_event = threading.Event()
        self._reconnect_thread = None
        self._health_thread = None
        self._connected = False
        self._last_heartbeat = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5
        self._lock = threading.Lock()

        # Link cache manager to this streaming manager
        self.cache_manager.streaming_manager = self

    def start(self):
        """Start streaming service and subscribe to all epics."""
        try:
            self._connect()
            self._subscribe_all()
            self._start_health_monitor()
            logger.info(f"📡 Streaming started for {len(self.epics_map)} symbols")
        except Exception as e:
            logger.error(f"❌ Failed to start streaming: {e}")
            self._schedule_reconnect()

    def _connect(self):
        """Establish streaming connection."""
        with self._lock:
            self.stream_service = IGStreamService(self.ig_service)
            self.stream_service.create_session()
            self._connected = True
            self._last_heartbeat = datetime.now()
            self._reconnect_attempts = 0
            logger.info("✅ Streaming connection established")

    def _subscribe_all(self):
        """Subscribe to candles for all configured epics."""
        for symbol, epic in self.epics_map.items():
            self._subscribe_symbol(symbol, epic)
            time.sleep(0.1)  # Small delay to avoid overwhelming the server

    def _subscribe_symbol(self, symbol, epic):
        """Subscribe to candle stream for a single symbol."""
        try:
            item = f"CHART:{epic}:{self.timeframe}"
            subscription = Subscription(
                mode="MERGE",
                items=[item],
                fields=self.CANDLE_FIELDS
            )

            listener = CandleListener(self.cache_manager, symbol)
            subscription.addlistener(listener)

            self.stream_service.subscribe(subscription)
            self.subscriptions[symbol] = subscription
            self.listeners[symbol] = listener

            logger.debug(f"📡 Subscribed to {symbol} ({item})")

        except Exception as e:
            logger.error(f"❌ Failed to subscribe to {symbol}: {e}")

    def _start_health_monitor(self):
        """Start background thread to monitor connection health."""
        def monitor():
            while not self._stop_event.is_set():
                time.sleep(30)
                if self._stop_event.is_set():
                    break

                now = datetime.now()
                if self._last_heartbeat and (now - self._last_heartbeat).seconds > 120:
                    logger.warning("⚠️ No streaming updates for 2 minutes, reconnecting...")
                    self._reconnect()

        self._health_thread = threading.Thread(target=monitor, daemon=True, name="StreamHealthMonitor")
        self._health_thread.start()

    def _schedule_reconnect(self):
        """Schedule a reconnection attempt with exponential backoff."""
        if self._stop_event.is_set():
            return

        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(f"❌ Max reconnect attempts ({self._max_reconnect_attempts}) reached")
            # Reset and try again after longer delay
            self._reconnect_attempts = 0
            time.sleep(60)

        self._reconnect_attempts += 1
        delay = min(self._reconnect_delay * (2 ** (self._reconnect_attempts - 1)), 300)
        logger.info(f"🔄 Reconnect attempt {self._reconnect_attempts} in {delay}s...")

        def delayed_reconnect():
            time.sleep(delay)
            if not self._stop_event.is_set():
                self._reconnect()

        thread = threading.Thread(target=delayed_reconnect, daemon=True)
        thread.start()

    def _reconnect(self):
        """Disconnect and reconnect streaming service."""
        if self._stop_event.is_set():
            return

        try:
            self._disconnect()
            time.sleep(2)

            # Re-authenticate IG session first (token might be expired)
            try:
                self.ig_service.create_session()
            except Exception as e:
                logger.warning(f"⚠️ Session refresh failed: {e}")

            self._connect()
            self._subscribe_all()
            logger.info("✅ Streaming reconnected successfully")
        except Exception as e:
            logger.error(f"❌ Reconnect failed: {e}")
            self._schedule_reconnect()

    def _disconnect(self):
        """Clean disconnect from streaming service."""
        with self._lock:
            self._connected = False
            try:
                if self.stream_service:
                    self.stream_service.unsubscribe_all()
                    self.stream_service.disconnect()
            except Exception as e:
                logger.warning(f"⚠️ Error during disconnect: {e}")
            finally:
                self.stream_service = None
                self.subscriptions = {}
                self.listeners = {}

    def update_heartbeat(self):
        """Called by listeners to indicate activity."""
        self._last_heartbeat = datetime.now()

    def stop(self):
        """Stop streaming service."""
        self._stop_event.set()
        self._disconnect()
        logger.info("📴 Streaming stopped")

    @property
    def is_connected(self):
        return self._connected


class StreamingCacheManager:
    """
    Builds and maintains OHLC cache from streaming data.
    Handles the conversion from streaming candles to DataFrame format.
    """

    def __init__(self, bot, max_candles=1000):
        """
        Args:
            bot: EliteBot instance for callbacks
            max_candles: Maximum candles to keep per symbol
        """
        self.bot = bot
        self.max_candles = max_candles
        self.streaming_manager = None
        self._lock = threading.Lock()
        self._pending_signals = []

    def on_candle_complete(self, symbol, candle_time, candle_data):
        """
        Called when a new candle is complete.
        Updates the bot's OHLC cache and triggers signal check.
        """
        with self._lock:
            try:
                logger.info(
                    f"🕯️ {symbol} candle complete: {candle_time.strftime('%Y-%m-%d %H:%M')} "
                    f"O={candle_data['O']:.5f} C={candle_data['C']:.5f}"
                )

                # Get or create cache DataFrame with proper dtypes
                if symbol not in self.bot.ohlc_cache or self.bot.ohlc_cache[symbol].empty:
                    self.bot.ohlc_cache[symbol] = pd.DataFrame(
                        columns=["O", "H", "L", "C"],
                        dtype=float
                    )

                cache_df = self.bot.ohlc_cache[symbol]

                # Append to cache (avoid duplicates)
                if candle_time not in cache_df.index:
                    # Use loc to add row directly (avoids concat warning)
                    cache_df.loc[candle_time] = [
                        candle_data["O"],
                        candle_data["H"],
                        candle_data["L"],
                        candle_data["C"]
                    ]
                    cache_df = cache_df.sort_index()

                    # Trim to max size
                    if len(cache_df) > self.max_candles:
                        cache_df = cache_df.tail(self.max_candles)

                    self.bot.ohlc_cache[symbol] = cache_df
                    self.bot.last_bar_time[symbol] = candle_time

                    # Trigger signal check in bot
                    self.bot.on_streaming_candle(symbol, candle_time, candle_data)

            except Exception as e:
                logger.error(f"❌ Cache update error for {symbol}: {e}")

    def get_cache_stats(self):
        """Return statistics about current cache state."""
        stats = {}
        for symbol, df in self.bot.ohlc_cache.items():
            if len(df) > 0:
                stats[symbol] = {
                    "count": len(df),
                    "first": df.index[0].isoformat() if len(df) > 0 else None,
                    "last": df.index[-1].isoformat() if len(df) > 0 else None,
                }
        return stats
