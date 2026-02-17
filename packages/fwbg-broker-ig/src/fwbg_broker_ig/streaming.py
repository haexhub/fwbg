"""
IG Markets Lightstreamer Streaming.

Enthält den CandleListener für Live-Bar-Updates via Lightstreamer.
"""
from typing import Callable, TYPE_CHECKING
from datetime import datetime
import logging

from fwbg_sdk import Symbol, Timeframe
from fwbg.adapters.broker import BarData

try:
    from lightstreamer.client import SubscriptionListener
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    SubscriptionListener = object  # Fallback für Type-Checking

if TYPE_CHECKING:
    from .adapter import IGBrokerAdapter

log = logging.getLogger(__name__)


class IGCandleListener(SubscriptionListener):
    """Lightstreamer Listener für Candle Updates."""

    def __init__(
        self,
        adapter: "IGBrokerAdapter",
        symbol: Symbol,
        callback: Callable = None
    ):
        self.adapter = adapter
        self.symbol = symbol
        self.callback = callback
        self.current_candle = {}
        self.last_utm = None

    def onSubscription(self):
        log.info(f"{self.symbol}: Streaming subscription active")

    def onSubscriptionError(self, code, message):
        log.error(f"{self.symbol}: Subscription error {code}: {message}")

    def onItemUpdate(self, update):
        """Verarbeitet Candle Updates."""
        try:
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

            # Check if candle complete
            if cons_end == "1" and all(k in self.current_candle for k in ["UTM", "O", "H", "L", "C"]):
                candle_time = datetime.fromtimestamp(self.current_candle["UTM"] / 1000)

                if self.last_utm is None or self.current_candle["UTM"] > self.last_utm:
                    self.last_utm = self.current_candle["UTM"]

                    bar = BarData(
                        symbol=self.symbol,
                        timeframe=Timeframe.H1,
                        timestamp=candle_time,
                        open=self.current_candle["O"],
                        high=self.current_candle["H"],
                        low=self.current_candle["L"],
                        close=self.current_candle["C"],
                    )

                    # Notify all callbacks
                    self.adapter._notify_bar_callbacks(bar)

                self.current_candle = {}

        except Exception as e:
            log.warning(f"{self.symbol}: Candle update error: {e}")

    def onUnsubscription(self):
        log.info(f"{self.symbol}: Streaming subscription ended")
