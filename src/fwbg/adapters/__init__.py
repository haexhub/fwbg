"""
Adapter System - Modulare Integration von Datenquellen und Brokern.

Adapters sind die Schnittstelle zwischen FWBG und der Außenwelt:
- DataAdapter: Liefern Marktdaten (CSV, REST API, WebSocket)
- ExecutionAdapter: Führen Orders aus (IG, Binance, etc.)

Alle Adapter kommunizieren über den MessageBus via Events.

Beispiel - Eigenen DataAdapter schreiben:

    from fwbg.adapters import DataAdapter
    from fwbg.core.events import BarEvent

    class MyDataAdapter(DataAdapter):
        def connect(self):
            # Verbindung herstellen
            pass

        def subscribe_bars(self, symbol, timeframe):
            # Bars abonnieren, Events via self.publish() senden
            pass
"""
from .base import BaseAdapter
from .data import DataAdapter, CSVDataAdapter
from .execution import (
    ExecutionAdapter,
    Order, Position, AccountInfo,
    OrderType, OrderSide,
    IGExecutionAdapter,
)

__all__ = [
    # Base
    "BaseAdapter",
    # Data Adapters
    "DataAdapter",
    "CSVDataAdapter",
    # Execution Adapters
    "ExecutionAdapter",
    "Order",
    "Position",
    "AccountInfo",
    "OrderType",
    "OrderSide",
    "IGExecutionAdapter",
]
