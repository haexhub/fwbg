"""
Adapter System - Modulare Integration von Brokern.

BrokerAdapter ist das Interface für alle Broker-Integrationen:
- Historische Marktdaten abrufen
- Live-Streaming (optional)
- Order-Ausführung
- Position-Management
- Account-Information

Beispiel:

    from fwbg.adapters import IGBrokerAdapter, OrderSide

    adapter = IGBrokerAdapter(
        username="...",
        password="...",
        api_key="...",
        env="DEMO"
    )

    with adapter:
        df = adapter.get_historical_bars("EURUSD", limit=1000)
        result = adapter.submit_order("EURUSD", OrderSide.BUY, size=0.1)
"""
from .base import BaseAdapter
from .broker import (
    BrokerAdapter,
    IGBrokerAdapter,
    OrderSide,
    OrderType,
    OrderStatus,
    OrderResult,
    Position,
    AccountInfo,
    BarData,
)

__all__ = [
    "BaseAdapter",
    "BrokerAdapter",
    "IGBrokerAdapter",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    "OrderResult",
    "Position",
    "AccountInfo",
    "BarData",
]
