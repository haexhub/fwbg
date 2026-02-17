"""
fwbg-broker-ig: IG Markets Broker Adapter für FWBG.

Dieses Paket kann eigenständig installiert werden:
    pip install fwbg-broker-ig

Oder als Teil von fwbg:
    pip install fwbg[ig]

Verwendung:
    from fwbg_broker_ig import IGBrokerAdapter
    from fwbg_sdk import Symbol, Timeframe

    adapter = IGBrokerAdapter(
        username="...",
        password="...",
        api_key="...",
        env="DEMO"
    )

    with adapter:
        df = adapter.get_historical_bars(Symbol.EURUSD, limit=1000)
"""
from .adapter import IGBrokerAdapter
from .mappings import (
    SYMBOL_TO_EPIC,
    SYMBOL_TO_YFINANCE,
    SYMBOL_POINT_VALUE,
    TIMEFRAME_TO_RESOLUTION,
    TIMEFRAME_TO_YF_INTERVAL,
)

__version__ = "1.0.0"

__all__ = [
    "IGBrokerAdapter",
    "SYMBOL_TO_EPIC",
    "SYMBOL_TO_YFINANCE",
    "SYMBOL_POINT_VALUE",
    "TIMEFRAME_TO_RESOLUTION",
    "TIMEFRAME_TO_YF_INTERVAL",
]
