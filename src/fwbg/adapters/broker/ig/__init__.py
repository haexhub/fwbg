"""
IG Markets Broker Adapter.

Beispiel:
    from fwbg.adapters.broker.ig import IGBrokerAdapter
    from fwbg_sdk import Symbol

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

__all__ = ["IGBrokerAdapter"]
