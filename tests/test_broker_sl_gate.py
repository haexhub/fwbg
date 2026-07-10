"""Tests für den verpflichtenden Stop-Loss-Gate an der Broker-Grenze.

Der Gate lebt in BrokerAdapter.submit_order() (Basisklasse): jede Entry-Order
ohne positiven Stop-Loss wird deterministisch abgelehnt, ohne den Broker zu
kontaktieren. Exits (close_position) sind bewusst ausgenommen und übergeben
_submit_order_impl direkt keinen Stop.
"""
from typing import List, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

from fwbg.adapters.broker import (
    AccountInfo,
    BrokerAdapter,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Symbol,
)


# -----------------------------------------------------------------------------
# Spy-Adapter: schreibt jeden _submit_order_impl-Aufruf mit, kontaktiert keinen
# Broker. So lässt sich prüfen, ob der Gate den Impl überhaupt erreicht.
# -----------------------------------------------------------------------------


class _SpyBrokerAdapter(BrokerAdapter):
    adapter_type = "spy"

    def __init__(self, positions: Optional[List[Position]] = None):
        super().__init__()
        self._connected = True
        self._positions = positions or []
        self.impl_calls: List[dict] = []

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def get_historical_bars(self, symbol, timeframe=None, limit=1000, start=None, end=None):
        return pd.DataFrame()

    def _submit_order_impl(
        self,
        symbol,
        direction: OrderSide,
        size: float,
        stop_distance: float = None,
        limit_distance: float = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        self.impl_calls.append(
            {
                "symbol": symbol,
                "direction": direction,
                "size": size,
                "stop_distance": stop_distance,
                "limit_distance": limit_distance,
            }
        )
        return OrderResult(success=True, status=OrderStatus.FILLED)

    def get_positions(self) -> List[Position]:
        return list(self._positions)

    def get_account_info(self) -> AccountInfo:
        return AccountInfo(balance=10_000.0, equity=10_000.0)

    def get_broker_symbol(self, symbol) -> Optional[str]:
        return str(symbol)


# -----------------------------------------------------------------------------
# Base-class Gate
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("bad_stop", [None, 0, -5])
def test_entry_without_positive_stop_is_rejected(bad_stop):
    """Kein/nicht-positiver Stop => REJECTED, Impl/Broker wird nicht kontaktiert."""
    adapter = _SpyBrokerAdapter()

    result = adapter.submit_order(
        Symbol.EURUSD, OrderSide.BUY, size=1.0, stop_distance=bad_stop
    )

    assert result.success is False
    assert result.status == OrderStatus.REJECTED
    assert "stop-loss" in result.message.lower()
    assert adapter.impl_calls == []  # Gate hat vor dem Impl abgebrochen


def test_entry_with_valid_stop_is_forwarded_atomically():
    """Positiver Stop => genau ein Impl-Aufruf, Stop wird durchgereicht."""
    adapter = _SpyBrokerAdapter()

    result = adapter.submit_order(
        Symbol.EURUSD, OrderSide.BUY, size=1.0, stop_distance=42, limit_distance=84
    )

    assert result.success is True
    assert len(adapter.impl_calls) == 1
    call = adapter.impl_calls[0]
    assert call["stop_distance"] == 42
    assert call["limit_distance"] == 84


def test_close_position_is_exempt_from_gate():
    """Exit (Gegenorder) geht ohne Stop durch — nicht vom Gate abgelehnt."""
    pos = Position(
        symbol=Symbol.EURUSD,
        direction=OrderSide.BUY,
        size=1.0,
        entry_price=1.0,
        position_id="P1",
    )
    adapter = _SpyBrokerAdapter(positions=[pos])

    result = adapter.close_position("P1")

    assert result.success is True
    assert len(adapter.impl_calls) == 1
    assert adapter.impl_calls[0]["stop_distance"] is None
    assert adapter.impl_calls[0]["direction"] == OrderSide.SELL  # Gegenrichtung


def test_submit_order_override_is_forbidden():
    """Ein Adapter darf den Gate nicht durch Override von submit_order umgehen."""
    with pytest.raises(TypeError):

        class _Bypass(BrokerAdapter):
            adapter_type = "bypass"

            def submit_order(self, *args, **kwargs):  # umgeht den Gate -> verboten
                return OrderResult(success=True, status=OrderStatus.FILLED)


# -----------------------------------------------------------------------------
# IG-Adapter: der echte Broker wird ohne Stop nie kontaktiert
# -----------------------------------------------------------------------------


def _make_ig_adapter():
    pytest.importorskip("trading_ig", reason="trading-ig nicht installiert")
    from fwbg.adapters.broker.ig.adapter import IGBrokerAdapter

    adapter = IGBrokerAdapter(username="u", password="p", api_key="k")
    adapter._ig = MagicMock()
    adapter._last_request_time = 0
    adapter._ig.create_open_position.return_value = {"dealReference": "D1"}
    adapter._ig.fetch_deal_by_deal_reference.return_value = {
        "dealStatus": "ACCEPTED",
        "level": 1.1000,
    }
    return adapter


@pytest.mark.parametrize("bad_stop", [None, 0])
def test_ig_broker_untouched_without_stop(bad_stop):
    adapter = _make_ig_adapter()

    result = adapter.submit_order(
        Symbol.EURUSD, OrderSide.BUY, size=1.0, stop_distance=bad_stop
    )

    assert result.status == OrderStatus.REJECTED
    adapter._ig.create_open_position.assert_not_called()


def test_ig_valid_stop_forwarded_in_single_call():
    adapter = _make_ig_adapter()

    result = adapter.submit_order(
        Symbol.EURUSD, OrderSide.BUY, size=1.0, stop_distance=50, limit_distance=100
    )

    assert result.success is True
    adapter._ig.create_open_position.assert_called_once()
    _, kwargs = adapter._ig.create_open_position.call_args
    assert kwargs["stop_distance"] == 50
    assert kwargs["limit_distance"] == 100
