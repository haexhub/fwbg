"""Contract tests for both public IG adapter import paths."""

from unittest.mock import create_autospec

import pandas as pd
import pytest

trading_ig = pytest.importorskip("trading_ig")
from trading_ig import IGService

from fwbg.adapters.broker import BrokerUnavailableError, OrderSide, OrderStatus
from fwbg.adapters.broker.ig.adapter import IGBrokerAdapter as CoreIGBrokerAdapter
from fwbg_sdk import Symbol


def _adapter():
    adapter = CoreIGBrokerAdapter(username="u", password="p", api_key="k")
    adapter._ig = create_autospec(IGService, instance=True)
    adapter._last_request_time = 0
    return adapter


def test_package_import_path_is_the_core_implementation():
    from fwbg_broker_ig import IGBrokerAdapter as PackageIGBrokerAdapter

    assert PackageIGBrokerAdapter is CoreIGBrokerAdapter


def test_create_open_position_receives_trading_ig_0024_contract():
    adapter = _adapter()
    adapter._ig.create_open_position.return_value = {
        "dealStatus": "ACCEPTED",
        "dealId": "D1",
        "level": 1.1,
    }

    result = adapter.submit_order(
        Symbol.EURUSD, OrderSide.BUY, size=1.0, stop_distance=50, limit_distance=100
    )

    assert result.success is True
    assert result.status is OrderStatus.FILLED
    kwargs = adapter._ig.create_open_position.call_args.kwargs
    assert kwargs == {
        "currency_code": "EUR",
        "direction": "BUY",
        "epic": "CS.D.EURUSD.CFD.IP",
        "expiry": "DFB",
        "force_open": True,
        "level": None,
        "order_type": "MARKET",
        "size": 1.0,
        "guaranteed_stop": False,
        "stop_distance": 50,
        "limit_distance": 100,
        "limit_level": None,
        "quote_id": None,
        "stop_level": None,
        "trailing_stop": False,
        "trailing_stop_increment": None,
    }


def test_unconfirmed_create_response_is_rejected():
    adapter = _adapter()
    adapter._ig.create_open_position.return_value = {"dealReference": "D1"}
    adapter._ig.fetch_deal_by_deal_reference.return_value = None

    result = adapter.submit_order(
        Symbol.EURUSD, OrderSide.BUY, size=1.0, stop_distance=50
    )

    assert result.success is False
    assert result.status is OrderStatus.REJECTED
    assert result.status is not OrderStatus.PENDING


def test_close_uses_deal_id_and_sdk_close_endpoint():
    adapter = _adapter()
    adapter._ig.fetch_open_positions.return_value = pd.DataFrame(
        {
            "dealId": ["D1"],
            "epic": ["CS.D.EURUSD.CFD.IP"],
            "direction": ["BUY"],
            "size": [1.0],
            "openLevel": [1.1],
            "level": [1.11],
        }
    )
    adapter._ig.close_open_position.return_value = {
        "dealStatus": "ACCEPTED",
        "dealReference": "C1",
        "level": 1.1,
    }

    result = adapter.close_position("D1")

    assert result.success is True
    assert result.status is OrderStatus.FILLED
    adapter._ig.close_open_position.assert_called_once_with(
        deal_id="D1",
        direction="SELL",
        epic="CS.D.EURUSD.CFD.IP",
        expiry="DFB",
        level=None,
        order_type="MARKET",
        quote_id=None,
        size=1.0,
    )


def test_empty_positions_is_distinct_from_positions_timeout():
    adapter = _adapter()
    adapter._ig.fetch_open_positions.return_value = pd.DataFrame()
    assert adapter.get_positions() == []

    adapter._ig.fetch_open_positions.side_effect = TimeoutError("timed out")
    with pytest.raises(BrokerUnavailableError, match="positions query failed"):
        adapter.get_positions()
