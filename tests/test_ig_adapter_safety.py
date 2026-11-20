"""Safety contracts of the IG adapter, with the IG HTTP boundary mocked.

No test here contacts IG or any network: ``IGService`` is replaced by an
autospec mock (or patched at the class level for re-authentication).
"""

import logging
from datetime import datetime
from unittest.mock import MagicMock, create_autospec, patch

import pandas as pd
import pytest

pytest.importorskip("trading_ig")
from trading_ig import IGService
from trading_ig.rest import IGException, TokenInvalidException

from fwbg.adapters.broker import BrokerUnavailableError, OrderSide, OrderStatus
from fwbg.adapters.broker.ig.adapter import IGBrokerAdapter
from fwbg_sdk import Symbol, Timeframe

IG_CLASS = "fwbg.adapters.broker.ig.adapter.IGService"


def _adapter(**kwargs) -> IGBrokerAdapter:
    kwargs.setdefault("rate_limit_delay", 0.0)
    adapter = IGBrokerAdapter(username="u", password="p", api_key="k", **kwargs)
    adapter._ig = create_autospec(IGService, instance=True)
    adapter._connected = True
    adapter._last_request_time = 0
    return adapter


def _positions_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["dealId", "epic", "direction", "size", "openLevel", "level"],
    )


# ---------------------------------------------------------------------------
# yfinance fallback is explicit opt-in
# ---------------------------------------------------------------------------


def test_yfinance_fallback_is_off_by_default():
    assert IGBrokerAdapter("u", "p", "k").use_yfinance_fallback is False


def test_default_adapter_never_mixes_in_yfinance_data():
    adapter = _adapter()
    adapter._ig.fetch_historical_prices_by_epic.return_value = {"prices": []}

    with (
        patch("fwbg.adapters.broker.ig.adapter.YFINANCE_AVAILABLE", True),
        patch("fwbg.adapters.broker.ig.adapter.yf") as yf_mock,
    ):
        df = adapter.get_historical_bars(Symbol.EURUSD, Timeframe.H1)

    yf_mock.download.assert_not_called()
    assert df.empty


def test_opt_in_yfinance_fallback_logs_a_warning(caplog):
    adapter = _adapter(use_yfinance_fallback=True)
    adapter._ig.fetch_historical_prices_by_epic.return_value = {"prices": []}
    index = pd.DatetimeIndex([datetime(2024, 1, 15, 14), datetime(2024, 1, 15, 15)])
    frame = pd.DataFrame(
        {
            "Open": [1.1, 1.2],
            "High": [1.2, 1.3],
            "Low": [1.0, 1.1],
            "Close": [1.15, 1.25],
        },
        index=index,
    )

    with (
        caplog.at_level(logging.WARNING),
        patch("fwbg.adapters.broker.ig.adapter.YFINANCE_AVAILABLE", True),
        patch("fwbg.adapters.broker.ig.adapter.yf") as yf_mock,
    ):
        yf_mock.download.return_value = frame
        df = adapter.get_historical_bars(Symbol.EURUSD, Timeframe.H1)

    assert len(df) == 2
    assert any("falling back to yfinance" in r.getMessage() for r in caplog.records)
