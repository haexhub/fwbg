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


# ---------------------------------------------------------------------------
# Session expiry: every call re-authenticates, including the kill switch
# ---------------------------------------------------------------------------

_EURUSD_POSITION = {
    "dealId": "D1",
    "epic": "CS.D.EURUSD.CFD.IP",
    "direction": "BUY",
    "size": 1.0,
    "openLevel": 1.1,
    "level": 1.11,
}


def _expired_adapter() -> tuple[IGBrokerAdapter, MagicMock]:
    """An adapter whose current service rejects every call as token-invalid."""
    adapter = _adapter()
    adapter.reconnect_delay_seconds = 0
    expired = adapter._ig
    for name in ("fetch_open_positions", "close_open_position", "fetch_accounts"):
        getattr(expired, name).side_effect = TokenInvalidException()
    return adapter, expired


def test_kill_reauthenticates_and_closes_positions_after_token_expiry(tmp_path):
    from fwbg.bot import TradingBot

    adapter, expired = _expired_adapter()
    fresh = create_autospec(IGService, instance=True)
    fresh.fetch_open_positions.return_value = _positions_frame([_EURUSD_POSITION])
    fresh.close_open_position.return_value = {
        "dealStatus": "ACCEPTED",
        "dealReference": "C1",
        "level": 1.1,
    }
    bot = TradingBot(
        adapter=adapter,
        assets_config={},
        account_config={"account_id": "t", "currency": "EUR"},
        stats_dir=str(tmp_path),
        use_streaming=False,
    )

    with patch(IG_CLASS, return_value=fresh) as service_class:
        assert bot.kill(close_positions=True) is True

    service_class.assert_called_once_with("u", "p", "k", "DEMO")
    fresh.create_session.assert_called_once()
    fresh.close_open_position.assert_called_once()
    assert fresh.close_open_position.call_args.kwargs["deal_id"] == "D1"
    expired.close_open_position.assert_not_called()


def test_get_positions_and_account_info_reauthenticate():
    adapter, _ = _expired_adapter()
    fresh = create_autospec(IGService, instance=True)
    fresh.fetch_open_positions.return_value = _positions_frame([_EURUSD_POSITION])
    fresh.fetch_accounts.return_value = pd.DataFrame(
        [{"balance": 1000.0, "profitLoss": 5.0, "deposit": 10.0, "available": 990.0}]
    )

    with patch(IG_CLASS, return_value=fresh):
        positions = adapter.get_positions()
        account = adapter.get_account_info()

    assert [p.position_id for p in positions] == ["D1"]
    assert account.balance == 1000.0
    assert account.equity == 1005.0


def test_failed_reauthentication_is_reported_not_swallowed():
    adapter, _ = _expired_adapter()
    with patch(IG_CLASS, side_effect=ConnectionError("down")):
        with pytest.raises(BrokerUnavailableError):
            adapter.get_positions()


def test_non_auth_errors_are_not_retried():
    adapter = _adapter()
    adapter._ig.fetch_open_positions.side_effect = TimeoutError("read timed out")

    with patch(IG_CLASS) as service_class:
        with pytest.raises(BrokerUnavailableError):
            adapter.get_positions()

    service_class.assert_not_called()
    assert adapter._ig.fetch_open_positions.call_count == 1


def test_reauthentication_is_attempted_only_once_per_call():
    adapter, _ = _expired_adapter()
    still_expired = create_autospec(IGService, instance=True)
    still_expired.fetch_open_positions.side_effect = TokenInvalidException()

    with patch(IG_CLASS, return_value=still_expired) as service_class:
        with pytest.raises(BrokerUnavailableError):
            adapter.get_positions()

    service_class.assert_called_once()
    assert still_expired.fetch_open_positions.call_count == 1
