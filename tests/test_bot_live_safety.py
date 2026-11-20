"""Safety contracts of the live/paper bot, with the broker adapter mocked."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from fwbg.adapters.broker import AccountInfo, OrderResult, OrderSide, OrderStatus
from fwbg.bot import TradingBot


def _bot(adapter, tmp_path, **account) -> TradingBot:
    bot = TradingBot(
        adapter=adapter,
        assets_config={
            "EURUSD": {
                "features": [],
                "risk_per_trade": 0.005,
                "point_value": 0.0001,
                "sl_mult": 25,
                "tp_mult": 40,
            }
        },
        account_config={
            "account_id": "acct",
            "currency": "EUR",
            "min_lot_size": 0.1,
            "max_risk_percent": 0.05,
            **account,
        },
        stats_dir=str(tmp_path / "stats"),
        use_streaming=False,
    )
    bot.ohlc_cache["EURUSD"] = pd.DataFrame(
        {
            "O": np.full(50, 1.08),
            "H": np.full(50, 1.09),
            "L": np.full(50, 1.07),
            "C": np.full(50, 1.08),
        },
        index=pd.date_range("2025-01-01", periods=50, freq="h"),
    )
    return bot


def _adapter(order_result: OrderResult, balance: float = 100_000.0) -> MagicMock:
    adapter = MagicMock()
    adapter.get_account_info.return_value = AccountInfo(balance, balance)
    adapter.get_closed_trade_events.return_value = []
    adapter.get_positions.return_value = []
    adapter.get_value_per_point.return_value = 1.0
    adapter.submit_order.return_value = order_result
    return adapter


def test_unknown_order_outcome_is_not_counted_as_a_rejection(tmp_path, caplog):
    unknown = OrderResult(success=False, status=OrderStatus.PENDING, message="timeout")
    bot = _bot(_adapter(unknown), tmp_path)

    with caplog.at_level(logging.CRITICAL):
        bot._execute_signal("EURUSD", OrderSide.BUY, 0.9, bot.assets["EURUSD"])

    bot.adapter.submit_order.assert_called_once()
    assert bot._rejection_count == 0
    assert bot._risk_state.rejected_orders == 0
    assert any("UNKNOWN" in r.getMessage() for r in caplog.records)


def test_definite_rejection_is_still_counted(tmp_path):
    rejected = OrderResult(success=False, status=OrderStatus.REJECTED, message="no")
    bot = _bot(_adapter(rejected), tmp_path)

    bot._execute_signal("EURUSD", OrderSide.BUY, 0.9, bot.assets["EURUSD"])

    assert bot._rejection_count == 1
