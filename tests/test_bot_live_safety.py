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


# ---------------------------------------------------------------------------
# Position sizing uses the account-currency value of a point
# ---------------------------------------------------------------------------

FILLED = OrderResult(success=True, status=OrderStatus.FILLED, fill_price=1.08)


def _sized(tmp_path, value_per_point):
    adapter = _adapter(FILLED)
    adapter.get_value_per_point.return_value = value_per_point
    bot = _bot(adapter, tmp_path)
    bot._execute_signal("EURUSD", OrderSide.BUY, 0.9, bot.assets["EURUSD"])
    return adapter


def test_size_is_risk_over_stop_distance_times_value_per_point(tmp_path):
    # Risk budget: 0.5% of 100 000 = 500. ATR 0.02 x 25 / 0.0001 = 5000 points.
    # 0.1 per point per lot -> 500 / (5000 x 0.1) = 1.0 lot (the old
    # risk / stop_points formula would have sent 0.1).
    adapter = _sized(tmp_path, 0.1)

    kwargs = adapter.submit_order.call_args.kwargs
    assert kwargs["stop_distance"] in (4999, 5000)
    assert kwargs["size"] == 1.0
    adapter.get_value_per_point.assert_called_once_with("EURUSD", 0.0001)


def test_size_is_rounded_down_to_stay_inside_the_risk_budget(tmp_path):
    # 500 / (5000 x 0.84) = 0.119 lots -> 0.11, not 0.12.
    adapter = _sized(tmp_path, 0.84)

    assert adapter.submit_order.call_args.kwargs["size"] == 0.11


def test_trade_is_refused_when_one_lot_already_exceeds_the_risk_budget(tmp_path):
    # 500 / (5000 x 10) = 0.01 lots < minimum lot 0.1: the old formula traded 0.1
    # lots, ten times the intended risk.
    adapter = _sized(tmp_path, 10.0)

    adapter.submit_order.assert_not_called()


def test_trade_is_refused_without_instrument_metadata(tmp_path):
    for unavailable in (None, 0.0, -1.0, float("nan"), "1.0", MagicMock()):
        adapter = _sized(tmp_path, unavailable)
        adapter.submit_order.assert_not_called()


def test_trade_is_refused_for_a_non_positive_point_value(tmp_path):
    adapter = _adapter(FILLED)
    bot = _bot(adapter, tmp_path)
    bot.assets["EURUSD"].point_value = 0.0

    bot._execute_signal("EURUSD", OrderSide.BUY, 0.9, bot.assets["EURUSD"])

    adapter.submit_order.assert_not_called()
