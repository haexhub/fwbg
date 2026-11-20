"""Safety contracts of the live/paper bot, with the broker adapter mocked."""

import json
import logging
from datetime import datetime, timedelta
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


# ---------------------------------------------------------------------------
# A signal already acted on must not fire again after a restart
# ---------------------------------------------------------------------------

HOUR = datetime(2025, 3, 3, 12)  # naive UTC, like the bar index


class _BuyModel:
    def predict_probability(self, _features):
        return np.array([[0.05, 0.95]])


def _armed_bot(tmp_path, monkeypatch, hour=HOUR):
    import fwbg.bot as bot_module

    monkeypatch.setattr(bot_module, "_current_utc_hour", lambda: hour)
    bot = _bot(_adapter(FILLED), tmp_path)
    bot.models["EURUSD"] = _BuyModel()
    bot.features_cache["EURUSD"] = pd.DataFrame(
        {"C": np.full(120, 1.08)},
        index=pd.date_range("2025-01-01", periods=120, freq="h"),
    )
    return bot


def test_signal_hour_survives_a_restart(tmp_path, monkeypatch):
    first = _armed_bot(tmp_path, monkeypatch)
    first._check_signal("EURUSD")
    first.adapter.submit_order.assert_called_once()

    restarted = _armed_bot(tmp_path, monkeypatch)  # same stats_dir, fresh process
    assert restarted.last_signal_hour == {"EURUSD": HOUR}
    restarted._check_signal("EURUSD")
    restarted.adapter.submit_order.assert_not_called()

    next_hour = _armed_bot(tmp_path, monkeypatch, hour=HOUR + timedelta(hours=1))
    next_hour._check_signal("EURUSD")
    next_hour.adapter.submit_order.assert_called_once()


def test_signal_hour_is_written_before_the_order_is_sent(tmp_path, monkeypatch):
    bot = _armed_bot(tmp_path, monkeypatch)
    seen = {}

    def submit(**_kwargs):
        seen["state"] = json.loads(bot._state_path().read_text())
        return FILLED

    bot.adapter.submit_order.side_effect = submit
    bot._check_signal("EURUSD")

    assert seen["state"] == {"last_signal_hour": {"EURUSD": HOUR.isoformat()}}


def test_unreadable_state_blocks_signals_for_the_current_hour(tmp_path, monkeypatch):
    state = tmp_path / "stats" / "acct" / "bot_state.json"
    state.parent.mkdir(parents=True)
    state.write_text("{not json")

    corrupt = _armed_bot(tmp_path, monkeypatch)
    corrupt._check_signal("EURUSD")
    corrupt.adapter.submit_order.assert_not_called()

    import fwbg.bot as bot_module

    monkeypatch.setattr(
        bot_module, "_current_utc_hour", lambda: HOUR + timedelta(hours=1)
    )
    corrupt._check_signal("EURUSD")  # same process, one hour later
    corrupt.adapter.submit_order.assert_called_once()


def test_signal_is_not_executed_when_state_cannot_be_persisted(tmp_path, monkeypatch):
    bot = _armed_bot(tmp_path, monkeypatch)
    (tmp_path / "stats").mkdir()
    (tmp_path / "stats" / "acct").write_text("a file where the directory should be")

    bot._check_signal("EURUSD")

    bot.adapter.submit_order.assert_not_called()
    assert "EURUSD" not in bot.last_signal_hour


# ---------------------------------------------------------------------------
# Training labels
# ---------------------------------------------------------------------------


def test_last_bar_is_not_trained_as_a_down_sample(tmp_path, monkeypatch):
    import fwbg.bot as bot_module

    hour = datetime(2025, 1, 3, 12)
    monkeypatch.setattr(bot_module, "_current_utc_hour", lambda: hour)
    closes = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]  # strictly rising
    adapter = MagicMock()
    adapter.get_historical_bars.return_value = pd.DataFrame(
        {"O": closes, "H": closes, "L": closes, "C": closes},
        index=pd.date_range(hour - timedelta(hours=9), periods=9, freq="h"),
    )
    seen = {}

    class Model:
        def train(self, features, target, _context, **_kwargs):
            seen["index"] = features.index
            seen["target"] = list(target)

    monkeypatch.setattr(bot_module, "get_model", lambda _name: Model)
    bot = _bot(adapter, tmp_path)
    bot.min_training_samples = 5
    bot._compute_features = lambda df: df

    assert bot._train_model("EURUSD", bot.assets["EURUSD"]) is True
    # 9 completed bars, the last has no next close: 8 labelled samples, all "up".
    assert seen["target"] == [1] * 8
    assert seen["index"].max() == hour - timedelta(hours=2)
