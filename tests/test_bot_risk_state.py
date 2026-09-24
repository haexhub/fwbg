"""Contract tests for realized-PnL circuit-breaker state and bot gating."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from fwbg.adapters.broker import AccountInfo, OrderSide
from fwbg.bot import TradingBot
from fwbg.core.risk_state import ClosedTradeEvent, RiskState


UTC = timezone.utc


def _event(event_id, pnl, hour=12, day=1):
    return ClosedTradeEvent(
        event_id=event_id,
        closed_at=datetime(2025, 1, day, hour, tzinfo=UTC),
        net_pnl=pnl,
    )


def _known_state(**kwargs):
    state = RiskState(
        max_daily_loss_percent=0.05,
        pause_after_consecutive_losses=3,
        **kwargs,
    )
    now = datetime(2025, 1, 1, 12, tzinfo=UTC)
    assert state.observe(1000.0, [], observed_at=now, now=now)
    return state, now


def test_daily_loss_threshold_uses_confirmed_realized_pnl():
    state, now = _known_state()
    state.process_close_event(_event("loss-1", -51))

    assert state.daily_pnl == pytest.approx(-51)
    assert state.can_trade(now) is False


def test_rejection_count_does_not_create_a_loss():
    state, now = _known_state()
    state.count_rejection()
    state.count_rejection()

    assert state.rejected_orders == 2
    assert state.daily_pnl == 0
    assert state.consecutive_losses == 0
    assert state.can_trade(now) is True


def test_confirmed_win_resets_loss_streak():
    state, _ = _known_state()
    state.process_close_event(_event("loss-1", -1))
    state.process_close_event(_event("loss-2", -1))
    state.process_close_event(_event("win-1", 2))

    assert state.consecutive_losses == 0
    assert state.daily_pnl == pytest.approx(0)


def test_duplicate_close_event_is_idempotent():
    state, _ = _known_state()
    event = _event("deal-1", -10)

    assert state.process_close_event(event) is True
    assert state.process_close_event(event) is False
    assert state.daily_pnl == pytest.approx(-10)


def test_day_rollover_resets_daily_metrics_but_keeps_replay_ids():
    state, now = _known_state()
    event = _event("deal-1", -10)
    state.process_close_event(event)
    next_day = now + timedelta(days=1)

    assert state.observe(990.0, [], observed_at=next_day, now=next_day)
    assert state.daily_pnl == 0
    assert state.consecutive_losses == 0
    assert state.process_close_event(event) is False


def test_unknown_and_stale_state_block_trading():
    state = RiskState(max_observation_age_seconds=30)
    now = datetime(2025, 1, 1, 12, tzinfo=UTC)
    assert state.can_trade(now) is False
    assert state.observe(1000.0, None, observed_at=now, now=now) is False
    assert state.can_trade(now) is False
    assert state.observe(
        1000.0,
        [],
        observed_at=now - timedelta(seconds=31),
        now=now,
    ) is False


def test_restart_replay_preserves_idempotent_state():
    state, now = _known_state()
    state.process_close_event(_event("deal-1", -10))
    restored = RiskState.from_snapshot(state.snapshot())

    assert restored.process_close_event(_event("deal-1", -10)) is False
    assert restored.daily_pnl == pytest.approx(-10)
    assert restored.consecutive_losses == 1


def _make_bot(adapter, *, risk_per_trade=0.005, min_lot_size=0.1):
    bot = TradingBot(
        adapter=adapter,
        assets_config={
            "EURUSD": {
                "features": [],
                "risk_per_trade": risk_per_trade,
                "point_value": 0.0001,
                "sl_mult": 25,
                "tp_mult": 40,
            }
        },
        account_config={
            "currency": "EUR",
            "min_lot_size": min_lot_size,
            "max_risk_percent": 0.05,
        },
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


def test_bot_unknown_broker_state_blocks_new_order():
    adapter = MagicMock()
    adapter.get_account_info.return_value = AccountInfo(1000, 1000)
    adapter.get_closed_trade_events.return_value = None
    adapter.get_positions.return_value = []
    bot = _make_bot(adapter)

    bot._execute_signal("EURUSD", OrderSide.BUY, 0.9, bot.assets["EURUSD"])

    adapter.submit_order.assert_not_called()


def test_minimum_lot_is_not_allowed_to_exceed_risk_budget():
    adapter = MagicMock()
    adapter.get_account_info.return_value = AccountInfo(10, 10)
    adapter.get_closed_trade_events.return_value = []
    adapter.get_positions.return_value = []
    bot = _make_bot(adapter, risk_per_trade=0.001, min_lot_size=0.1)

    bot._execute_signal("EURUSD", OrderSide.BUY, 0.9, bot.assets["EURUSD"])

    adapter.submit_order.assert_not_called()
