"""
M6a Task 3 — per-strategy paper/live-trade telemetry writer tests.

Behaviour:
- When ``TradingBot.strategy_slug`` is set, the bot writes three files under
  ``<paper_data_dir>/account-trades/<strategy_slug>/``:
    * ``trades.jsonl`` — append-only, one JSON line per successful trade entry
      (event-driven in ``_execute_signal``).
    * ``status.json`` — overwritten in ``_write_status``; per-strategy
      equity snapshot.
    * ``positions.json`` — overwritten in ``_write_status``; open positions
      from ``adapter.get_positions()``.
- When ``strategy_slug`` is ``None`` (legacy mode), NOTHING is written under
  ``data/account-trades/`` — the existing dashboard `_write_status` path is
  untouched.
- Telemetry failures are best-effort: logged at WARNING level, never raised.

All tests are offline (mocked adapter, ``tmp_path`` for the writer base dir).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

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
)
from fwbg.bot import AssetConfig, TradingBot


# -----------------------------------------------------------------------------
# Test double for BrokerAdapter (offline, no network)
# -----------------------------------------------------------------------------


class _StubBrokerAdapter(BrokerAdapter):
    """Minimal in-memory broker adapter for offline telemetry tests."""

    adapter_type = "stub"

    def __init__(
        self,
        *,
        positions: Optional[List[Position]] = None,
        account: Optional[AccountInfo] = None,
        order_result: Optional[OrderResult] = None,
    ):
        super().__init__()
        self._connected = True
        self._positions = positions or []
        self._account = account or AccountInfo(balance=10_000.0, equity=10_000.0)
        self._order_result = order_result or OrderResult(
            success=True, status=OrderStatus.FILLED, fill_price=1.0823, filled_quantity=1000.0
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def get_historical_bars(self, symbol, timeframe=None, limit=1000, start=None, end=None):
        # Provide enough OHLC rows so _execute_signal's ATR computation can run.
        idx = pd.date_range("2024-01-01", periods=50, freq="h")
        return pd.DataFrame(
            {
                "O": [1.08] * 50,
                "H": [1.09] * 50,
                "L": [1.07] * 50,
                "C": [1.08] * 50,
            },
            index=idx,
        )

    def _submit_order_impl(
        self,
        symbol,
        direction: OrderSide,
        size: float,
        stop_distance: float = None,
        limit_distance: float = None,
        order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        return self._order_result

    def get_positions(self) -> List[Position]:
        return list(self._positions)

    def get_account_info(self) -> AccountInfo:
        return self._account

    def get_broker_symbol(self, symbol) -> Optional[str]:
        return str(symbol)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_assets_config(strategy_slug: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    cfg: Dict[str, Any] = {
        "features": ["rsi_14"],
        "conf_thresh": 0.6,
        "risk_per_trade": 0.005,
        "point_value": 0.0001,
        "sl_mult": 25.0,
        "tp_mult": 40.0,
        "ensemble": {},
    }
    if strategy_slug is not None:
        cfg["strategy_slug"] = strategy_slug
    return {"EURUSD": cfg}


def _make_account_config() -> Dict[str, Any]:
    return {
        "account_id": "test_account",
        "currency": "EUR",
        "min_lot_size": 0.1,
        "max_risk_percent": 0.05,
    }


def _make_bot(
    *,
    tmp_path,
    strategy_slug: Optional[str] = None,
    adapter: Optional[_StubBrokerAdapter] = None,
) -> TradingBot:
    """Build a TradingBot wired for telemetry tests, using tmp_path as paper-data dir."""
    adapter = adapter or _StubBrokerAdapter()
    bot = TradingBot(
        adapter=adapter,
        assets_config=_make_assets_config(strategy_slug),
        account_config=_make_account_config(),
        stats_dir=str(tmp_path / "stats_export"),
        paper_data_dir=str(tmp_path),
        use_streaming=False,
    )
    return bot


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_execute_signal_with_strategy_slug_appends_trades_jsonl(tmp_path):
    """After a successful trade, trades.jsonl gains one valid JSON line."""
    bot = _make_bot(tmp_path=tmp_path, strategy_slug="foo")
    # Seed OHLC cache so ATR computation succeeds.
    bot.ohlc_cache["EURUSD"] = bot.adapter.get_historical_bars("EURUSD")

    cfg = bot.assets["EURUSD"]
    bot._execute_signal("EURUSD", OrderSide.BUY, probability=0.7, cfg=cfg)

    trades_file = tmp_path / "account-trades" / "foo" / "trades.jsonl"
    assert trades_file.exists(), "trades.jsonl must be created on successful trade"

    lines = trades_file.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["strategy_slug"] == "foo"
    assert entry["symbol"] == "EURUSD"
    assert entry["side"] == "buy"
    assert entry["entry_price"] == pytest.approx(1.0823)
    assert "entry_time" in entry
    assert "trade_id" in entry


def test_execute_signal_records_signal_price_and_assumed_spread(tmp_path, monkeypatch):
    """trades.jsonl entry carries signal_price (last close) and assumed_spread.

    Plan 016 — paper fidelity telemetry.
    """
    from fwbg.data.assets import AssetConfig as DataAssetConfig

    def _fake_get_asset(symbol):
        return DataAssetConfig(
            symbol=symbol,
            asset_class="FOREX",
            point=0.0001,
            spread=0.00042,
            currencies=["EUR", "USD"],
        )

    monkeypatch.setattr("fwbg.bot.get_asset", _fake_get_asset)

    bot = _make_bot(tmp_path=tmp_path, strategy_slug="foo")
    bot.ohlc_cache["EURUSD"] = bot.adapter.get_historical_bars("EURUSD")

    cfg = bot.assets["EURUSD"]
    bot._execute_signal("EURUSD", OrderSide.BUY, probability=0.7, cfg=cfg)

    trades_file = tmp_path / "account-trades" / "foo" / "trades.jsonl"
    entry = json.loads(trades_file.read_text().strip().splitlines()[0])
    # Stub OHLC has a constant close of 1.08 (see _StubBrokerAdapter.get_historical_bars).
    assert entry["signal_price"] == pytest.approx(1.08)
    assert entry["assumed_spread"] == pytest.approx(0.00042)


def test_execute_signal_spread_lookup_failure_defaults_to_none(tmp_path, monkeypatch):
    """A failing spread lookup must not raise; assumed_spread falls back to None."""

    def _boom(symbol):
        raise RuntimeError("no asset meta")

    monkeypatch.setattr("fwbg.bot.get_asset", _boom)

    bot = _make_bot(tmp_path=tmp_path, strategy_slug="foo")
    bot.ohlc_cache["EURUSD"] = bot.adapter.get_historical_bars("EURUSD")

    cfg = bot.assets["EURUSD"]
    bot._execute_signal("EURUSD", OrderSide.BUY, probability=0.7, cfg=cfg)  # must not raise

    trades_file = tmp_path / "account-trades" / "foo" / "trades.jsonl"
    entry = json.loads(trades_file.read_text().strip().splitlines()[0])
    assert entry["assumed_spread"] is None
    assert entry["signal_price"] == pytest.approx(1.08)


def test_execute_signal_without_strategy_slug_writes_nothing(tmp_path):
    """Legacy mode (slug=None) writes nothing under account-trades/."""
    bot = _make_bot(tmp_path=tmp_path, strategy_slug=None)
    bot.ohlc_cache["EURUSD"] = bot.adapter.get_historical_bars("EURUSD")

    cfg = bot.assets["EURUSD"]
    bot._execute_signal("EURUSD", OrderSide.BUY, probability=0.7, cfg=cfg)

    assert not (tmp_path / "account-trades").exists(), (
        "Legacy mode must not create account-trades/"
    )


def test_write_status_writes_status_and_positions_json(tmp_path):
    """_write_status with slug writes both status.json and positions.json."""
    positions = [
        Position(
            symbol="EURUSD",
            direction=OrderSide.BUY,
            size=1000.0,
            entry_price=1.0823,
            current_price=1.0851,
            stop_loss=1.0790,
            take_profit=1.0900,
        )
    ]
    adapter = _StubBrokerAdapter(positions=positions)
    bot = _make_bot(tmp_path=tmp_path, strategy_slug="foo", adapter=adapter)

    bot._write_status("RUNNING")

    status_file = tmp_path / "account-trades" / "foo" / "status.json"
    positions_file = tmp_path / "account-trades" / "foo" / "positions.json"
    assert status_file.exists()
    assert positions_file.exists()

    status_data = json.loads(status_file.read_text())
    assert status_data["strategy_slug"] == "foo"
    assert "current_equity" in status_data
    assert "starting_equity" in status_data
    assert "equity_curve_sample" in status_data

    positions_data = json.loads(positions_file.read_text())
    assert positions_data["strategy_slug"] == "foo"
    assert isinstance(positions_data["positions"], list)
    assert len(positions_data["positions"]) == 1


def test_status_json_equity_curve_bounded_to_200(tmp_path):
    """Equity curve sample is bounded to <=200, keeps first and last."""
    bot = _make_bot(tmp_path=tmp_path, strategy_slug="foo")

    # Pump 500 equity observations into the curve.
    base_ts = pd.Timestamp("2024-01-01T00:00:00Z")
    for i in range(500):
        bot._equity_curve.append(
            {"t": (base_ts + pd.Timedelta(minutes=i)).isoformat(), "equity": 10_000.0 + i}
        )

    first_equity = bot._equity_curve[0]["equity"]
    last_equity = bot._equity_curve[-1]["equity"]

    bot._write_status("RUNNING")

    status_data = json.loads((tmp_path / "account-trades" / "foo" / "status.json").read_text())
    sample = status_data["equity_curve_sample"]
    assert len(sample) <= 200
    # First + last preserved (under the FIRST/LAST guarantee).
    assert sample[0]["equity"] == first_equity
    # Note: _write_status itself may have appended a new last observation
    # via adapter.get_account_info().equity. Either the original last is in
    # the sample, or the new last (from _write_status's own append) is —
    # but the sample's last must reflect a real data point.
    assert sample[-1]["equity"] >= last_equity or sample[-1]["equity"] == bot._equity_curve[-1]["equity"]


def test_positions_json_includes_sl_tp_from_adapter(tmp_path):
    """positions.json carries stop_loss / take_profit / current_price from Position."""
    positions = [
        Position(
            symbol="EURUSD",
            direction=OrderSide.BUY,
            size=1.0,
            entry_price=1.08,
            stop_loss=1.07,
            take_profit=1.10,
            current_price=1.085,
        )
    ]
    adapter = _StubBrokerAdapter(positions=positions)
    bot = _make_bot(tmp_path=tmp_path, strategy_slug="foo", adapter=adapter)

    bot._write_status("RUNNING")

    positions_data = json.loads(
        (tmp_path / "account-trades" / "foo" / "positions.json").read_text()
    )
    pos = positions_data["positions"][0]
    assert pos["symbol"] == "EURUSD"
    assert pos["side"] == "buy"
    assert pos["quantity"] == pytest.approx(1.0)
    assert pos["entry_price"] == pytest.approx(1.08)
    assert pos["current_price"] == pytest.approx(1.085)
    assert pos["stop_loss"] == pytest.approx(1.07)
    assert pos["take_profit"] == pytest.approx(1.10)


def test_write_failures_are_logged_not_raised(tmp_path, caplog):
    """Telemetry write failures are logged at WARNING; bot must not raise."""
    bot = _make_bot(tmp_path=tmp_path, strategy_slug="foo")

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    caplog.set_level(logging.WARNING)
    # Patch the per-strategy writer's underlying write_text on Path.
    with patch.object(Path, "write_text", _boom):
        # Must not raise.
        bot._write_status("RUNNING")

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("telemetry" in m.lower() or "write" in m.lower() for m in warning_messages), (
        f"Expected a WARNING about the failed write. Got: {warning_messages}"
    )


def test_position_dataclass_accepts_optional_sl_tp_current_price():
    """Position(...) accepts the M6a optional fields and round-trips them."""
    p = Position(
        symbol="EURUSD",
        direction=OrderSide.BUY,
        size=1.0,
        entry_price=1.08,
        stop_loss=1.07,
        take_profit=1.10,
        current_price=1.085,
    )
    assert p.stop_loss == 1.07
    assert p.take_profit == 1.10
    assert p.current_price == 1.085


def test_broker_adapter_is_paper_property_exists():
    """is_paper property exists on adapters and returns a bool."""
    adapter = _StubBrokerAdapter()
    # Stub adapter inherits the safe default (True).
    assert isinstance(adapter.is_paper, bool)
    assert adapter.is_paper is True
