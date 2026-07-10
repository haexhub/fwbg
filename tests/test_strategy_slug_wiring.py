"""
M6a — strategy_slug wiring tests.

Behaviour:
- AssetConfig has an optional `strategy_slug: str | None = None`.
- TradingBot exposes `bot.strategy_slug` set from the asset configs.
- run_bot_for_account accepts `strategy_slug` (kwarg + --strategy-slug CLI flag)
  and propagates it down to the bot.
- Legacy mode (no slug) keeps working unchanged.

Task 3 (separate session) will USE `bot.strategy_slug` to gate telemetry writes
under data/account-trades/<strategy_slug>/. M6a only wires the field through.
"""
from __future__ import annotations

import json
import os
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
    """Minimal in-memory broker adapter for offline tests."""

    adapter_type = "stub"

    def __init__(self):
        super().__init__()
        self._connected = False

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
        return OrderResult(success=True, status=OrderStatus.FILLED)

    def get_positions(self) -> List[Position]:
        return []

    def get_account_info(self) -> AccountInfo:
        return AccountInfo(balance=10_000.0, equity=10_000.0)

    def get_broker_symbol(self, symbol) -> Optional[str]:
        return str(symbol)


def _make_assets_config() -> Dict[str, Dict[str, Any]]:
    return {
        "EURUSD": {
            "features": ["rsi_14", "ema_50"],
            "conf_thresh": 0.6,
            "ensemble": {},
        }
    }


def _make_account_config() -> Dict[str, Any]:
    return {
        "account_id": "test_account",
        "currency": "EUR",
        "min_lot_size": 0.1,
        "max_risk_percent": 0.05,
    }


# -----------------------------------------------------------------------------
# 1+2 — AssetConfig
# -----------------------------------------------------------------------------


def test_asset_config_strategy_slug_defaults_none():
    cfg = AssetConfig(symbol="EURUSD", features=["rsi_14"])
    assert cfg.strategy_slug is None


def test_asset_config_strategy_slug_can_be_set():
    cfg = AssetConfig(symbol="EURUSD", features=["rsi_14"], strategy_slug="foo")
    assert cfg.strategy_slug == "foo"


# -----------------------------------------------------------------------------
# 3 — TradingBot exposes strategy_slug
# -----------------------------------------------------------------------------


def test_trading_bot_exposes_strategy_slug_from_config():
    adapter = _StubBrokerAdapter()
    assets_config = _make_assets_config()
    assets_config["EURUSD"]["strategy_slug"] = "orb__forex__001"

    bot = TradingBot(
        adapter=adapter,
        assets_config=assets_config,
        account_config=_make_account_config(),
    )

    assert bot.strategy_slug == "orb__forex__001"


# -----------------------------------------------------------------------------
# 4+5 — run_bot_for_account wires --strategy-slug into AssetConfig
# -----------------------------------------------------------------------------


def _write_account_dir(tmp_path) -> str:
    """Write minimal account_info.json + assets.json under tmp_path."""
    account_dir = tmp_path / "stub_account"
    account_dir.mkdir()
    (account_dir / "account_info.json").write_text(
        json.dumps(
            {
                "credentials": {
                    "username": "u",
                    "password": "p",
                    "api_key": "k",
                    "env": "DEMO",
                },
                "metadata": {"currency": "EUR"},
                "money_management": {"min_lot_size": 0.1, "max_risk_percent": 0.05},
            }
        )
    )
    (account_dir / "assets.json").write_text(
        json.dumps(_make_assets_config())
    )
    return str(account_dir)


def test_run_bot_for_account_accepts_strategy_slug_kwarg(tmp_path):
    """Passing strategy_slug propagates into the TradingBot."""
    from fwbg import __main__ as bot_main

    account_dir = _write_account_dir(tmp_path)

    captured = {}

    def _capture_init(self, adapter, assets_config, account_config, **kwargs):
        # Stash assets_config so we can assert slug propagation
        captured["assets_config"] = assets_config
        captured["account_config"] = account_config
        # Construct a minimal viable bot for caller; avoid network
        self.adapter = adapter
        self.assets = {
            s: AssetConfig.from_dict(s, cfg) for s, cfg in assets_config.items()
        }
        first_slug = next(iter(self.assets.values())).strategy_slug
        self.strategy_slug = first_slug

    def _no_run(self):
        return None

    with patch.object(bot_main, "create_adapter", return_value=_StubBrokerAdapter()), \
         patch.object(TradingBot, "__init__", _capture_init), \
         patch.object(TradingBot, "run", _no_run), \
         patch.object(TradingBot, "stop", lambda self: None):
        bot_main.run_bot_for_account(
            broker="ig",
            account_path=account_dir,
            use_streaming=False,
            strategy_slug="orb__forex__001",
        )

    assert "assets_config" in captured
    for symbol, cfg in captured["assets_config"].items():
        assert cfg.get("strategy_slug") == "orb__forex__001", (
            f"strategy_slug not propagated to asset {symbol}"
        )


def test_run_bot_for_account_legacy_mode_when_slug_none(tmp_path):
    """No strategy_slug → assets keep their original config (no slug injected)."""
    from fwbg import __main__ as bot_main

    account_dir = _write_account_dir(tmp_path)

    captured = {}

    def _capture_init(self, adapter, assets_config, account_config, **kwargs):
        captured["assets_config"] = assets_config
        self.adapter = adapter
        self.assets = {
            s: AssetConfig.from_dict(s, cfg) for s, cfg in assets_config.items()
        }
        first_slug = next(iter(self.assets.values())).strategy_slug
        self.strategy_slug = first_slug

    def _no_run(self):
        return None

    with patch.object(bot_main, "create_adapter", return_value=_StubBrokerAdapter()), \
         patch.object(TradingBot, "__init__", _capture_init), \
         patch.object(TradingBot, "run", _no_run), \
         patch.object(TradingBot, "stop", lambda self: None):
        bot_main.run_bot_for_account(
            broker="ig",
            account_path=account_dir,
            use_streaming=False,
        )

    assert "assets_config" in captured
    for symbol, cfg in captured["assets_config"].items():
        assert cfg.get("strategy_slug") is None, (
            f"strategy_slug should be None in legacy mode for {symbol}"
        )
