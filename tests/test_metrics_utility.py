"""Tests for the consolidated trade-metrics utility module."""
import math

import pytest

from fwbg.utils.metrics import (
    compute_drawdown_from_equity,
    pnl_to_returns,
    profit_factor,
    simulate_equity_from_binary_returns,
    simulate_equity_from_returns,
    win_rate,
)


def test_compute_drawdown_simple_curve():
    equity = [100.0, 110.0, 105.0, 120.0, 90.0, 100.0]
    assert compute_drawdown_from_equity(equity) == pytest.approx(30 / 120)


def test_compute_drawdown_monotone_up():
    assert compute_drawdown_from_equity([100.0, 110.0, 120.0]) == 0.0


def test_compute_drawdown_empty():
    assert compute_drawdown_from_equity([]) == 0.0


def test_simulate_equity_from_binary_returns():
    eq = simulate_equity_from_binary_returns([1, -1, 1], risk_per_trade=0.1, rrr=2.0)
    assert eq[0] == 100.0
    assert eq[1] == pytest.approx(120.0)
    assert eq[2] == pytest.approx(120.0 * 0.9)
    assert eq[3] == pytest.approx(120.0 * 0.9 * 1.2)


def test_simulate_equity_from_returns_compounds():
    eq = simulate_equity_from_returns([0.1, -0.05, 0.2])
    assert eq[-1] == pytest.approx(100.0 * 1.1 * 0.95 * 1.2)


def test_pnl_to_returns_scales_avg_loss_to_minus_fk():
    pnls = [10.0, -5.0, 20.0, -10.0]
    fk = 0.02
    returns = pnl_to_returns(pnls, fk)
    losses = [r for r in returns if r < 0]
    assert sum(losses) / len(losses) == pytest.approx(-fk)


def test_pnl_to_returns_empty():
    assert pnl_to_returns([], 0.02) == []


def test_win_rate_basic():
    trades = [{"result": 1.0}, {"result": -1.0}, {"result": 1.0}, {"result": 0.0}]
    assert win_rate(trades) == pytest.approx(2 / 4)


def test_win_rate_empty():
    assert win_rate([]) == 0.0


def test_win_rate_custom_key():
    trades = [{"pnl": 1.0}, {"pnl": -1.0}]
    assert win_rate(trades, key="pnl") == pytest.approx(0.5)


def test_profit_factor():
    assert profit_factor([10.0, -5.0, 20.0, -10.0]) == pytest.approx(2.0)


def test_profit_factor_no_losses():
    assert profit_factor([10.0, 20.0]) == math.inf


def test_profit_factor_empty():
    assert profit_factor([]) == 0.0
