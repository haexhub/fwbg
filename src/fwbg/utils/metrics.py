"""Consolidated trade-metrics utility.

Single source of truth for drawdown, equity simulation, win-rate, and
profit-factor computations. Multiple call sites previously inlined these
loops, risking bug-by-divergence when one was fixed but the others weren't.

All functions accept the data shapes already used elsewhere in the codebase
and return numerical results without rounding so callers can format as they
need.
"""
from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

import numpy as np


def compute_drawdown_from_equity(equity: Sequence[float]) -> float:
    """Return max drawdown of an equity curve as a positive fraction.

    Drawdown is computed relative to the running peak. Returns 0.0 if the
    curve is empty or never falls below its starting peak.
    """
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        if peak > 0:
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


def simulate_equity_from_binary_returns(
    returns: Iterable,
    risk_per_trade: float,
    rrr: float,
    starting_equity: float = 100.0,
) -> List[float]:
    """Reconstruct an equity curve from a sequence of binary trade outcomes.

    For each item in ``returns``: positive → win at ``risk_per_trade * rrr``,
    non-positive → loss at ``risk_per_trade``. Returns the equity curve
    including the starting equity at index 0.
    """
    equity = [starting_equity]
    for r in returns:
        if r > 0:
            equity.append(equity[-1] * (1 + risk_per_trade * rrr))
        else:
            equity.append(equity[-1] * (1 - risk_per_trade))
    return equity


def simulate_equity_from_returns(
    trade_returns: Iterable[float],
    starting_equity: float = 100.0,
) -> List[float]:
    """Reconstruct equity curve by compounding per-trade returns."""
    equity = [starting_equity]
    for r in trade_returns:
        equity.append(equity[-1] * (1 + r))
    return equity


def pnl_to_returns(pnl_raw: Sequence[float], fk: float) -> List[float]:
    """Convert raw PnL values into Kelly-scaled per-trade returns.

    Scales so the average loss-return equals exactly ``-fk``. Winners reflect
    the realised reward-risk ratio.

    Args:
        pnl_raw: Raw PnL values (positive = win, negative = loss).
        fk: Kelly fraction / risk per trade (e.g. 0.02 = 2%).

    Returns:
        Per-trade returns as fractions of capital. Empty input → empty list.
    """
    if not pnl_raw:
        return []
    losses = [abs(p) for p in pnl_raw if p < 0]
    scale = float(np.mean(losses)) if losses else float(np.mean(np.abs(pnl_raw))) or 1.0
    return [fk * p / scale for p in pnl_raw]


def win_rate(trades: Sequence[Mapping], *, key: str = "result") -> float:
    """Fraction of trades with ``trade[key] > 0``. Empty input → 0.0."""
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get(key, 0) > 0)
    return wins / len(trades)


def profit_factor(pnls: Iterable[float]) -> float:
    """Gross profit / gross loss. Inf if profits exist with zero losses."""
    gross_win = 0.0
    gross_loss = 0.0
    for p in pnls:
        if p > 0:
            gross_win += p
        elif p < 0:
            gross_loss += -p
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss
