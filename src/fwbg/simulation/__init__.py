"""
Simulation Module - Trade-Simulation und Numba-Kernfunktionen.
"""
from .numba_core import (
    _simulate_trade_numba,
    _simulate_trade_trailing_numba,
    _simulate_trade_scale_in_numba,
    compute_targets_numba,
)
from .trade import (
    simulate_pro_trade,
    compute_session_mask,
    calculate_sharpe_ratio,
    calculate_calmar_ratio,
    monte_carlo_permutation_test,
    monte_carlo_equity_simulation,
    adjust_risk_for_target_dd,
    find_optimal_circuit_breaker,
    calculate_equity_smoothness,
    pnl_to_returns,
)
from .equity import simulate_equity, simulate_equity_from_pnl, filter_correlated_assets

__all__ = [
    "_simulate_trade_numba",
    "_simulate_trade_trailing_numba",
    "_simulate_trade_scale_in_numba",
    "compute_targets_numba",
    "simulate_pro_trade",
    "compute_session_mask",
    "calculate_sharpe_ratio",
    "calculate_calmar_ratio",
    "monte_carlo_permutation_test",
    "monte_carlo_equity_simulation",
    "adjust_risk_for_target_dd",
    "find_optimal_circuit_breaker",
    "calculate_equity_smoothness",
    "pnl_to_returns",
    "simulate_equity",
    "simulate_equity_from_pnl",
    "filter_correlated_assets",
]
