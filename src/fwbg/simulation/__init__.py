"""
Simulation Module - Trade-Simulation und Numba-Kernfunktionen.
"""
from .numba_core import (
    _simulate_trade_numba,
    compute_targets_numba,
)
from .trade import (
    simulate_pro_trade,
    calculate_sharpe_ratio,
    calculate_calmar_ratio,
    monte_carlo_permutation_test,
    monte_carlo_equity_simulation,
    adjust_kelly_for_target_dd,
    find_optimal_circuit_breaker,
    calculate_equity_smoothness,
)
from .equity import simulate_equity, filter_correlated_assets

__all__ = [
    "_simulate_trade_numba",
    "compute_targets_numba",
    "simulate_pro_trade",
    "calculate_sharpe_ratio",
    "calculate_calmar_ratio",
    "monte_carlo_permutation_test",
    "monte_carlo_equity_simulation",
    "adjust_kelly_for_target_dd",
    "find_optimal_circuit_breaker",
    "calculate_equity_smoothness",
    "simulate_equity",
    "filter_correlated_assets",
]
