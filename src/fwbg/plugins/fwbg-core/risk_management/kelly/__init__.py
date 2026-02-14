"""Kelly Criterion Risk Manager Plugin."""
from typing import Dict, Any, List

from fwbg.plugins import BaseRiskManager
from fwbg.core import register_risk_manager
from fwbg.simulation.trade import (
    adjust_risk_for_target_dd,
    find_optimal_circuit_breaker,
)


@register_risk_manager("kelly")
class KellyRiskManager(BaseRiskManager):
    """Quarter-Kelly with DD adjustment and circuit breaker."""

    def compute_risk_params(
        self,
        trades: List[float],
        win_rate: float,
        rrr: float,
        *,
        kelly_fraction: float = 0.25,
        max_risk: float = 0.05,
        target_max_dd: float = 0.30,
        circuit_breaker_loss_range: tuple = (3, 8),
        circuit_breaker_pause_range: tuple = (5, 30),
        **params
    ) -> Dict[str, Any]:
        # Full Kelly formula
        full_kelly = (win_rate * rrr - (1 - win_rate)) / rrr if rrr > 0 else 0
        fk = max(0, min(max_risk, full_kelly * kelly_fraction))

        if fk <= 0:
            return {
                "risk_per_trade": 0,
                "is_profitable": False,
                "full_kelly": full_kelly,
                "circuit_breaker": {
                    "pause_after_losses": 0, "pause_bars": 0, "enabled": False,
                },
                "risk_adjustment": {
                    "original_risk": 0, "scale_factor": 1.0, "target_dd": target_max_dd,
                },
            }

        # Drawdown adjustment
        kelly_adj = adjust_risk_for_target_dd(
            trades, fk, rrr, target_max_dd=target_max_dd
        )
        if kelly_adj["scale_factor"] < 1.0:
            fk = kelly_adj["adjusted_risk"]

        # Circuit breaker
        cb = find_optimal_circuit_breaker(
            trades, fk, rrr,
            loss_range=circuit_breaker_loss_range,
            pause_range=circuit_breaker_pause_range,
        )

        return {
            "risk_per_trade": fk,
            "is_profitable": True,
            "full_kelly": full_kelly,
            "circuit_breaker": {
                "pause_after_losses": cb["optimal_pause_after_losses"],
                "pause_bars": cb["optimal_pause_bars"],
                "enabled": cb["optimal_pause_after_losses"] > 0,
            },
            "risk_adjustment": {
                "original_risk": kelly_adj["adjusted_risk"] / kelly_adj["scale_factor"]
                    if kelly_adj["scale_factor"] > 0 else fk,
                "scale_factor": kelly_adj["scale_factor"],
                "target_dd": target_max_dd,
            },
        }

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "kelly_fraction": 0.25,
            "max_risk": 0.05,
            "target_max_dd": 0.30,
            "circuit_breaker_loss_range": [3, 8],
            "circuit_breaker_pause_range": [5, 30],
        }


__all__ = ["KellyRiskManager"]
