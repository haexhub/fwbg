"""Kelly Criterion + Volatility Targeting Risk Manager.

Scales position size per trade based on realized volatility:
low RV → bigger position, high RV → smaller position.
Smooths equity curve and improves risk-adjusted returns.
"""
from typing import Dict, Any, List, Optional

import numpy as np

from fwbg.plugins import BaseRiskManager
from fwbg.core import register_risk_manager
from fwbg.simulation.trade import (
    adjust_risk_for_target_dd,
    find_optimal_circuit_breaker,
)


@register_risk_manager("vol_targeted_kelly")
class VolTargetedKellyRiskManager(BaseRiskManager):
    """Quarter-Kelly with per-trade volatility targeting."""

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
        target_vol: float = 15.0,
        min_scale: float = 0.25,
        max_scale: float = 2.0,
        rv_values: Optional[List[float]] = None,
        **params
    ) -> Dict[str, Any]:
        # Base Kelly
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

        # DD adjustment
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

        result = {
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

        # Vol targeting: per-trade position scaling
        if rv_values and len(rv_values) == len(trades):
            rv_arr = np.array(rv_values, dtype=float)
            scales = np.clip(target_vol / np.clip(rv_arr, 1e-6, None), min_scale, max_scale)

            result["trade_returns"] = [
                fk * s * rrr if t > 0 else -fk * s
                for t, s in zip(trades, scales)
            ]
            result["vol_targeting"] = {
                "target_vol": target_vol,
                "mean_scale": float(np.mean(scales)),
                "min_scale_used": float(np.min(scales)),
                "max_scale_used": float(np.max(scales)),
                "mean_fk_adjusted": float(fk * np.mean(scales)),
            }
        else:
            # No RV data: fixed sizing (same as Kelly)
            result["trade_returns"] = [fk * rrr if t > 0 else -fk for t in trades]

        return result

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "kelly_fraction": 0.25,
            "max_risk": 0.05,
            "target_max_dd": 0.30,
            "circuit_breaker_loss_range": [3, 8],
            "circuit_breaker_pause_range": [5, 30],
            "target_vol": 15.0,
            "min_scale": 0.25,
            "max_scale": 2.0,
        }


__all__ = ["VolTargetedKellyRiskManager"]
