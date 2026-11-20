"""Deprecated alias of the ``kelly`` risk manager.

The volatility scaling this plugin used to compute (target_vol / realized_vol
per trade) was never applied to any sized return, drawdown or reported metric,
so the plugin sized exactly like ``kelly``.  The scaling and its outputs
(``vol_targeting``, scaled ``trade_returns``) were removed; the name stays
registered so existing strategy configs keep working.  Use ``kelly``.
"""

import warnings
from typing import Dict, Any, List

from fwbg_sdk import BaseRiskManager, register_risk_manager
from fwbg.simulation.trade import (
    adjust_risk_for_target_dd,
    find_optimal_circuit_breaker,
)


@register_risk_manager("vol_targeted_kelly")
class VolTargetedKellyRiskManager(BaseRiskManager):
    """Deprecated: identical to ``kelly`` (volatility scaling was removed)."""

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
        **params,
    ) -> Dict[str, Any]:
        legacy = sorted({"target_vol", "min_scale", "max_scale"} & set(params))
        if legacy:
            warnings.warn(
                f"vol_targeted_kelly no longer scales by volatility; ignoring "
                f"{', '.join(legacy)}. Use risk_management 'kelly'.",
                DeprecationWarning,
                stacklevel=2,
            )
        # Base Kelly
        full_kelly = (win_rate * rrr - (1 - win_rate)) / rrr if rrr > 0 else 0
        fk = max(0, min(max_risk, full_kelly * kelly_fraction))

        if fk <= 0:
            return {
                "risk_per_trade": 0,
                "is_profitable": False,
                "full_kelly": full_kelly,
                "trade_returns": [0.0] * len(trades),
                "circuit_breaker": {
                    "pause_after_losses": 0,
                    "pause_bars": 0,
                    "enabled": False,
                },
                "risk_adjustment": {
                    "original_risk": 0,
                    "scale_factor": 1.0,
                    "target_dd": target_max_dd,
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
            trades,
            fk,
            rrr,
            loss_range=circuit_breaker_loss_range,
            pause_range=circuit_breaker_pause_range,
        )

        result = {
            "risk_per_trade": fk,
            "is_profitable": True,
            "full_kelly": full_kelly,
            "trade_returns": [fk * rrr if t > 0 else -fk for t in trades],
            "circuit_breaker": {
                "pause_after_losses": cb["optimal_pause_after_losses"],
                "pause_bars": cb["optimal_pause_bars"],
                "enabled": cb["optimal_pause_after_losses"] > 0,
            },
            "risk_adjustment": {
                "original_risk": kelly_adj["adjusted_risk"] / kelly_adj["scale_factor"]
                if kelly_adj["scale_factor"] > 0
                else fk,
                "scale_factor": kelly_adj["scale_factor"],
                "target_dd": target_max_dd,
            },
        }

        return result

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "kelly_fraction": 0.25,
            "max_risk": 0.05,
            "target_max_dd": 0.30,
            "circuit_breaker_loss_range": [3, 8],
            "circuit_breaker_pause_range": [5, 30],
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "kelly_fraction": {
                "type": "float",
                "default": 0.25,
                "description": "Fraction of full Kelly to use (0.25 = quarter-Kelly for conservative sizing)",
                "min": 0.01,
                "max": 1.0,
                "step": 0.05,
            },
            "max_risk": {
                "type": "float",
                "default": 0.05,
                "description": "Maximum risk per trade as fraction of account (hard cap regardless of Kelly)",
                "min": 0.001,
                "max": 1.0,
                "step": 0.005,
            },
            "target_max_dd": {
                "type": "float",
                "default": 0.30,
                "description": "Target maximum drawdown; risk is scaled down if simulated DD exceeds this",
                "min": 0.01,
                "max": 1.0,
                "step": 0.05,
            },
            "circuit_breaker_loss_range": {
                "type": "list[int]",
                "default": [3, 8],
                "description": "Range [min, max] of consecutive losses to search for optimal circuit breaker trigger",
            },
            "circuit_breaker_pause_range": {
                "type": "list[int]",
                "default": [5, 30],
                "description": "Range [min, max] of pause bars to search for optimal circuit breaker pause duration",
            },
        }


__all__ = ["VolTargetedKellyRiskManager"]
