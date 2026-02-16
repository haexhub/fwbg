"""Base class for risk management plugins."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from fwbg_sdk.base import BasePlugin, PluginPhase


class BaseRiskManager(BasePlugin, ABC):
    """
    Base class for risk management plugins.

    Risk managers compute position sizing and risk controls
    from trade history, win rate, and risk-reward ratio.
    """

    phase = PluginPhase.RISK_MANAGEMENT
    name: str = "base"

    @abstractmethod
    def compute_risk_params(
        self,
        trades: List[float],
        win_rate: float,
        rrr: float,
        **params
    ) -> Dict[str, Any]:
        """
        Compute risk parameters from trade results.

        Returns dict with at minimum:
            - risk_per_trade: float (position size as fraction of capital)
            - trade_returns: List[float] (per-trade returns for metrics)
            - circuit_breaker: dict (pause_after_losses, pause_bars, enabled)
            - risk_adjustment: dict (original_risk, scale_factor, target_dd)
        """
        ...

    @classmethod
    def get_default_params(cls) -> dict:
        return {}
