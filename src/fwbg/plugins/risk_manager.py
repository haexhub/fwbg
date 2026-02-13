"""Base class for risk management plugins."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseRiskManager(ABC):
    """
    Base class for risk management plugins.

    Risk managers compute position sizing and risk controls
    from trade history, win rate, and risk-reward ratio.
    """

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
            - kelly_risk: float (position size as fraction of capital)
            - circuit_breaker: dict (pause_after_losses, pause_bars, enabled)
            - kelly_adjustment: dict (original_kelly, scale_factor, target_dd)
        """
        ...

    @classmethod
    def get_default_params(cls) -> dict:
        return {}
