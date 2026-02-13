"""
Basis-Klassen für Exit-Strategien.

GridParams kapselt alle Parameter für eine einzelne Grid-Kombination
und wird an compute_targets übergeben.
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ExitConfig:
    """Basis-Konfiguration für Exit-Strategien."""
    pass


@dataclass
class GridParams:
    """
    Parameter für eine einzelne Grid-Kombination.

    Wird von nested_cv an compute_targets übergeben.
    """
    tp_value: float  # TP-Wert (Pips bei fixed, Multiplikator bei ATR)
    sl_value: float  # SL-Wert (Pips bei fixed, Multiplikator bei ATR)
    timeout_bars: int = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def rrr(self) -> float:
        """Risk-Reward-Ratio."""
        return self.tp_value / self.sl_value if self.sl_value > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert GridParams zu dict für neue Plugin-API."""
        result = {
            "tp_mult": self.tp_value,
            "sl_mult": self.sl_value,
        }
        if self.timeout_bars is not None:
            result["timeout_bars"] = self.timeout_bars
        if self.extra:
            result.update(self.extra)
        return result


__all__ = ["ExitConfig", "GridParams"]
