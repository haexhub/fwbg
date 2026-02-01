"""
Exit Strategy Plugins.

Verfügbare Strategien:
- fixed: Fixe TP/SL-Werte basierend auf Spread-Multiplikatoren
- atr_based: Dynamische TP/SL-Werte basierend auf ATR-Multiplikatoren
"""
from .fixed import FixedExitStrategy
from .atr_based import AtrExitStrategy

__all__ = ["FixedExitStrategy", "AtrExitStrategy"]
