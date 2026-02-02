"""
Exit Strategy Plugins.

Verfügbare Strategien:
- fixed: Fixe TP/SL-Werte basierend auf Spread-Multiplikatoren
- atr_based: Dynamische TP/SL-Werte basierend auf ATR-Multiplikatoren
"""
from typing import Type

from fwbg.core import get_exit_strategy as _get_exit_strategy, list_exit_strategies
from fwbg.plugins import BaseExitStrategy

from .base import ExitConfig, GridParams
from .fixed import FixedExitStrategy
from .atr_based import AtrExitStrategy


def get_strategy(name: str) -> Type[BaseExitStrategy]:
    """
    Gibt Exit-Strategie-Klasse für den gegebenen Namen zurück.

    Args:
        name: Name der Strategie (z.B. "fixed", "atr_based")

    Returns:
        Die Strategie-Klasse

    Raises:
        ValueError: Wenn Strategie nicht gefunden
    """
    return _get_exit_strategy(name)


def get_default_strategy() -> Type[BaseExitStrategy]:
    """
    Gibt die Default-Strategie zurück (atr_based).

    Returns:
        AtrExitStrategy Klasse
    """
    return get_strategy("atr_based")


__all__ = [
    "FixedExitStrategy",
    "AtrExitStrategy",
    "BaseExitStrategy",
    "ExitConfig",
    "GridParams",
    "get_strategy",
    "get_default_strategy",
    "list_exit_strategies",
]
