"""
Exit-Strategien Registry.

Dieses Modul verwaltet alle verfügbaren Exit-Strategien.
Neue Strategien registrieren sich automatisch beim Import.

Verwendung:
    from optimizer.exit_strategies import get_strategy, list_strategies

    # Strategie abrufen
    strategy_cls = get_strategy("fixed")
    strategy = strategy_cls.from_config(config)

    # Alle Strategien auflisten
    available = list_strategies()  # ["fixed", "atr_based"]
"""
from typing import Dict, Type

from .base import BaseExitStrategy, ExitConfig, GridParams

# Registry für Exit-Strategien
_REGISTRY: Dict[str, Type[BaseExitStrategy]] = {}


def register(name: str):
    """
    Decorator zum Registrieren einer Exit-Strategie.

    Verwendung:
        @register("my_strategy")
        class MyExitStrategy(BaseExitStrategy):
            ...
    """
    def decorator(cls: Type[BaseExitStrategy]):
        if name in _REGISTRY:
            raise ValueError(f"Exit strategy '{name}' is already registered")
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


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
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(
            f"Unknown exit strategy: '{name}'. Available: {available}"
        )
    return _REGISTRY[name]


def list_strategies() -> list:
    """
    Listet alle registrierten Exit-Strategien.

    Returns:
        Liste der Strategie-Namen
    """
    return list(_REGISTRY.keys())


def get_default_strategy() -> Type[BaseExitStrategy]:
    """
    Gibt die Default-Strategie zurück (fixed).

    Returns:
        FixedExitStrategy Klasse
    """
    return get_strategy("fixed")


# Auto-Import aller Strategien (registrieren sich beim Import)
from .fixed import FixedExitStrategy
from .atr_based import AtrExitStrategy

__all__ = [
    "BaseExitStrategy",
    "ExitConfig",
    "GridParams",
    "register",
    "get_strategy",
    "list_strategies",
    "get_default_strategy",
    "FixedExitStrategy",
    "AtrExitStrategy",
]
