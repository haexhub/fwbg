"""
Basis-Klassen für Exit-Strategien.

Exit-Strategien definieren, wie TP/SL für Trades berechnet werden.
Jede Strategie muss von BaseExitStrategy erben und sich registrieren.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Iterator
import numpy as np
import pandas as pd


@dataclass
class ExitConfig:
    """Basis-Konfiguration für Exit-Strategien."""
    pass


@dataclass
class GridParams:
    """Parameter für eine einzelne Grid-Kombination."""
    tp_value: float  # TP-Wert (Pips bei fixed, Multiplikator bei ATR)
    sl_value: float  # SL-Wert (Pips bei fixed, Multiplikator bei ATR)
    timeout_bars: int = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def rrr(self) -> float:
        """Risk-Reward-Ratio."""
        return self.tp_value / self.sl_value if self.sl_value > 0 else 0


class BaseExitStrategy(ABC):
    """
    Abstrakte Basisklasse für Exit-Strategien.

    Jede Exit-Strategie muss implementieren:
    - compute_targets(): Berechnet Win/Loss Targets für Training
    - iterate_grid(): Gibt Grid-Parameter-Kombinationen zurück
    - get_cache_key(): Eindeutiger Key für Target-Caching
    """

    name: str = "base"

    @abstractmethod
    def compute_targets(
        self,
        df: pd.DataFrame,
        ctx: "SimulationContext",
        params: GridParams,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Win/Loss Targets für Long und Short.

        Args:
            df: DataFrame mit OHLC-Daten und Indikatoren
            ctx: SimulationContext mit Asset-spezifischen Parametern
            params: Grid-Parameter für diese Berechnung

        Returns:
            (targets_long, targets_short) - Arrays mit 1.0 für Win, 0.0 sonst
        """
        pass

    @abstractmethod
    def iterate_grid(
        self,
        grid_config: Dict[str, Any],
        ctx: "SimulationContext",
    ) -> Iterator[GridParams]:
        """
        Iteriert über alle Grid-Parameter-Kombinationen.

        Args:
            grid_config: Grid-Konfiguration aus Strategy-Config
            ctx: SimulationContext

        Yields:
            GridParams für jede Kombination
        """
        pass

    @abstractmethod
    def get_cache_key(self, params: GridParams) -> str:
        """
        Gibt eindeutigen Cache-Key für diese Parameter-Kombination zurück.

        Wird verwendet, um berechnete Targets zu cachen und wiederzuverwenden.

        Args:
            params: Grid-Parameter

        Returns:
            Eindeutiger String-Key
        """
        pass

    def total_combinations(
        self,
        grid_config: Dict[str, Any],
        ctx: "SimulationContext",
    ) -> int:
        """
        Berechnet Gesamtzahl der Grid-Kombinationen.

        Default-Implementierung zählt die iterate_grid() Ergebnisse.
        Kann für Performance überschrieben werden.
        """
        return sum(1 for _ in self.iterate_grid(grid_config, ctx))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseExitStrategy":
        """
        Factory-Methode zum Erstellen aus Config-Dictionary.

        Args:
            config: Konfiguration aus Strategy-JSON

        Returns:
            Konfigurierte Exit-Strategie-Instanz
        """
        return cls()
