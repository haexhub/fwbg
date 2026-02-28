"""
BaseExitStrategy - Abstrakte Basisklasse für Exit-Strategy-Plugins.
"""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import pandas as pd

from fwbg_sdk.base import BasePlugin, PluginPhase
from fwbg_sdk.contexts import AssetInfo


class BaseExitStrategy(BasePlugin, ABC):
    """
    Basisklasse für Exit-Strategy-Plugins.

    Exit-Strategien definieren wie TP/SL berechnet werden.
    Each strategy instance has fixed params — the optimizer iterates
    over instances, not over parameter grids.
    """

    phase = PluginPhase.EXIT_STRATEGIES
    name: str = "base"

    @abstractmethod
    def compute_targets(
        self,
        df: pd.DataFrame,
        ctx: "AssetInfo",
        **params
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Win/Loss Targets für Long und Short.

        Args:
            df: DataFrame mit OHLC-Daten
            ctx: AssetInfo mit Spread, Max-Bars, etc.
            **params: Strategy-spezifische Parameter (z.B. tp_mult, sl_mult)

        Returns:
            Tuple (targets_long, targets_short) - Arrays mit 1.0 für Win, 0.0 sonst
        """
        pass

    @abstractmethod
    def resolve_distances(
        self,
        df: pd.DataFrame,
        tp: float,
        sl: float,
        ctx: "AssetInfo",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet TP/SL-Distanzen in Preiseinheiten pro Bar.

        Wird vom Optimizer für die Trade-Evaluation aufgerufen.
        Jedes Plugin berechnet die Distanzen nach eigener Logik:
        - FixedExitStrategy: konstant (spread * tp) für alle Bars
        - AtrExitStrategy: dynamisch (atr[i] * tp) pro Bar

        Args:
            df: DataFrame mit OHLC-Daten (und ggf. ATR-Spalten)
            tp: TP-Wert (Spread-Multiplikator bei fixed, ATR-Multiplikator bei ATR)
            sl: SL-Wert
            ctx: AssetInfo mit spread, exit_params, etc.

        Returns:
            (tp_distances, sl_distances) — Arrays der Länge len(df) in Preiseinheiten
        """
        pass

    @abstractmethod
    def get_cache_key(self, params: dict) -> str:
        """
        Generiert eindeutigen Cache-Key für Parameter-Kombination.

        Wird verwendet um berechnete Targets zu cachen.

        Args:
            params: Parameter-Dict

        Returns:
            Eindeutiger String-Key
        """
        pass

    @classmethod
    def get_default_params(cls) -> dict:
        """
        Default-Parameter für die Exit-Strategy.

        Returns:
            Dict mit Default-Werten
        """
        return {}
