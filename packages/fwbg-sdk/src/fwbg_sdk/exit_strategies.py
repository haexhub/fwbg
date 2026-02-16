"""
BaseExitStrategy - Abstrakte Basisklasse für Exit-Strategy-Plugins.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Iterator
import numpy as np
import pandas as pd

from fwbg_sdk.base import BasePlugin, PluginPhase
from fwbg_sdk.contexts import AssetInfo


class BaseExitStrategy(BasePlugin, ABC):
    """
    Basisklasse für Exit-Strategy-Plugins.

    Exit-Strategien definieren wie TP/SL berechnet werden und
    wie über Parameter-Kombinationen iteriert wird.
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
    def iterate_grid(
        self,
        grid_config: dict,
        ctx: "AssetInfo"
    ) -> Iterator[dict]:
        """
        Iteriert über alle Parameter-Kombinationen aus der Grid-Config.

        Args:
            grid_config: Grid-Konfiguration aus Strategy-JSON
            ctx: AssetInfo

        Yields:
            Dict mit Parameter-Kombination für compute_targets
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
