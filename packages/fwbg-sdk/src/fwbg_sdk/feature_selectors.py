"""
BaseFeatureSelector - Abstrakte Basisklasse für Feature-Selection-Plugins.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import pandas as pd

from fwbg_sdk.base import BasePlugin, PluginPhase


class BaseFeatureSelector(BasePlugin, ABC):
    """
    Basisklasse für Feature-Selection-Plugins.

    Feature-Selectors wählen die wichtigsten Features für das ML-Modell aus.
    """

    phase = PluginPhase.FEATURE_SELECTION
    name: str = "base"

    @abstractmethod
    def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        max_features: int = None,
        **params
    ) -> Tuple[List[str], dict]:
        """
        Wählt die wichtigsten Features aus.

        Args:
            X: Feature-DataFrame
            y: Target-Array (0/1 für Loss/Win)
            max_features: Maximale Anzahl Features (None = unbegrenzt)
            **params: Methoden-spezifische Parameter

        Returns:
            Tuple (selected_features, metadata)
            - selected_features: Liste der ausgewählten Feature-Namen
            - metadata: Dict mit zusätzlichen Infos (z.B. Feature-Importance)
        """
        pass

    @classmethod
    def get_default_params(cls) -> dict:
        """
        Default-Parameter für die Feature-Selection.

        Returns:
            Dict mit Default-Werten
        """
        return {}
