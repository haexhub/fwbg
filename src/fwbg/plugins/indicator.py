"""
BaseIndicator - Abstrakte Basisklasse für Indicator-Plugins.
"""
from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class BaseIndicator(ABC):
    """
    Basisklasse für Indicator-Plugins.

    Indicators berechnen technische Features basierend auf OHLCV-Daten.

    Beispiel:
        ```python
        from fwbg.plugins import BaseIndicator

        class RSIIndicator(BaseIndicator):
            group = "momentum"

            def compute(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
                # RSI berechnen...
                df['rsi_value'] = ...
                return df

            def get_feature_columns(self) -> List[str]:
                return ['rsi_value']
        ```
    """

    # Plugin-Name (wird vom Registry gesetzt)
    name: str = "base"

    # Feature-Gruppe für Kategorisierung
    group: str = "custom"

    @abstractmethod
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        Berechnet Indicator-Spalten und fügt sie zum DataFrame hinzu.

        Args:
            df: DataFrame mit OHLC-Daten (Spalten: O, H, L, C, V)
            **params: Indicator-spezifische Parameter

        Returns:
            DataFrame mit neuen Feature-Spalten
        """
        pass

    @abstractmethod
    def get_feature_columns(self) -> List[str]:
        """
        Gibt Liste der berechneten Feature-Spalten zurück.

        Returns:
            Liste der Spaltennamen die vom Indicator erzeugt werden
        """
        pass

    @classmethod
    def get_default_params(cls) -> dict:
        """
        Default-Parameter für den Indicator.

        Returns:
            Dict mit Default-Werten für alle Parameter
        """
        return {}
