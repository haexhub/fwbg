"""
BasePreprocessor - Abstrakte Basisklasse für Preprocessing-Plugins.
"""
from abc import ABC, abstractmethod
import pandas as pd


class BasePreprocessor(ABC):
    """
    Basisklasse für Preprocessing-Plugins.

    Preprocessors transformieren Daten vor der Feature-Berechnung.

    Beispiel:
        ```python
        from fwbg.plugins import BasePreprocessor

        class FractionalDiffPreprocessor(BasePreprocessor):
            order = 10  # Früh ausführen

            def transform(self, df, auto_d=True, default_d=0.4):
                # Fractional Differentiation...
                return df_transformed
        ```
    """

    # Plugin-Name (wird vom Registry gesetzt)
    name: str = "base"

    # Ausführungsreihenfolge (niedriger = früher)
    order: int = 100

    @abstractmethod
    def transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        Transformiert den DataFrame.

        Args:
            df: Input DataFrame mit OHLC-Daten
            **params: Preprocessor-spezifische Parameter

        Returns:
            Transformierter DataFrame
        """
        pass

    def inverse_transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        Rücktransformation (optional).

        Nicht alle Preprocessors unterstützen Rücktransformation.
        Default: Gibt den DataFrame unverändert zurück.

        Args:
            df: Transformierter DataFrame

        Returns:
            Original-DataFrame (falls möglich)
        """
        return df

    @classmethod
    def get_default_params(cls) -> dict:
        """
        Default-Parameter für den Preprocessor.

        Returns:
            Dict mit Default-Werten
        """
        return {}
