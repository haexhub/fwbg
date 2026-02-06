"""
BaseIndicator - Abstrakte Basisklasse für Indicator-Plugins.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Union
import numpy as np
import pandas as pd


# Epsilon für Division-durch-Null Vermeidung (einheitlich für alle Indikatoren)
EPSILON = 1e-10


def safe_divide(
    numerator: Union[pd.Series, np.ndarray],
    denominator: Union[pd.Series, np.ndarray],
) -> Union[pd.Series, np.ndarray]:
    """
    Sichere Division die NaN zurückgibt bei Division durch Null.

    Einheitliche Methode für alle Indikatoren um Division-durch-Null
    konsistent zu behandeln.

    Args:
        numerator: Zähler
        denominator: Nenner

    Returns:
        Ergebnis der Division, NaN wo Nenner ~0
    """
    if isinstance(denominator, pd.Series):
        denom_safe = denominator.replace(0, np.nan)
        # Auch sehr kleine Werte als Null behandeln
        denom_safe = denom_safe.where(denom_safe.abs() > EPSILON, np.nan)
        return numerator / denom_safe
    else:
        # numpy array
        denom_safe = np.where(np.abs(denominator) > EPSILON, denominator, np.nan)
        return numerator / denom_safe


def shift_features(
    features: Dict[str, Union[pd.Series, np.ndarray]],
    index: pd.Index,
) -> pd.DataFrame:
    """
    Shiftet alle Features um 1 Bar um Lookahead Bias zu vermeiden.

    KRITISCH: Bei Bar i soll das Modell nur Features von Bar i-1 sehen.
    Diese Funktion stellt sicher, dass alle Features konsistent
    um 1 Bar verschoben werden.

    Args:
        features: Dict mit Feature-Namen und Werten
        index: Index für den resultierenden DataFrame

    Returns:
        DataFrame mit allen Features, um 1 Bar geshiptet
    """
    features_df = pd.DataFrame(features, index=index)
    for col in features_df.columns:
        features_df[col] = features_df[col].shift(1)
    return features_df


class BaseIndicator(ABC):
    """
    Basisklasse für Indicator-Plugins.

    Indicators berechnen technische Features basierend auf OHLCV-Daten.

    WICHTIG - Lookahead Bias Prevention:
    Alle Features MÜSSEN um 1 Bar geshiptet werden bevor sie zurückgegeben
    werden. Nutze die `shift_features()` Hilfsfunktion dafür.

    WICHTIG - Division durch Null:
    Nutze `safe_divide()` für alle Divisionen um konsistentes
    NaN-Handling zu gewährleisten.

    Beispiel:
        ```python
        from fwbg.plugins import BaseIndicator
        from fwbg.plugins.indicator import shift_features, safe_divide

        class RSIIndicator(BaseIndicator):
            group = "momentum"

            def compute(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
                features = {}

                # Features berechnen mit safe_divide
                features['rsi_value'] = safe_divide(gain, loss)

                # KRITISCH: Features shiften und zurückgeben
                features_df = shift_features(features, df.index)
                return pd.concat([df, features_df], axis=1)

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

        WICHTIG: Alle Features müssen mit shift_features() um 1 Bar
        geshiptet werden um Lookahead Bias zu vermeiden!
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
