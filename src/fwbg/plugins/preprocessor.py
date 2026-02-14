"""
BasePreprocessor - Abstrakte Basisklasse für Preprocessing-Plugins.
"""
from abc import ABC, abstractmethod
import pandas as pd


class BasePreprocessor(ABC):
    """
    Basisklasse für Preprocessing-Plugins.

    Preprocessors transformieren OHLC-Daten (z.B. Fractional Differentiation
    für Stationarität). Indicator-Plugins deklarieren in ihrer Manifest-Datei
    per ``benefits_from_stationary``, ob sie von stationären Daten profitieren.
    Wenn ja, wird die Feature-Berechnung nach dem Preprocessing gemacht,
    ansonsten auf den Original-Daten.

    Preprocessors folgen dem sklearn fit/transform Pattern
    um Lookahead Bias zu verhindern.

    WICHTIG - Lookahead Bias Prevention:
    - fit() lernt Parameter NUR von Train-Daten
    - transform() wendet diese Parameter auf Test/OOS-Daten an
    - fit() wird für jeden CV-Fold SEPARAT aufgerufen
    """

    # Plugin-Name (wird vom Registry gesetzt)
    name: str = "base"

    # Ausführungsreihenfolge (niedriger = früher)
    order: int = 100

    # Gelernte Parameter (werden von fit() gesetzt)
    fitted_: bool = False

    def fit(self, df: pd.DataFrame, **params) -> "BasePreprocessor":
        """
        Lernt Parameter von Train-Daten.

        WICHTIG: Diese Methode MUSS auf Train-Daten aufgerufen werden,
        BEVOR transform() verwendet wird. Sie darf NIEMALS auf Test/OOS-Daten
        laufen, um Lookahead Bias zu verhindern.

        Args:
            df: Train DataFrame mit OHLC-Daten
            **params: Preprocessor-spezifische Parameter

        Returns:
            self (fitted preprocessor)
        """
        self.fitted_ = True
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        Transformiert den DataFrame mit gelernten Parametern.

        Diese Methode wendet die in fit() gelernten Parameter an.
        Sie kann auf Train, Test und OOS-Daten angewendet werden.

        Args:
            df: Input DataFrame mit OHLC-Daten
            **params: Zusätzliche Transform-Parameter (optional)

        Returns:
            Transformierter DataFrame

        Raises:
            RuntimeError: Wenn fit() noch nicht aufgerufen wurde
        """
        if not self.fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__}: fit() must be called before transform()"
            )

    def fit_transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        Kombiniert fit() und transform() für Train-Daten.

        Args:
            df: Train DataFrame
            **params: Preprocessor-spezifische Parameter

        Returns:
            Transformierter DataFrame
        """
        return self.fit(df, **params).transform(df, **params)

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
