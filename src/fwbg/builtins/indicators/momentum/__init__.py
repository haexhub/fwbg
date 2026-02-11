"""
Momentum Indicator Plugin.

Enthält: RSI, Stochastic, Williams %R, Ultimate Oscillator, ROC.
"""
from typing import List
import pandas as pd
import ta

from fwbg.plugins import BaseIndicator
from fwbg.plugins.indicator import shift_features
from fwbg.core import register_indicator
from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext


@register_indicator("momentum")
class MomentumIndicators(BasePlugin):
    """
    Momentum-Indikatoren für Trading-Strategien.

    Features:
    - RSI (7, 14, 21 Perioden)
    - Stochastic K/D (14, 21 Perioden)
    - Williams %R (14, 21 Perioden)
    - Ultimate Oscillator
    - Rate of Change (5, 10, 20 Perioden)
    """

    # BasePlugin required attributes
    name = "momentum"
    version = "2.0.0"
    phase = PluginPhase.INDICATORS

    # Optional attributes
    stateful = False
    cacheable = True

    # Legacy attribute for backwards compatibility
    group = "momentum"

    def __init__(self) -> None:
        """Initialize MomentumIndicators plugin."""
        super().__init__()
        self._feature_columns: List[str] = []

    def validate(self) -> bool:
        """Validate that the plugin is properly configured."""
        return True

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        """
        Execute the momentum indicators on the pipeline context.

        Args:
            ctx: Pipeline context with DataFrame
            **params: Optional parameters for compute()

        Returns:
            Updated pipeline context with momentum indicator columns
        """
        result_df = self.compute(ctx.df, **params)
        ctx.df = result_df
        return ctx

    def compute(
        self,
        df: pd.DataFrame,
        rsi_periods: List[int] = None,
        stoch_periods: List[int] = None,
        williams_periods: List[int] = None,
        roc_periods: List[int] = None,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet alle Momentum-Indikatoren.

        Args:
            df: DataFrame mit OHLC-Daten (O, H, L, C)
            rsi_periods: RSI-Perioden (default: [7, 14, 21])
            stoch_periods: Stochastic-Perioden (default: [14, 21])
            williams_periods: Williams %R Perioden (default: [14, 21])
            roc_periods: ROC-Perioden (default: [5, 10, 20])

        Returns:
            DataFrame mit Momentum-Features
        """
        if rsi_periods is None:
            rsi_periods = [7, 14, 21]
        if stoch_periods is None:
            stoch_periods = [14, 21]
        if williams_periods is None:
            williams_periods = [14, 21]
        if roc_periods is None:
            roc_periods = [5, 10, 20]

        features = {}

        # RSI
        for period in rsi_periods:
            features[f"mom_rsi_{period}"] = ta.momentum.rsi(df["C"], window=period)

        # Stochastic
        for period in stoch_periods:
            stoch = ta.momentum.StochasticOscillator(
                df["H"], df["L"], df["C"], window=period
            )
            features[f"mom_stoch_k_{period}"] = stoch.stoch()
            features[f"mom_stoch_d_{period}"] = stoch.stoch_signal()

        # Williams %R
        for period in williams_periods:
            features[f"mom_williams_{period}"] = ta.momentum.williams_r(
                df["H"], df["L"], df["C"], lbp=period
            )

        # Ultimate Oscillator
        features["mom_uo"] = ta.momentum.ultimate_oscillator(
            df["H"], df["L"], df["C"]
        )

        # Rate of Change
        for period in roc_periods:
            features[f"mom_roc_{period}"] = ta.momentum.roc(df["C"], window=period)

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        """Gibt Liste aller Momentum-Feature-Spalten zurück."""
        return [
            # RSI
            "mom_rsi_7", "mom_rsi_14", "mom_rsi_21",
            # Stochastic
            "mom_stoch_k_14", "mom_stoch_d_14",
            "mom_stoch_k_21", "mom_stoch_d_21",
            # Williams
            "mom_williams_14", "mom_williams_21",
            # Ultimate Oscillator
            "mom_uo",
            # ROC
            "mom_roc_5", "mom_roc_10", "mom_roc_20",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "rsi_periods": [7, 14, 21],
            "stoch_periods": [14, 21],
            "williams_periods": [14, 21],
            "roc_periods": [5, 10, 20],
        }


__all__ = ["MomentumIndicators"]
