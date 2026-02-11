"""
Example Custom Indicator Plugin for FWBG.

This file demonstrates how to create a custom indicator plugin
that can be used in the FWBG trading strategy backtester.

To use this plugin:
1. Copy this folder to ~/.fwbg/plugins/
2. The plugin will be automatically discovered on startup
3. Reference it in your strategy JSON as {"name": "custom_zscore", "params": {...}}

Plugin Requirements:
- Inherit from BasePlugin
- Define class attributes: name, version, phase
- Implement execute() and validate() methods
- For indicators: stateful=False, cacheable=True is typical
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext


class CustomZScoreIndicator(BasePlugin):
    """
    Example: Z-Score Indicator Plugin.

    Computes rolling z-scores of price data, useful for mean-reversion strategies.
    This demonstrates a stateless indicator plugin.
    """

    # Required class attributes
    name = "custom_zscore"
    version = "1.0.0"
    phase = PluginPhase.INDICATORS

    # Optional: stateless indicators don't need fit/transform pattern
    stateful = False
    cacheable = True

    def __init__(self) -> None:
        """Initialize plugin instance state."""
        super().__init__()
        self._feature_columns: List[str] = []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """
        Return default parameters for this plugin.

        These can be overridden in the strategy JSON config.
        """
        return {
            "periods": [20, 50, 100],
            "columns": ["C"],  # Which columns to compute z-score for
        }

    def validate(self) -> bool:
        """
        Validate plugin is ready to run.

        Check that required dependencies are available.
        """
        try:
            import numpy
            import pandas

            return True
        except ImportError as e:
            raise ImportError(f"Missing required dependency: {e}")

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        """
        Compute z-score indicators and add to DataFrame.

        Args:
            ctx: Pipeline context with OHLCV DataFrame
            **params: Plugin parameters (merged with defaults)

        Returns:
            Updated PipelineContext with new indicator columns
        """
        # Merge default params with provided params
        merged_params = {**self.get_default_params(), **params}

        periods = merged_params["periods"]
        columns = merged_params["columns"]

        df = ctx.df
        features: Dict[str, pd.Series] = {}

        for col in columns:
            if col not in df.columns:
                continue

            series = df[col]

            for period in periods:
                # Rolling mean and std
                rolling_mean = series.rolling(window=period).mean()
                rolling_std = series.rolling(window=period).std()

                # Z-score = (value - mean) / std
                zscore = (series - rolling_mean) / (rolling_std + 1e-10)

                # Store with naming convention: {plugin_name}_{metric}_{period}
                features[f"custom_zscore_{col}_{period}"] = zscore

                # Also add percentile rank (useful for thresholds)
                features[f"custom_zscore_{col}_{period}_rank"] = (
                    series.rolling(window=period).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                        raw=False,
                    )
                )

        # Shift features by 1 to prevent lookahead bias
        # (we only know the indicator AFTER the bar closes)
        features_df = pd.DataFrame(features, index=df.index).shift(1)

        # Track created feature columns
        self._feature_columns = list(features_df.columns)

        # Add to DataFrame
        ctx.df = pd.concat([df, features_df], axis=1)

        return ctx

    def get_feature_columns(self) -> List[str]:
        """Return list of feature columns created by this plugin."""
        return self._feature_columns


# You can define multiple plugins in the same file
class CustomMomentumRatioIndicator(BasePlugin):
    """
    Example: Momentum Ratio Indicator.

    Computes ratio of short-term vs long-term momentum.
    """

    name = "custom_momentum_ratio"
    version = "1.0.0"
    phase = PluginPhase.INDICATORS
    stateful = False
    cacheable = True

    def __init__(self) -> None:
        super().__init__()
        self._feature_columns: List[str] = []

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        return {
            "fast_period": 5,
            "slow_period": 20,
        }

    def validate(self) -> bool:
        return True

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        merged_params = {**self.get_default_params(), **params}

        fast = merged_params["fast_period"]
        slow = merged_params["slow_period"]

        df = ctx.df
        close = df["C"]

        # Short-term and long-term returns
        fast_ret = close.pct_change(fast)
        slow_ret = close.pct_change(slow)

        # Momentum ratio
        ratio = fast_ret / (slow_ret.abs() + 1e-10)

        features = {
            f"custom_mom_ratio_{fast}_{slow}": ratio.shift(1),
            f"custom_mom_fast_{fast}": fast_ret.shift(1),
            f"custom_mom_slow_{slow}": slow_ret.shift(1),
        }

        features_df = pd.DataFrame(features, index=df.index)
        self._feature_columns = list(features_df.columns)

        ctx.df = pd.concat([df, features_df], axis=1)
        return ctx

    def get_feature_columns(self) -> List[str]:
        return self._feature_columns


# Export all plugin classes
__all__ = ["CustomZScoreIndicator", "CustomMomentumRatioIndicator"]
