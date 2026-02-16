"""
Regime Cluster Indicator Plugin.

Computes a composite regime score from orthogonal market structure inputs,
then assigns quantile-based cluster labels (0/1/2) for use in the bitmask
regime_filter_grid.

Score semantics:
- High score = favorable for directional trading (trending, persistent, low entropy)
- Low score = unfavorable (choppy, random, mean-reverting)

Cluster labels:
- 0 = unfavorable phase (lower third)
- 1 = neutral phase (middle third)
- 2 = favorable phase (upper third)
"""
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from fwbg.core import register_indicator
from fwbg.plugins import BaseIndicator
from fwbg.plugins.indicator import shift_features


class Columns:
    """Output column names for regime_cluster."""

    SCORE = "regime_cluster_score"
    LABEL = "regime_cluster_label"
    SCORE_CHG = "regime_cluster_score_chg"
    N_INPUTS = "regime_cluster_n_inputs"

    ALL = [SCORE, LABEL, SCORE_CHG, N_INPUTS]


# Core inputs: (column_name, sign_multiplier)
# Positive sign = higher value → more favorable for directional trading
CORE_INPUTS: List[tuple] = [
    ("regime_hurst_200", 1.0),         # Persistence/trending
    ("regime_entropy_100", -1.0),      # Predictability (flip: low entropy = good)
    ("regime_vr_200_5", 1.0),          # Momentum vs mean-reversion (centered at 0)
    ("vol_atr_pct_14_rank", 1.0),      # Volatility level
    ("regime_hurst_divergence", 1.0),  # Regime-shift signal
]

# Optional inputs (used only when column is present)
OPTIONAL_INPUTS: List[tuple] = [
    ("regime_risk_composite", 1.0),    # Macro risk-on/off
]


def _rolling_zscore(
    series: np.ndarray,
    window: int,
) -> np.ndarray:
    """Rolling z-score normalization."""
    s = pd.Series(series)
    rolling_mean = s.rolling(window, min_periods=window // 2).mean()
    rolling_std = s.rolling(window, min_periods=window // 2).std()
    return ((s - rolling_mean) / (rolling_std + 1e-10)).values


@register_indicator("regime_cluster")
class RegimeClusterIndicator(BaseIndicator):
    """
    Composite regime score with quantile-based clustering.

    Combines orthogonal market structure inputs into a single score,
    then assigns cluster labels via rolling quantiles.
    """

    name = "regime_cluster"
    version = "1.0.0"
    group = "regime"
    depends_on = ["regime", "volatility"]

    def compute(
        self,
        df: pd.DataFrame,
        zscore_window: int = 200,
        quantile_window: int = 500,
        n_regimes: int = 3,
        **params,
    ) -> pd.DataFrame:
        """
        Compute composite regime score and cluster labels.

        Args:
            df: DataFrame with pre-computed regime and volatility columns
            zscore_window: Rolling window for z-scoring inputs
            quantile_window: Rolling window for quantile computation
            n_regimes: Number of regime clusters (quantile bins)
        """
        n = len(df)
        zscored: List[np.ndarray] = []

        # Collect and z-score core inputs
        for col, sign in CORE_INPUTS:
            if col not in df.columns:
                continue
            values = df[col].values.astype(float).copy()
            # Center variance ratio at 0 (VR - 1.0)
            if col.startswith("regime_vr_"):
                values = values - 1.0
            values = values * sign
            zscored.append(_rolling_zscore(values, zscore_window))

        # Collect optional inputs
        for col, sign in OPTIONAL_INPUTS:
            if col not in df.columns:
                continue
            values = df[col].values.astype(float) * sign
            zscored.append(_rolling_zscore(values, zscore_window))

        n_inputs = len(zscored)

        if n_inputs == 0:
            features: Dict[str, Union[np.ndarray, float]] = {
                Columns.SCORE: np.full(n, np.nan),
                Columns.LABEL: np.full(n, np.nan),
                Columns.SCORE_CHG: np.full(n, np.nan),
                Columns.N_INPUTS: 0,
            }
            features_df = shift_features(features, df.index)
            return pd.concat([df, features_df], axis=1)

        # Composite score: equally-weighted average of z-scored inputs
        stacked = np.column_stack(zscored)
        score = np.nanmean(stacked, axis=1)

        # Quantile clustering
        score_series = pd.Series(score, index=df.index)
        cluster = np.full(n, np.nan)

        for k in range(1, n_regimes):
            quantile = score_series.rolling(
                quantile_window, min_periods=quantile_window // 2
            ).quantile(k / n_regimes)
            mask = score > quantile.values
            cluster[mask] = k

        # Fill remaining NaN values below lowest quantile with 0
        valid_score = ~np.isnan(score)
        cluster[valid_score & np.isnan(cluster)] = 0

        # Score change over 24 bars
        score_chg = score_series - score_series.shift(24)

        features = {
            Columns.SCORE: score,
            Columns.LABEL: cluster,
            Columns.SCORE_CHG: score_chg.values,
            Columns.N_INPUTS: np.full(n, n_inputs, dtype=np.int8),
        }

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return list(Columns.ALL)

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "zscore_window": 200,
            "quantile_window": 500,
            "n_regimes": 3,
        }


__all__ = ["RegimeClusterIndicator", "Columns"]
