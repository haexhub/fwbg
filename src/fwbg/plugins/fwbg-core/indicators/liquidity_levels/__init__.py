"""
Liquidity Levels Indicator Plugin.

Detects liquidity pools from equal highs/lows and sweep events:
- Equal highs: multiple swing highs at similar price -> stops cluster above
- Equal lows: multiple swing lows at similar price -> stops cluster below
- Sweeps: price briefly exceeds level then reverses (stop hunt)
"""
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, EPSILON


def _find_equal_levels(
    prices: np.ndarray,
    indices: np.ndarray,
    tolerance: float,
    min_touches: int,
):
    """Group price levels within tolerance and return clusters with >= min_touches."""
    if len(prices) == 0:
        return []

    order = np.argsort(prices)
    sorted_prices = prices[order]
    sorted_indices = indices[order]

    clusters = []
    cluster_start = 0

    for i in range(1, len(sorted_prices) + 1):
        if i == len(sorted_prices) or sorted_prices[i] - sorted_prices[cluster_start] > tolerance:
            count = i - cluster_start
            if count >= min_touches:
                level = np.mean(sorted_prices[cluster_start:i])
                last_bar = int(np.max(sorted_indices[cluster_start:i]))
                clusters.append((level, count, last_bar))
            cluster_start = i

    return clusters


def _compute_liquidity_features(
    df: pd.DataFrame,
    swing_lookback: int,
    tolerance_atr_mult: float,
    atr_period: int,
    min_touches: int,
    lookback_window: int,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """Compute all liquidity level features."""
    n = len(df)
    h = df["H"].values
    l = df["L"].values
    c = df["C"].values

    # ATR
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values

    # Find all swing highs and lows
    swing_high_bars = []
    swing_high_prices = []
    swing_low_bars = []
    swing_low_prices = []

    for i in range(swing_lookback, n - swing_lookback):
        if h[i] == np.max(h[i - swing_lookback:i + swing_lookback + 1]):
            swing_high_bars.append(i)
            swing_high_prices.append(h[i])
        if l[i] == np.min(l[i - swing_lookback:i + swing_lookback + 1]):
            swing_low_bars.append(i)
            swing_low_prices.append(l[i])

    swing_high_bars = np.array(swing_high_bars)
    swing_high_prices = np.array(swing_high_prices)
    swing_low_bars = np.array(swing_low_bars)
    swing_low_prices = np.array(swing_low_prices)

    eqh_dist = np.full(n, np.nan)
    eql_dist = np.full(n, np.nan)
    eqh_count = np.zeros(n)
    eql_count = np.zeros(n)
    eqh_active = np.zeros(n)
    eql_active = np.zeros(n)
    sweep_up = np.zeros(n)
    sweep_down = np.zeros(n)

    for i in range(swing_lookback * 2, n):
        current_atr = atr[i] if atr[i] > EPSILON else 1.0
        tolerance = current_atr * tolerance_atr_mult

        sh_mask = (swing_high_bars < i) & (swing_high_bars >= i - lookback_window)
        sl_mask = (swing_low_bars < i) & (swing_low_bars >= i - lookback_window)

        if sh_mask.any():
            eq_highs = _find_equal_levels(
                swing_high_prices[sh_mask], swing_high_bars[sh_mask],
                tolerance, min_touches,
            )
            above = [(lvl, cnt) for lvl, cnt, _ in eq_highs if lvl > c[i]]
            if above:
                nearest = min(above, key=lambda x: x[0] - c[i])
                eqh_dist[i] = (nearest[0] - c[i]) / current_atr
                eqh_count[i] = nearest[1]
                eqh_active[i] = 1.0
                if h[i] > nearest[0] and c[i] < nearest[0]:
                    sweep_up[i] = 1.0

        if sl_mask.any():
            eq_lows = _find_equal_levels(
                swing_low_prices[sl_mask], swing_low_bars[sl_mask],
                tolerance, min_touches,
            )
            below = [(lvl, cnt) for lvl, cnt, _ in eq_lows if lvl < c[i]]
            if below:
                nearest = max(below, key=lambda x: x[0])
                eql_dist[i] = (c[i] - nearest[0]) / current_atr
                eql_count[i] = nearest[1]
                eql_active[i] = 1.0
                if l[i] < nearest[0] and c[i] > nearest[0]:
                    sweep_down[i] = 1.0

    return {
        "liq_eqh_dist": eqh_dist,
        "liq_eql_dist": eql_dist,
        "liq_eqh_count": eqh_count,
        "liq_eql_count": eql_count,
        "liq_eqh_active": eqh_active,
        "liq_eql_active": eql_active,
        "liq_sweep_up": sweep_up,
        "liq_sweep_down": sweep_down,
    }


@register_indicator("liquidity_levels")
class LiquidityLevelsIndicator(BaseIndicator):
    """Liquidity level detection features for ML trading."""

    name = "liquidity_levels"
    version = "1.0.0"
    benefits_from_stationary = False
    group = "structure"

    _FEATURES = [
        "liq_eqh_dist", "liq_eql_dist",
        "liq_eqh_count", "liq_eql_count",
        "liq_eqh_active", "liq_eql_active",
        "liq_sweep_up", "liq_sweep_down",
    ]

    def compute(
        self, df: pd.DataFrame,
        swing_lookback: int = 10, tolerance_atr_mult: float = 0.2,
        atr_period: int = 14, min_touches: int = 2,
        lookback_window: int = 200, **params,
    ) -> pd.DataFrame:
        features = _compute_liquidity_features(
            df, swing_lookback, tolerance_atr_mult, atr_period,
            min_touches, lookback_window,
        )
        if not features:
            return df
        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    def get_signal_columns(self) -> List[str]:
        return ["liq_eqh_active", "liq_eql_active", "liq_sweep_up", "liq_sweep_down"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "swing_lookback": 10, "tolerance_atr_mult": 0.2,
            "atr_period": 14, "min_touches": 2, "lookback_window": 200,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "swing_lookback": {
                "type": "int", "default": 10,
                "description": "N-bar lookback for swing high/low detection.",
                "min": 3, "max": 50, "step": 1,
            },
            "tolerance_atr_mult": {
                "type": "float", "default": 0.2,
                "description": "How close swing highs/lows must be (in ATR units) to count as equal.",
                "min": 0.05, "max": 1.0, "step": 0.05,
            },
            "atr_period": {
                "type": "int", "default": 14,
                "description": "ATR period for normalizing distances and tolerance.",
                "min": 2, "max": 100, "step": 1,
            },
            "min_touches": {
                "type": "int", "default": 2,
                "description": "Minimum swing points at similar price to form a liquidity level.",
                "min": 2, "max": 5, "step": 1,
            },
            "lookback_window": {
                "type": "int", "default": 200,
                "description": "Bars to look back for swing points when building liquidity levels.",
                "min": 50, "max": 1000, "step": 50,
            },
        }


__all__ = ["LiquidityLevelsIndicator"]
