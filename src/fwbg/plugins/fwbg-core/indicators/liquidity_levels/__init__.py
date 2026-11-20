"""
Liquidity Levels Indicator Plugin.

Detects liquidity pools from equal highs/lows and sweep events:
- Equal highs: multiple swing highs at similar price -> stops cluster above
- Equal lows: multiple swing lows at similar price -> stops cluster below
- Sweeps: price briefly exceeds level then reverses (stop hunt)

Swings are only used once confirmed (``swing_lookback`` bars after the swing
bar), a level is consumed as soon as price trades through it, and each level
sweeps at most once.  See ``_compute_liquidity_features`` for details.
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
    """Group price levels within tolerance and return clusters with >= min_touches.

    Each cluster is ``(level, count, last_bar, member_bars)``.
    """
    if len(prices) == 0:
        return []

    order = np.argsort(prices)
    sorted_prices = prices[order]
    sorted_indices = indices[order]

    clusters = []
    cluster_start = 0

    for i in range(1, len(sorted_prices) + 1):
        if (
            i == len(sorted_prices)
            or sorted_prices[i] - sorted_prices[cluster_start] > tolerance
        ):
            count = i - cluster_start
            if count >= min_touches:
                level = np.mean(sorted_prices[cluster_start:i])
                members = sorted_indices[cluster_start:i]
                clusters.append(
                    (level, count, int(np.max(members)), [int(m) for m in members])
                )
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
    """Compute all liquidity level features (unshifted, row i uses bars <= i).

    Causality: a swing at bar ``s`` is a window extreme over ``[s-k, s+k]``
    (``k = swing_lookback``) and only becomes known at bar ``s + k``.  It is
    added to the active set exactly then, so row ``i`` never sees a bar after
    ``i``.  The caller applies the usual 1-bar feature shift on top.

    Level lifecycle: an active swing is consumed when a later bar closes
    through it (close above a swing high / below a swing low) or wicks beyond it
    by more than the equality tolerance.  When a sweep fires, every swing of the
    swept level is consumed, so a level sweeps at most once and a level that
    price took out is no longer reported as liquidity above/below.  A wick that
    stays inside the tolerance band without a sweep leaves the swing active.

    Tied pivots: when equal highs (lows) sit within ``k`` bars of each other
    only the first one is a swing, so adjacent equal bars are one touch, not
    several.

    Warm-up rows (no ATR yet, or no bar that could confirm a swing yet) are NaN
    in every output column.
    """
    n = len(df)
    h = df["H"].values
    low = df["L"].values
    c = df["C"].values
    k = swing_lookback

    # ATR (needs a full window; earlier rows are warm-up)
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - low, np.maximum(np.abs(h - prev_c), np.abs(low - prev_c)))
    atr = pd.Series(tr).rolling(atr_period, min_periods=atr_period).mean().values

    first_valid = max(2 * k, atr_period - 1)

    eqh_dist = np.full(n, np.nan)
    eql_dist = np.full(n, np.nan)
    eqh_count = np.full(n, np.nan)
    eql_count = np.full(n, np.nan)
    eqh_active = np.full(n, np.nan)
    eql_active = np.full(n, np.nan)
    sweep_up = np.full(n, np.nan)
    sweep_down = np.full(n, np.nan)

    # Active (confirmed, not yet traded-through) swing points
    hi_bars: List[int] = []
    hi_prices: List[float] = []
    lo_bars: List[int] = []
    lo_prices: List[float] = []

    for i in range(n):
        # Confirm the swing at bar s = i - k (its window [s-k, s+k] ends at i)
        s = i - k
        if s >= k:
            left_h = h[s - k : s]
            right_h = h[s + 1 : i + 1]
            if h[s] > left_h.max() and h[s] >= right_h.max():
                hi_bars.append(s)
                hi_prices.append(h[s])
            left_l = low[s - k : s]
            right_l = low[s + 1 : i + 1]
            if low[s] < left_l.min() and low[s] <= right_l.min():
                lo_bars.append(s)
                lo_prices.append(low[s])

        # Drop swings older than the lookback window
        oldest = i - lookback_window
        while hi_bars and hi_bars[0] < oldest:
            hi_bars.pop(0)
            hi_prices.pop(0)
        while lo_bars and lo_bars[0] < oldest:
            lo_bars.pop(0)
            lo_prices.pop(0)

        swept_hi: set = set()
        swept_lo: set = set()
        # Equality band for this bar; NaN during ATR warm-up (no wick rule)
        tolerance = np.nan
        if i >= first_valid and not np.isnan(atr[i]):
            current_atr = atr[i] if atr[i] > EPSILON else 1.0
            tolerance = current_atr * tolerance_atr_mult

            eqh_count[i] = eql_count[i] = 0.0
            eqh_active[i] = eql_active[i] = 0.0
            sweep_up[i] = sweep_down[i] = 0.0

            if hi_bars:
                eq_highs = _find_equal_levels(
                    np.array(hi_prices),
                    np.array(hi_bars),
                    tolerance,
                    min_touches,
                )
                above = [cl for cl in eq_highs if cl[0] > c[i]]
                if above:
                    nearest = min(above, key=lambda x: x[0])
                    eqh_dist[i] = (nearest[0] - c[i]) / current_atr
                    eqh_count[i] = nearest[1]
                    eqh_active[i] = 1.0
                    if h[i] > nearest[0] and c[i] < nearest[0]:
                        sweep_up[i] = 1.0
                        swept_hi = set(nearest[3])

            if lo_bars:
                eq_lows = _find_equal_levels(
                    np.array(lo_prices),
                    np.array(lo_bars),
                    tolerance,
                    min_touches,
                )
                below = [cl for cl in eq_lows if cl[0] < c[i]]
                if below:
                    nearest = max(below, key=lambda x: x[0])
                    eql_dist[i] = (c[i] - nearest[0]) / current_atr
                    eql_count[i] = nearest[1]
                    eql_active[i] = 1.0
                    if low[i] < nearest[0] and c[i] > nearest[0]:
                        sweep_down[i] = 1.0
                        swept_lo = set(nearest[3])

        # Consume swings bar i took out (visible from bar i+1 on). A swing is
        # always kept unless a rule below removes it; before the ATR exists only
        # the close rule applies.
        tol = 0.0 if np.isnan(tolerance) else tolerance
        wick_rule = not np.isnan(tolerance)
        if hi_bars:
            keep = [
                j
                for j, p in enumerate(hi_prices)
                if not (
                    c[i] > p or (wick_rule and h[i] > p + tol) or hi_bars[j] in swept_hi
                )
            ]
            if len(keep) != len(hi_prices):
                hi_bars = [hi_bars[j] for j in keep]
                hi_prices = [hi_prices[j] for j in keep]
        if lo_bars:
            keep = [
                j
                for j, p in enumerate(lo_prices)
                if not (
                    c[i] < p
                    or (wick_rule and low[i] < p - tol)
                    or lo_bars[j] in swept_lo
                )
            ]
            if len(keep) != len(lo_prices):
                lo_bars = [lo_bars[j] for j in keep]
                lo_prices = [lo_prices[j] for j in keep]

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
    version = "1.1.0"
    benefits_from_stationary = False
    group = "structure"

    _FEATURES = [
        "liq_eqh_dist",
        "liq_eql_dist",
        "liq_eqh_count",
        "liq_eql_count",
        "liq_eqh_active",
        "liq_eql_active",
        "liq_sweep_up",
        "liq_sweep_down",
    ]

    def compute(
        self,
        df: pd.DataFrame,
        swing_lookback: int = 10,
        tolerance_atr_mult: float = 0.2,
        atr_period: int = 14,
        min_touches: int = 2,
        lookback_window: int = 200,
        **params,
    ) -> pd.DataFrame:
        features = _compute_liquidity_features(
            df,
            swing_lookback,
            tolerance_atr_mult,
            atr_period,
            min_touches,
            lookback_window,
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
            "swing_lookback": 10,
            "tolerance_atr_mult": 0.2,
            "atr_period": 14,
            "min_touches": 2,
            "lookback_window": 200,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "swing_lookback": {
                "type": "int",
                "default": 10,
                "description": "N-bar lookback for swing high/low detection.",
                "min": 3,
                "max": 50,
                "step": 1,
            },
            "tolerance_atr_mult": {
                "type": "float",
                "default": 0.2,
                "description": "How close swing highs/lows must be (in ATR units) to count as equal.",
                "min": 0.05,
                "max": 1.0,
                "step": 0.05,
            },
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR period for normalizing distances and tolerance.",
                "min": 2,
                "max": 100,
                "step": 1,
            },
            "min_touches": {
                "type": "int",
                "default": 2,
                "description": "Minimum swing points at similar price to form a liquidity level.",
                "min": 2,
                "max": 5,
                "step": 1,
            },
            "lookback_window": {
                "type": "int",
                "default": 200,
                "description": "Bars to look back for swing points when building liquidity levels.",
                "min": 50,
                "max": 1000,
                "step": 50,
            },
        }


__all__ = ["LiquidityLevelsIndicator"]
