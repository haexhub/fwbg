"""
Supply/Demand Flip Zone Indicator Plugin.

Detects zones where supply turns into demand (or vice versa):
- Identifies swing highs/lows as potential S/R zones
- When a zone is broken, it flips polarity (support -> resistance, resistance -> support)
- Tracks active flip zones with distance, strength, and touch count
"""
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, EPSILON


def _find_swing_points(highs: np.ndarray, lows: np.ndarray, lookback: int):
    """Find swing highs and swing lows using a simple N-bar lookback."""
    n = len(highs)
    swing_highs = []
    swing_lows = []

    for i in range(lookback, n - lookback):
        if highs[i] == np.max(highs[i - lookback:i + lookback + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == np.min(lows[i - lookback:i + lookback + 1]):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def _compute_sdf_features(
    df: pd.DataFrame,
    swing_lookback: int,
    zone_atr_width: float,
    atr_period: int,
    max_active_zones: int,
    zone_expiry: int,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """Compute supply/demand flip zone features."""
    n = len(df)
    h = df["H"].values
    low = df["L"].values
    c = df["C"].values

    # ATR
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - low, np.maximum(np.abs(h - prev_c), np.abs(low - prev_c)))
    atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values

    swing_highs, swing_lows = _find_swing_points(h, low, swing_lookback)

    bull_active = np.zeros(n)
    bear_active = np.zeros(n)
    bull_dist = np.full(n, np.nan)
    bear_dist = np.full(n, np.nan)
    bull_strength = np.full(n, np.nan)
    bear_strength = np.full(n, np.nan)
    bull_touches = np.zeros(n)
    bear_touches = np.zeros(n)

    resistance_zones = []
    support_zones = []
    flip_zones = []

    sh_by_bar = {bar: price for bar, price in swing_highs}
    sl_by_bar = {bar: price for bar, price in swing_lows}

    for i in range(n):
        current_atr = atr[i] if atr[i] > EPSILON else 1.0
        zone_width = current_atr * zone_atr_width

        if i in sh_by_bar:
            resistance_zones.append({"level": sh_by_bar[i], "bar": i})
        if i in sl_by_bar:
            support_zones.append({"level": sl_by_bar[i], "bar": i})

        surviving_resistance = []
        for zone in resistance_zones:
            if i - zone["bar"] > zone_expiry:
                continue
            if c[i] > zone["level"] + zone_width:
                strength = (c[i] - zone["level"]) / current_atr
                flip_zones.append({
                    "level": zone["level"], "type": "bull", "bar": i,
                    "strength": strength, "touches": 0,
                })
            else:
                surviving_resistance.append(zone)
        resistance_zones = surviving_resistance

        surviving_support = []
        for zone in support_zones:
            if i - zone["bar"] > zone_expiry:
                continue
            if c[i] < zone["level"] - zone_width:
                strength = (zone["level"] - c[i]) / current_atr
                flip_zones.append({
                    "level": zone["level"], "type": "bear", "bar": i,
                    "strength": strength, "touches": 0,
                })
            else:
                surviving_support.append(zone)
        support_zones = surviving_support

        surviving_flips = []
        for fz in flip_zones:
            if i - fz["bar"] > zone_expiry:
                continue
            if fz["type"] == "bull" and c[i] < fz["level"] - zone_width:
                continue
            if fz["type"] == "bear" and c[i] > fz["level"] + zone_width:
                continue
            if abs(c[i] - fz["level"]) <= zone_width:
                fz["touches"] += 1
            surviving_flips.append(fz)
        flip_zones = surviving_flips[-max_active_zones:]

        nearest_bull_d = np.inf
        nearest_bull_s = 0.0
        nearest_bull_t = 0
        nearest_bear_d = np.inf
        nearest_bear_s = 0.0
        nearest_bear_t = 0

        for fz in flip_zones:
            if fz["type"] == "bull":
                d = (c[i] - fz["level"]) / current_atr
                if d > 0 and d < nearest_bull_d:
                    nearest_bull_d = d
                    nearest_bull_s = fz["strength"]
                    nearest_bull_t = fz["touches"]
            else:
                d = (fz["level"] - c[i]) / current_atr
                if d > 0 and d < nearest_bear_d:
                    nearest_bear_d = d
                    nearest_bear_s = fz["strength"]
                    nearest_bear_t = fz["touches"]

        if nearest_bull_d < np.inf:
            bull_active[i] = 1.0
            bull_dist[i] = nearest_bull_d
            bull_strength[i] = nearest_bull_s
            bull_touches[i] = nearest_bull_t

        if nearest_bear_d < np.inf:
            bear_active[i] = 1.0
            bear_dist[i] = nearest_bear_d
            bear_strength[i] = nearest_bear_s
            bear_touches[i] = nearest_bear_t

    return {
        "sdf_bull_active": bull_active,
        "sdf_bear_active": bear_active,
        "sdf_bull_dist": bull_dist,
        "sdf_bear_dist": bear_dist,
        "sdf_bull_strength": bull_strength,
        "sdf_bear_strength": bear_strength,
        "sdf_bull_touches": bull_touches,
        "sdf_bear_touches": bear_touches,
    }


@register_indicator("supply_demand_flip")
class SupplyDemandFlipIndicator(BaseIndicator):
    """Supply/Demand Flip Zone features for ML trading."""

    name = "supply_demand_flip"
    version = "1.0.0"
    benefits_from_stationary = False
    group = "structure"

    _FEATURES = [
        "sdf_bull_active", "sdf_bear_active",
        "sdf_bull_dist", "sdf_bear_dist",
        "sdf_bull_strength", "sdf_bear_strength",
        "sdf_bull_touches", "sdf_bear_touches",
    ]

    def compute(
        self, df: pd.DataFrame,
        swing_lookback: int = 10, zone_atr_width: float = 0.3,
        atr_period: int = 14, max_active_zones: int = 20,
        zone_expiry: int = 200, **params,
    ) -> pd.DataFrame:
        features = _compute_sdf_features(
            df, swing_lookback, zone_atr_width, atr_period,
            max_active_zones, zone_expiry,
        )
        if not features:
            return df
        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    def get_signal_columns(self) -> List[str]:
        return ["sdf_bull_active", "sdf_bear_active"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "swing_lookback": 10, "zone_atr_width": 0.3,
            "atr_period": 14, "max_active_zones": 20, "zone_expiry": 200,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "swing_lookback": {
                "type": "int", "default": 10,
                "description": "N-bar lookback for swing high/low detection.",
                "min": 3, "max": 50, "step": 1,
            },
            "zone_atr_width": {
                "type": "float", "default": 0.3,
                "description": "Zone width as fraction of ATR.",
                "min": 0.1, "max": 1.0, "step": 0.1,
            },
            "atr_period": {
                "type": "int", "default": 14,
                "description": "ATR period for normalizing distances and zone width.",
                "min": 2, "max": 100, "step": 1,
            },
            "max_active_zones": {
                "type": "int", "default": 20,
                "description": "Maximum active flip zones to track.",
                "min": 5, "max": 50, "step": 5,
            },
            "zone_expiry": {
                "type": "int", "default": 200,
                "description": "Bars after which a zone expires.",
                "min": 50, "max": 1000, "step": 50,
            },
        }


__all__ = ["SupplyDemandFlipIndicator"]
