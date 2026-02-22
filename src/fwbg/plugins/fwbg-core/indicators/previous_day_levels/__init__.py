"""
Previous Day Levels Indicator Plugin.

Computes features relative to previous day's high and low:
- Distance to PDH/PDL in ATR units
- Position within daily range
- Break detection (first cross above PDH or below PDL)
- Range expansion/contraction

Timeframe: Intraday only (M1-H4). On daily bars returns df unchanged.
"""
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, EPSILON


def _compute_atr(df: pd.DataFrame, period: int) -> np.ndarray:
    """Compute ATR from OHLC data."""
    highs = df["H"].values
    lows = df["L"].values
    close = df["C"].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
    )
    return pd.Series(tr).rolling(period, min_periods=1).mean().values


def _compute_pdl_features(
    df: pd.DataFrame,
    atr: np.ndarray,
    ma_period: int,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """Compute all previous day level features."""
    features: Dict[str, Union[pd.Series, np.ndarray]] = {}
    n = len(df)

    # Group by trading day
    day_group = df.index.normalize()

    # Per-day high/low
    day_high = df["H"].groupby(day_group).transform("max")
    day_low = df["L"].groupby(day_group).transform("min")

    # Shift by 1 day to get PREVIOUS day high/low
    # Create mapping: date -> (high, low)
    _unique_days = day_group.unique()
    day_hl = pd.DataFrame({
        "high": df["H"].groupby(day_group).max(),
        "low": df["L"].groupby(day_group).min(),
    })
    prev_day_hl = day_hl.shift(1)

    # Map previous day H/L back to each bar
    pdh = prev_day_hl["high"].reindex(day_group).values
    pdl_low = prev_day_hl["low"].reindex(day_group).values
    pd_range = pdh - pdl_low

    close = df["C"].values
    safe_atr = np.where(atr > EPSILON, atr, 1.0)
    safe_range = np.where(np.abs(pd_range) > EPSILON, pd_range, np.nan)

    # Distance features (ATR-normalized)
    features["pdl_high_dist"] = (pdh - close) / safe_atr
    features["pdl_low_dist"] = (close - pdl_low) / safe_atr

    # Position within range: 0=at PDL, 1=at PDH
    features["pdl_position"] = (close - pdl_low) / safe_range

    # Range vs ATR
    features["pdl_range_vs_atr"] = pd_range / safe_atr

    # Binary: above PDH / below PDL
    features["pdl_above_high"] = (close > pdh).astype(float)
    features["pdl_below_low"] = (close < pdl_low).astype(float)

    # Break detection: first bar of the day where close crosses PDH/PDL
    above = close > pdh
    below = close < pdl_low
    day_ids = pd.Series(day_group).factorize()[0]

    high_break = np.zeros(n)
    low_break = np.zeros(n)
    prev_day_id = -1
    already_broke_high = False
    already_broke_low = False
    for i in range(n):
        if day_ids[i] != prev_day_id:
            prev_day_id = day_ids[i]
            already_broke_high = False
            already_broke_low = False
        if above[i] and not already_broke_high:
            high_break[i] = 1.0
            already_broke_high = True
        if below[i] and not already_broke_low:
            low_break[i] = 1.0
            already_broke_low = True

    features["pdl_high_break"] = high_break
    features["pdl_low_break"] = low_break

    # Rolling MA of position
    pos_series = pd.Series(features["pdl_position"])
    features["pdl_range_position_ma"] = pos_series.rolling(
        ma_period, min_periods=1
    ).mean().values

    # Range expanding: current day range > previous day range
    current_day_range = (day_high - day_low).values
    features["pdl_day_range_expanding"] = (current_day_range > pd_range).astype(float)

    # NaN out bars where no previous day data exists
    no_prev_day = np.isnan(pdh)
    for key in features:
        arr = features[key]
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=float)
        else:
            arr = arr.astype(float)
        arr[no_prev_day] = np.nan
        features[key] = arr

    return features


@register_indicator("previous_day_levels")
class PreviousDayLevelsIndicator(BaseIndicator):
    """Previous Day High/Low features for intraday trading."""

    name = "previous_day_levels"
    version = "1.0.0"
    benefits_from_stationary = False
    group = "session"

    _FEATURES = [
        "pdl_high_dist",
        "pdl_low_dist",
        "pdl_position",
        "pdl_range_vs_atr",
        "pdl_above_high",
        "pdl_below_low",
        "pdl_high_break",
        "pdl_low_break",
        "pdl_range_position_ma",
        "pdl_day_range_expanding",
    ]

    def compute(
        self,
        df: pd.DataFrame,
        atr_period: int = 14,
        ma_period: int = 20,
        **params,
    ) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        # Skip daily data — PDL is intraday only
        if len(df) > 1:
            median_diff = df.index.to_series().diff().median()
            if median_diff >= pd.Timedelta(hours=20):
                return df

        atr = _compute_atr(df, atr_period)
        features = _compute_pdl_features(df, atr, ma_period)

        if not features:
            return df

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    def get_signal_columns(self) -> List[str]:
        return [
            "pdl_above_high", "pdl_below_low",
            "pdl_high_break", "pdl_low_break",
            "pdl_day_range_expanding",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"atr_period": 14, "ma_period": 20}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR period for normalizing distances. PDH/PDL distances are expressed in ATR units.",
                "min": 2,
                "max": 100,
                "step": 1,
            },
            "ma_period": {
                "type": "int",
                "default": 20,
                "description": "Rolling window for position moving average. Smooths the intraday position within the previous day's range.",
                "min": 5,
                "max": 100,
                "step": 5,
            },
        }


__all__ = ["PreviousDayLevelsIndicator"]
