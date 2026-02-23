"""
Previous Day Levels Indicator Plugin.

Computes features relative to previous day's high and low:
- Distance to PDH/PDL in ATR units
- Position within daily range
- Break detection (first cross above PDH or below PDL)
- Range expansion/contraction
- Retest signals at configurable retracement levels (rl{N}_ prefix)
- SL distance for orb_based exit strategy

Range modes (grid-searchable):
- hl_session: H/L during session hours (default)
- hl_all:     H/L for all 24h bars
- close_session: Close during session hours
- close_all:     Close for all 24h bars

Timeframe: Intraday only (M1-H4). On daily bars returns df unchanged.
"""
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, EPSILON

_RANGE_MODE_PREFIX = {
    "hl_session": "",       # default, backward compatible
    "hl_all": "ha_",        # H/L all-hours
    "close_session": "cs_", # Close session-filtered
    "close_all": "ca_",     # Close all-hours
}


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


def _compute_break_state(
    close: np.ndarray,
    pdh: np.ndarray,
    pdl_low: np.ndarray,
    day_ids: np.ndarray,
    n: int,
    session_mask: np.ndarray = None,
) -> tuple:
    """Compute break detection and post-breakout state arrays.

    Only bars within the trading session (session_mask=True) can trigger
    breakouts. Off-session bars inherit the current state but cannot set it.

    Returns (high_break, low_break, post_bull, post_bear,
             broke_high_arr, broke_low_arr).
    """
    above = close > pdh
    below = close < pdl_low

    high_break = np.zeros(n)
    low_break = np.zeros(n)
    post_bull = np.zeros(n)
    post_bear = np.zeros(n)
    broke_high_arr = np.zeros(n, dtype=bool)
    broke_low_arr = np.zeros(n, dtype=bool)

    prev_day_id = -1
    broke_high = False
    broke_low = False

    for i in range(n):
        if day_ids[i] != prev_day_id:
            prev_day_id = day_ids[i]
            broke_high = False
            broke_low = False

        if np.isnan(pdh[i]):
            continue

        # Only detect breakouts during trading session
        in_session = session_mask[i] if session_mask is not None else True
        if in_session:
            if above[i] and not broke_high:
                high_break[i] = 1.0
                broke_high = True
            if below[i] and not broke_low:
                low_break[i] = 1.0
                broke_low = True

        broke_high_arr[i] = broke_high
        broke_low_arr[i] = broke_low

        if broke_high:
            post_bull[i] = 1.0
        if broke_low:
            post_bear[i] = 1.0

    return high_break, low_break, post_bull, post_bear, broke_high_arr, broke_low_arr


def _compute_retest_features(
    pdh: np.ndarray,
    pdl_low: np.ndarray,
    pd_range: np.ndarray,
    close: np.ndarray,
    day_ids: np.ndarray,
    retest_atr_width: float,
    rl: float,
    enable_retest: bool,
    broke_high_arr: np.ndarray,
    broke_low_arr: np.ndarray,
    n: int,
    session_mask: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Compute retest signals and SL distance for a given retracement level.

    Entry levels per rl (fraction of range from breakout boundary):
    - Bull entry = PDH - rl * pd_range (retrace from PDH downward)
    - Bear entry = PDL + rl * pd_range (retrace from PDL upward)

    SL distance = (1 - rl) * pd_range (distance from entry to opposite boundary).

    Retest signals only fire during trading session (session_mask=True).
    """
    features: Dict[str, np.ndarray] = {}

    if enable_retest:
        entry_bull = pdh - rl * pd_range
        entry_bear = pdl_low + rl * pd_range

        half_band = retest_atr_width * np.nan_to_num(pd_range, nan=0.0) / 2
        near_entry_bull = (close >= entry_bull - half_band) & (close <= entry_bull + half_band)
        near_entry_bear = (close >= entry_bear - half_band) & (close <= entry_bear + half_band)

        retest_bull = np.zeros(n)
        retest_bear = np.zeros(n)

        prev_day_id = -1
        retested_bull = False
        retested_bear = False

        for i in range(n):
            if day_ids[i] != prev_day_id:
                prev_day_id = day_ids[i]
                retested_bull = False
                retested_bear = False

            if np.isnan(pdh[i]):
                continue

            # Only fire retest signals during trading session
            in_session = session_mask[i] if session_mask is not None else True
            if not in_session:
                continue

            if broke_high_arr[i] and not retested_bull and near_entry_bull[i] and close[i] > pdl_low[i]:
                retest_bull[i] = 1.0
                retested_bull = True
            if broke_low_arr[i] and not retested_bear and near_entry_bear[i] and close[i] < pdh[i]:
                retest_bear[i] = 1.0
                retested_bear = True

        features["pdl_retest_bull"] = retest_bull
        features["pdl_retest_bear"] = retest_bear

    features["pdl_sl_dist"] = (1 - rl) * pd_range

    return features


def _compute_range_variant(
    df: pd.DataFrame,
    atr: np.ndarray,
    session_mask: np.ndarray,
    session_mask_arr: np.ndarray,
    day_group: pd.DatetimeIndex,
    day_ids: np.ndarray,
    mode: str,
    ma_period: int,
    enable_retest: bool,
    retest_atr_width: float,
    retracement_levels: Union[float, List[float]],
    skip_weekends: bool = True,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """Compute all PDL features for a single range mode.

    Modes control how PDH/PDL is derived:
    - hl_session:    H/L during session hours
    - hl_all:        H/L for all 24h bars
    - close_session: Close during session hours
    - close_all:     Close for all 24h bars

    Break detection and retest timing always use session_mask.
    When skip_weekends=True, Saturday/Sunday are excluded from PDH/PDL
    computation — Monday uses Friday's levels.
    """
    features: Dict[str, Union[pd.Series, np.ndarray]] = {}
    n = len(df)

    # Select price series for range computation based on mode
    if mode == "hl_session":
        range_highs = df["H"].where(session_mask)
        range_lows = df["L"].where(session_mask)
    elif mode == "hl_all":
        range_highs = df["H"]
        range_lows = df["L"]
    elif mode == "close_session":
        range_highs = df["C"].where(session_mask)
        range_lows = df["C"].where(session_mask)
    elif mode == "close_all":
        range_highs = df["C"]
        range_lows = df["C"]
    else:
        raise ValueError(f"Unknown range mode: {mode}")

    # Per-day high/low
    day_high = range_highs.groupby(day_group).transform("max")
    day_low = range_lows.groupby(day_group).transform("min")

    # Shift by 1 day to get PREVIOUS day high/low
    day_hl = pd.DataFrame({
        "high": range_highs.groupby(day_group).max(),
        "low": range_lows.groupby(day_group).min(),
    })

    if skip_weekends:
        # Filter to weekdays only (Mon=0..Fri=4), so Monday looks back to Friday
        trading_day_hl = day_hl[day_hl.index.dayofweek < 5]
        # Forward-fill trading day values onto weekend dates
        all_days_hl = trading_day_hl.reindex(day_hl.index, method="ffill")
        prev_day_hl = all_days_hl.shift(1)
    else:
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

    # Midpoint of PDH-PDL range
    midpoint = (pdh + pdl_low) / 2
    features["pdl_midpoint_dist"] = (close - midpoint) / safe_atr

    # Break detection + post-breakout state (NOT rl-dependent)
    # Break detection always uses session_mask regardless of range mode
    high_break, low_break, post_bull, post_bear, broke_high_arr, broke_low_arr = \
        _compute_break_state(close, pdh, pdl_low, day_ids, n, session_mask_arr)

    features["pdl_high_break"] = high_break
    features["pdl_low_break"] = low_break
    features["pdl_post_bull"] = post_bull
    features["pdl_post_bear"] = post_bear

    # Retest signals + SL distance per retracement level (always prefixed)
    rl_list = [retracement_levels] if isinstance(retracement_levels, (int, float)) else list(retracement_levels)

    for rl in rl_list:
        rl_pfx = f"rl{int(rl * 100)}_"
        retest_feats = _compute_retest_features(
            pdh, pdl_low, pd_range, close, day_ids,
            retest_atr_width, rl, enable_retest,
            broke_high_arr, broke_low_arr, n, session_mask_arr,
        )
        for k, v in retest_feats.items():
            features[f"{rl_pfx}{k}"] = v

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


def _compute_pdl_features(
    df: pd.DataFrame,
    atr: np.ndarray,
    ma_period: int,
    enable_retest: bool = True,
    retest_atr_width: float = 0.3,
    retracement_levels: Union[float, List[float]] = 0.5,
    session_start_hour: int = 7,
    session_end_hour: int = 21,
    range_modes: Union[List[str], Tuple[str, ...]] = ("hl_session",),
    skip_weekends: bool = True,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """Compute all previous day level features for all range modes."""
    all_features: Dict[str, Union[pd.Series, np.ndarray]] = {}

    # Group by trading day
    day_group = df.index.normalize()

    # Session mask: only bars within trading hours
    hours = df.index.hour
    if session_start_hour < session_end_hour:
        session_mask = (hours >= session_start_hour) & (hours < session_end_hour)
    else:
        # Crosses midnight (e.g., ASX200: 23-06 UTC)
        session_mask = (hours >= session_start_hour) | (hours < session_end_hour)

    session_mask_arr = np.array(session_mask)
    day_ids = pd.Series(day_group).factorize()[0]

    for mode in range_modes:
        pfx = _RANGE_MODE_PREFIX[mode]
        mode_feats = _compute_range_variant(
            df, atr, session_mask, session_mask_arr, day_group, day_ids, mode,
            ma_period, enable_retest, retest_atr_width, retracement_levels,
            skip_weekends,
        )
        for k, v in mode_feats.items():
            all_features[f"{pfx}{k}"] = v

    return all_features


@register_indicator("previous_day_levels")
class PreviousDayLevelsIndicator(BaseIndicator):
    """Previous Day High/Low features for intraday trading."""

    name = "previous_day_levels"
    version = "1.0.0"
    benefits_from_stationary = False
    group = "session"

    # Representative feature list for default hl_session mode, rl=50 (0.5)
    _FEATURES = [
        "pdl_high_dist",
        "pdl_low_dist",
        "pdl_position",
        "pdl_range_vs_atr",
        "pdl_above_high",
        "pdl_below_low",
        "pdl_midpoint_dist",
        "pdl_high_break",
        "pdl_low_break",
        "pdl_post_bull",
        "pdl_post_bear",
        "rl50_pdl_retest_bull",
        "rl50_pdl_retest_bear",
        "rl50_pdl_sl_dist",
        "pdl_range_position_ma",
        "pdl_day_range_expanding",
    ]

    def compute(
        self,
        df: pd.DataFrame,
        atr_period: int = 14,
        ma_period: int = 20,
        enable_retest: bool = True,
        retest_atr_width: float = 0.3,
        retracement_levels: Union[float, List[float]] = 0.5,
        session_start_hour: int = 7,
        session_end_hour: int = 21,
        range_modes: Union[List[str], Tuple[str, ...]] = ("hl_session",),
        skip_weekends: bool = True,
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
        features = _compute_pdl_features(
            df, atr, ma_period, enable_retest, retest_atr_width, retracement_levels,
            session_start_hour, session_end_hour, range_modes, skip_weekends,
        )

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
            "pdl_post_bull", "pdl_post_bear",
            "rl50_pdl_retest_bull", "rl50_pdl_retest_bear",
            "pdl_day_range_expanding",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "atr_period": 14,
            "ma_period": 20,
            "enable_retest": True,
            "retest_atr_width": 0.3,
            "retracement_levels": 0.5,
            "session_start_hour": 7,
            "session_end_hour": 21,
            "range_modes": ["hl_session"],
            "skip_weekends": True,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR period for normalizing distances.",
                "min": 2,
                "max": 100,
                "step": 1,
            },
            "ma_period": {
                "type": "int",
                "default": 20,
                "description": "Rolling window for position moving average.",
                "min": 5,
                "max": 100,
                "step": 5,
            },
            "enable_retest": {
                "type": "bool",
                "default": True,
                "description": "Enable PDH/PDL retest signals at the configured retracement level(s).",
            },
            "retest_atr_width": {
                "type": "float",
                "default": 0.3,
                "description": "Retest zone width as fraction of half-range around entry level.",
                "min": 0.1,
                "max": 0.8,
                "step": 0.1,
            },
            "retracement_levels": {
                "type": "list[float]",
                "default": 0.5,
                "description": (
                    "Retracement fraction(s) of the PDH-PDL range for entry level. "
                    "0.5 = midpoint. 0.382 = shallow retrace. 0.7 = deep retrace. "
                    "Always generates rl{N}_ prefixed columns (e.g. rl50_pdl_retest_bull). "
                    "Pass a list to precompute multiple variants for grid search."
                ),
                "min": 0.1,
                "max": 0.9,
                "step": 0.1,
            },
            "session_start_hour": {
                "type": "int",
                "default": 7,
                "description": "UTC hour when trading session starts. Only bars within [start, end) are used for PDH/PDL computation in session modes.",
                "min": 0,
                "max": 23,
                "step": 1,
            },
            "session_end_hour": {
                "type": "int",
                "default": 21,
                "description": "UTC hour when trading session ends. Only bars within [start, end) are used for PDH/PDL computation in session modes.",
                "min": 1,
                "max": 24,
                "step": 1,
            },
            "range_modes": {
                "type": "list[string]",
                "default": ["hl_session"],
                "description": (
                    "Range computation modes to precompute. Each mode produces a full "
                    "feature set with a prefix: hl_session (default, no prefix), "
                    "hl_all (ha_ prefix), close_session (cs_ prefix), close_all (ca_ prefix). "
                    "Use model_hyperparameters_grid to grid-search across modes."
                ),
                "options": ["hl_session", "hl_all", "close_session", "close_all"],
            },
            "skip_weekends": {
                "type": "bool",
                "default": True,
                "description": (
                    "Skip Saturday/Sunday when computing previous day levels. "
                    "Monday uses Friday's range. Set False for 24/7 markets (crypto)."
                ),
            },
        }


__all__ = ["PreviousDayLevelsIndicator"]
