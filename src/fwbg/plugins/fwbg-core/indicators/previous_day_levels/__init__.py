"""
Previous Day Levels Indicator Plugin.

Computes features relative to previous day's high and low:
- Distance to PDH/PDL in ATR units
- Position within daily range
- Break detection (first cross above PDH or below PDL)
- Range expansion/contraction
- Retest signals at configurable retracement levels (rl{N}_ prefix)
- SL distance for orb_based exit strategy

candle_span controls which prices define the range:
- hl (default): H/L of candles (full candle including wicks)
- body: max(O,C)/min(O,C) of candles (body only, ignoring wicks)

range_scope controls which bars are included:
- session (default): only bars within session hours
- all: all 24h bars

Timeframe: Intraday only (M1-H4). On daily bars returns df unchanged.
"""
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, EPSILON, rl_tag
from fwbg_sdk.retest import apply_breakout_threshold, compute_break_state, compute_retest_signals

_CANDLE_SPAN_PREFIX = {
    "hl": "hl_",       # high/low range
    "body": "body_",   # body (O/C) range
}

_RANGE_SCOPE_PREFIX = {
    "session": "ses_",  # session hours only
    "all": "all_",      # all 24h bars
}

_BREAK_MODE_PREFIX = {
    "all_hours": "",           # default — breaks detected 24/7
    "session_only": "sesbrk_", # breaks only during session hours
}

_RETEST_MODE_PREFIX = {
    "all_hours": "",            # default — retests fire 24/7
    "session_only": "sesret_",  # retests only during session hours
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


def _compute_retest_features(
    pdh: np.ndarray,
    pdl_low: np.ndarray,
    pd_range: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    day_ids: np.ndarray,
    rl: float,
    enable_retest: bool,
    broke_high_arr: np.ndarray,
    broke_low_arr: np.ndarray,
    n: int,
    min_sl_atr_mult: float = 0.0,
    session_mask: np.ndarray = None,
    high: np.ndarray = None,
    low: np.ndarray = None,
    min_retracement: float = 0.3,
) -> Dict[str, np.ndarray]:
    """Compute retest signals and SL distance for a given retracement level.

    Uses shared compute_retest_signals for signal logic.  SL distance
    is PDHL-specific (range-based + ATR floor).
    """
    features: Dict[str, np.ndarray] = {}

    # Pre-compute entry levels
    entry_bull = pdh - rl * pd_range
    entry_bear = pdl_low + rl * pd_range

    if enable_retest:
        retest_result = compute_retest_signals(
            close=close,
            high=high,
            low=low,
            range_high=pdh,
            range_low=pdl_low,
            group_ids=day_ids,
            broke_high_arr=broke_high_arr,
            broke_low_arr=broke_low_arr,
            entry_bull=entry_bull,
            entry_bear=entry_bear,
            n=n,
            min_retracement=min_retracement,
            session_mask=session_mask,
        )
        features["pdl_retest_bull"] = retest_result["retest_bull"]
        features["pdl_retest_bear"] = retest_result["retest_bear"]

    # SL distance: entry to opposite boundary
    range_based_sl = (1 - rl) * pd_range
    if min_sl_atr_mult > 0:
        atr_floor = min_sl_atr_mult * atr
        features["pdl_sl_dist"] = np.maximum(range_based_sl, atr_floor)
    else:
        features["pdl_sl_dist"] = range_based_sl

    return features


def _get_range_columns(df, candle_span, session_mask, use_session_filter):
    """Get high/low source columns based on candle_span and scope."""
    if candle_span == "hl":
        highs = df["H"]
        lows = df["L"]
    else:  # body
        highs = df[["O", "C"]].max(axis=1)
        lows = df[["O", "C"]].min(axis=1)

    if use_session_filter:
        return highs.where(session_mask), lows.where(session_mask)
    return highs, lows


def _compute_range_variant(
    df: pd.DataFrame,
    atr: np.ndarray,
    session_mask: np.ndarray,
    session_mask_arr: np.ndarray,
    day_group: pd.DatetimeIndex,
    day_ids: np.ndarray,
    candle_span: str,
    scope: str,
    ma_period: int,
    enable_retest: bool,
    retracement_levels: Union[float, List[float]],
    skip_weekends: bool = True,
    session_break: bool = False,
    min_sl_atr_mult: float = 0.0,
    retest_modes: Union[List[str], Tuple[str, ...]] = ("all_hours",),
    resample_tf: str = None,
    min_retracement: float = 0.3,
    session_start_hour: int = None,
    session_end_hour: int = None,
    breakout_threshold: float = 0.0,
    breakout_threshold_abs: float = 0.0,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """Compute all PDL features for a single candle_span + scope + break mode.

    candle_span: 'hl' (H/L wicks) or 'body' (max/min of O/C)
    scope: 'session' (only session hours) or 'all' (24h)

    resample_tf (e.g. "1h"):
    - For body mode: PDH/PDL from resampled O/C body
    - For breakout: uses resampled Open/Close (hourly candle confirmation)
    - For hl mode: no effect on range (max of max = max)

    min_retracement: minimum fraction of pd_range price must retrace (via H/L)
    before retest signal fires.  0.3 = at least 30%.
    """
    features: Dict[str, Union[pd.Series, np.ndarray]] = {}
    n = len(df)
    use_session_filter = (scope == "session")

    # --- Range computation ---
    use_resampled_range = resample_tf and candle_span == "body"

    if use_resampled_range:
        # Resample O/C to target TF for body range computation
        r_df = df[["O", "C"]].resample(resample_tf).agg(
            {"O": "first", "C": "last"}
        ).dropna()
        body_high_r = r_df[["O", "C"]].max(axis=1)
        body_low_r = r_df[["O", "C"]].min(axis=1)

        # Session mask on resampled index
        if use_session_filter:
            r_hours = r_df.index.hour
            if session_start_hour < session_end_hour:
                r_session = (r_hours >= session_start_hour) & (r_hours < session_end_hour)
            else:
                r_session = (r_hours >= session_start_hour) | (r_hours < session_end_hour)
            range_highs_r = body_high_r.where(r_session)
            range_lows_r = body_low_r.where(r_session)
        else:
            range_highs_r = body_high_r
            range_lows_r = body_low_r

        # Day group on resampled index
        if session_start_hour is not None and session_start_hour >= session_end_hour:
            r_offset = pd.Timedelta(hours=24 - session_start_hour)
            r_day_group = (r_df.index + r_offset).normalize()
        else:
            r_day_group = r_df.index.normalize()

        day_hl = pd.DataFrame({
            "high": range_highs_r.groupby(r_day_group).max(),
            "low": range_lows_r.groupby(r_day_group).min(),
        })
    else:
        # Native-bar range
        range_highs, range_lows = _get_range_columns(
            df, candle_span, session_mask, use_session_filter,
        )

        day_hl = pd.DataFrame({
            "high": range_highs.groupby(day_group).max(),
            "low": range_lows.groupby(day_group).min(),
        })

    # --- Weekend skip + shift (common path) ---
    if skip_weekends:
        trading_day_hl = day_hl[day_hl.index.dayofweek < 5]
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

    # --- Current-day range (for pdl_day_range_expanding) ---
    cur_highs, cur_lows = _get_range_columns(
        df, candle_span, session_mask, use_session_filter,
    )
    day_high = cur_highs.groupby(day_group).transform("max")
    day_low = cur_lows.groupby(day_group).transform("min")

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

    # --- Breakout detection ---
    if resample_tf:
        # Resample to get hourly Open/Close for breakout confirmation
        r_df = df[["O", "C"]].resample(resample_tf).agg(
            {"O": "first", "C": "last"}
        ).dropna()

        # Day group on resampled index (for PDH/PDL mapping)
        if session_start_hour is not None and session_start_hour >= session_end_hour:
            r_offset = pd.Timedelta(hours=24 - session_start_hour)
            r_day_group = (r_df.index + r_offset).normalize()
        else:
            r_day_group = r_df.index.normalize()

        r_pdh = prev_day_hl["high"].reindex(r_day_group).values
        r_pdl = prev_day_hl["low"].reindex(r_day_group).values
        r_range = r_pdh - r_pdl

        # Apply breakout threshold to resampled data
        offset = np.maximum(breakout_threshold * r_range, breakout_threshold_abs)
        r_above = (r_df["O"].values > r_pdh + offset) | (r_df["C"].values > r_pdh + offset)
        r_below = (r_df["O"].values < r_pdl - offset) | (r_df["C"].values < r_pdl - offset)

        # Map back to original bars at the LAST bar of each resampled period
        # to avoid lookahead (hourly Close isn't known until the hour ends).
        bar_duration = df.index[1] - df.index[0] if len(df) > 1 else pd.Timedelta(0)
        end_offset = pd.Timedelta(resample_tf) - bar_duration
        above = pd.Series(
            r_above, index=r_df.index + end_offset,
        ).reindex(df.index, method="ffill").fillna(False).values.astype(bool)
        below = pd.Series(
            r_below, index=r_df.index + end_offset,
        ).reindex(df.index, method="ffill").fillna(False).values.astype(bool)

        # Ensure NaN PDH bars don't trigger breakout
        nan_mask = np.isnan(pdh)
        above[nan_mask] = False
        below[nan_mask] = False
    else:
        above, below = apply_breakout_threshold(
            close, pdh, pdl_low, pd_range,
            breakout_threshold, breakout_threshold_abs,
        )

    # Break state
    break_mask = session_mask_arr if session_break else None
    high_break, low_break, post_bull, post_bear, broke_high_arr, broke_low_arr = \
        compute_break_state(above, below, pdh, day_ids, n, break_mask)

    features["pdl_high_break"] = high_break
    features["pdl_low_break"] = low_break
    features["pdl_post_bull"] = post_bull
    features["pdl_post_bear"] = post_bear

    # Retest signals + SL distance per retracement level (always prefixed)
    rl_list = [retracement_levels] if isinstance(retracement_levels, (int, float)) else list(retracement_levels)

    high_arr = df["H"].values
    low_arr = df["L"].values

    for rl in rl_list:
        rl_pfx = f"{rl_tag(rl)}_"
        for retest_mode in retest_modes:
            retest_pfx = _RETEST_MODE_PREFIX[retest_mode]
            retest_mask = session_mask_arr if retest_mode == "session_only" else None
            retest_feats = _compute_retest_features(
                pdh, pdl_low, pd_range, close, atr, day_ids,
                rl, enable_retest,
                broke_high_arr, broke_low_arr, n,
                min_sl_atr_mult=min_sl_atr_mult,
                session_mask=retest_mask,
                high=high_arr,
                low=low_arr,
                min_retracement=min_retracement,
            )
            for k, v in retest_feats.items():
                features[f"{retest_pfx}{rl_pfx}{k}"] = v

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
    retracement_levels: Union[float, List[float]] = 0.5,
    session_start_hour: int = 7,
    session_end_hour: int = 21,
    candle_span: str = "hl",
    range_scope: Union[List[str], Tuple[str, ...]] = ("session",),
    break_modes: Union[List[str], Tuple[str, ...]] = ("all_hours",),
    retest_modes: Union[List[str], Tuple[str, ...]] = ("all_hours",),
    skip_weekends: bool = True,
    min_sl_atr_mult: float = 0.0,
    resample_tf: str = None,
    min_retracement: float = 0.3,
    breakout_threshold: float = 0.0,
    breakout_threshold_abs: float = 0.0,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """Compute all previous day level features for all scope × break × retest modes."""
    all_features: Dict[str, Union[pd.Series, np.ndarray]] = {}

    # Group by trading day
    if session_start_hour < session_end_hour:
        day_group = df.index.normalize()
    else:
        offset = pd.Timedelta(hours=24 - session_start_hour)
        day_group = (df.index + offset).normalize()

    # Session mask: only bars within trading hours
    hours = df.index.hour
    if session_start_hour < session_end_hour:
        session_mask = (hours >= session_start_hour) & (hours < session_end_hour)
    else:
        session_mask = (hours >= session_start_hour) | (hours < session_end_hour)

    session_mask_arr = np.array(session_mask)
    day_ids = pd.Series(day_group).factorize()[0]

    span_pfx = _CANDLE_SPAN_PREFIX[candle_span]

    invalid_scopes = [s for s in range_scope if s not in _RANGE_SCOPE_PREFIX]
    if invalid_scopes:
        raise ValueError(
            f"Invalid range_scope value(s): {invalid_scopes}. "
            f"Valid options: {sorted(_RANGE_SCOPE_PREFIX)}"
        )

    for scope in range_scope:
        scope_pfx = _RANGE_SCOPE_PREFIX[scope]
        for break_mode in break_modes:
            break_pfx = _BREAK_MODE_PREFIX[break_mode]
            combined_pfx = f"{span_pfx}{scope_pfx}{break_pfx}"
            mode_feats = _compute_range_variant(
                df, atr, session_mask, session_mask_arr, day_group, day_ids,
                candle_span, scope,
                ma_period, enable_retest, retracement_levels,
                skip_weekends, session_break=(break_mode == "session_only"),
                min_sl_atr_mult=min_sl_atr_mult,
                retest_modes=retest_modes,
                resample_tf=resample_tf,
                min_retracement=min_retracement,
                session_start_hour=session_start_hour,
                session_end_hour=session_end_hour,
                breakout_threshold=breakout_threshold,
                breakout_threshold_abs=breakout_threshold_abs,
            )
            for k, v in mode_feats.items():
                all_features[f"{combined_pfx}{k}"] = v

    return all_features


@register_indicator("previous_day_levels")
class PreviousDayLevelsIndicator(BaseIndicator):
    """Previous Day High/Low features for intraday trading."""

    name = "previous_day_levels"
    version = "1.0.0"
    benefits_from_stationary = False
    group = "session"

    PDL_SIGNAL_SUFFIXES = (
        "_above_high", "_below_low", "_high_break", "_low_break",
        "_post_bull", "_post_bear", "_retest_bull", "_retest_bear",
        "_day_range_expanding",
    )

    # Base features per (candle_span, range_scope, break/retest_mode) variant — no rl prefix
    _BASE_FEATURES = [
        "pdl_high_dist", "pdl_low_dist", "pdl_position", "pdl_range_vs_atr",
        "pdl_above_high", "pdl_below_low", "pdl_midpoint_dist",
        "pdl_high_break", "pdl_low_break", "pdl_post_bull", "pdl_post_bear",
        "pdl_range_position_ma", "pdl_day_range_expanding",
    ]
    # rl-dependent features (appended per retracement level)
    _RL_FEATURES = ["pdl_retest_bull", "pdl_retest_bear", "pdl_sl_dist"]
    # Signals (subset of base + rl features)
    _BASE_SIGNAL_SUFFIXES = (
        "pdl_above_high", "pdl_below_low", "pdl_high_break", "pdl_low_break",
        "pdl_post_bull", "pdl_post_bear", "pdl_day_range_expanding",
    )
    _RL_SIGNAL_SUFFIXES = ("pdl_retest_bull", "pdl_retest_bear")

    def compute(
        self,
        df: pd.DataFrame,
        atr_period: int = 14,
        ma_period: int = 20,
        enable_retest: bool = True,
        retracement_levels: Union[float, List[float]] = 0.5,
        session_start_hour: int = 7,
        session_end_hour: int = 21,
        candle_span: str = "hl",
        range_scope: Union[List[str], Tuple[str, ...]] = ("session",),
        break_modes: Union[List[str], Tuple[str, ...]] = ("all_hours",),
        retest_modes: Union[List[str], Tuple[str, ...]] = ("all_hours",),
        skip_weekends: bool = True,
        min_sl_atr_mult: float = 0.0,
        resample_tf: str = None,
        min_retracement: float = 0.3,
        breakout_threshold: float = 0.0,
        breakout_threshold_abs: float = 0.0,
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
            df, atr, ma_period, enable_retest, retracement_levels,
            session_start_hour, session_end_hour, candle_span, range_scope,
            break_modes, retest_modes, skip_weekends, min_sl_atr_mult,
            resample_tf=resample_tf, min_retracement=min_retracement,
            breakout_threshold=breakout_threshold,
            breakout_threshold_abs=breakout_threshold_abs,
        )

        if not features:
            return df

        self._feature_columns = list(features.keys())
        self._signal_columns = [
            k for k in features if any(k.endswith(s) for s in self.PDL_SIGNAL_SUFFIXES)
        ]

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    @classmethod
    def _default_fallback_columns(cls, params=None) -> Tuple[List[str], List[str]]:
        """Build feature/signal column lists from params (or defaults)."""
        d = {**cls.get_default_params(), **(params or {})}
        rl_vals = d["retracement_levels"]
        rl_list = [rl_vals] if isinstance(rl_vals, (int, float)) else list(rl_vals)

        span_pfx = _CANDLE_SPAN_PREFIX[d["candle_span"]]
        range_scope = d["range_scope"] if isinstance(d["range_scope"], (list, tuple)) else [d["range_scope"]]
        break_modes = d["break_modes"] if isinstance(d["break_modes"], (list, tuple)) else [d["break_modes"]]
        retest_modes = d["retest_modes"] if isinstance(d["retest_modes"], (list, tuple)) else [d["retest_modes"]]

        invalid_scopes = [s for s in range_scope if s not in _RANGE_SCOPE_PREFIX]
        if invalid_scopes:
            raise ValueError(
                f"Invalid range_scope value(s): {invalid_scopes}. "
                f"Valid options: {sorted(_RANGE_SCOPE_PREFIX)}"
            )

        features = []
        signals = []

        for scope in range_scope:
            scope_pfx = _RANGE_SCOPE_PREFIX[scope]
            for break_mode in break_modes:
                break_pfx = _BREAK_MODE_PREFIX[break_mode]
                combined_pfx = f"{span_pfx}{scope_pfx}{break_pfx}"

                for f in cls._BASE_FEATURES:
                    features.append(f"{combined_pfx}{f}")
                for f in cls._BASE_SIGNAL_SUFFIXES:
                    signals.append(f"{combined_pfx}{f}")

                for rl in rl_list:
                    rl_pfx_str = f"{rl_tag(rl)}_"
                    for retest_mode in retest_modes:
                        retest_pfx = _RETEST_MODE_PREFIX[retest_mode]
                        full_pfx = f"{combined_pfx}{retest_pfx}{rl_pfx_str}"
                        for f in cls._RL_FEATURES:
                            features.append(f"{full_pfx}{f}")
                        for f in cls._RL_SIGNAL_SUFFIXES:
                            signals.append(f"{full_pfx}{f}")

        return features, signals

    def get_feature_columns(self, params=None) -> List[str]:
        if self._feature_columns and not params:
            return self._feature_columns
        return self._default_fallback_columns(params)[0]

    def get_signal_columns(self, params=None) -> List[str]:
        if self._signal_columns and not params:
            return self._signal_columns
        return self._default_fallback_columns(params)[1]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "atr_period": 14,
            "ma_period": 20,
            "enable_retest": True,
            "retracement_levels": 0.5,
            "session_start_hour": 7,
            "session_end_hour": 21,
            "candle_span": "hl",
            "range_scope": ["session"],
            "break_modes": ["all_hours"],
            "retest_modes": ["all_hours"],
            "skip_weekends": True,
            "min_sl_atr_mult": 0.0,
            "resample_tf": None,
            "min_retracement": 0.3,
            "breakout_threshold": 0.0,
            "breakout_threshold_abs": 0.0,
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
            "retracement_levels": {
                "type": "list[float]",
                "default": 0.5,
                "description": (
                    "Retracement fraction(s) of the PDH-PDL range for entry level. "
                    "0.5 = midpoint. 0 = at boundary. 0.382 = shallow retrace. "
                    "Always generates rl{N}_ prefixed columns (e.g. hl_ses_rl50_pdl_retest_bull). "
                    "Pass a list to precompute multiple variants for grid search."
                ),
                "min": 0.0,
                "max": 0.9,
                "step": 0.1,
            },
            "session_start_hour": {
                "type": "int",
                "default": 7,
                "description": "UTC hour when trading session starts.",
                "min": 0,
                "max": 23,
                "step": 1,
            },
            "session_end_hour": {
                "type": "int",
                "default": 21,
                "description": "UTC hour when trading session ends.",
                "min": 1,
                "max": 24,
                "step": 1,
            },
            "candle_span": {
                "type": "choice",
                "default": "hl",
                "description": (
                    "Vertical extent of candles used for range. "
                    "'hl': full candle including wicks (H/L). "
                    "'body': candle body only (max/min of O/C, ignoring wicks)."
                ),
                "choices": ["hl", "body"],
            },
            "range_scope": {
                "type": "list[string]",
                "default": ["session"],
                "description": (
                    "Which bars to include for range computation. "
                    "'session' (ses_ prefix): only bars within session hours. "
                    "'all' (all_ prefix): all 24h bars. "
                    "Pass a list to precompute both for grid search."
                ),
                "options": ["session", "all"],
            },
            "break_modes": {
                "type": "list[string]",
                "default": ["all_hours"],
                "description": (
                    "Break detection timing modes. all_hours (default): "
                    "breakouts detected 24/7. "
                    "session_only (sesbrk_ prefix): only session bars trigger breakouts."
                ),
                "options": ["all_hours", "session_only"],
            },
            "retest_modes": {
                "type": "list[string]",
                "default": ["all_hours"],
                "description": (
                    "Retest signal timing modes. "
                    "all_hours (default): retest signals fire 24/7. "
                    "session_only (sesret_ prefix): retest signals only during session hours."
                ),
                "options": ["all_hours", "session_only"],
            },
            "skip_weekends": {
                "type": "bool",
                "default": True,
                "description": (
                    "Skip Saturday/Sunday when computing previous day levels. "
                    "Monday uses Friday's range. Set False for 24/7 markets (crypto)."
                ),
            },
            "min_sl_atr_mult": {
                "type": "float",
                "default": 0.0,
                "description": (
                    "Minimum SL distance as a multiple of ATR. "
                    "0 = no floor (pure range-based). "
                    "E.g., 1.0 means SL is at least 1x ATR."
                ),
                "min": 0.0,
                "max": 5.0,
                "step": 0.5,
            },
            "resample_tf": {
                "type": "string",
                "default": None,
                "description": (
                    "Resample timeframe for body range computation and breakout "
                    "detection. Uses resampled O/C for body range and breakout "
                    "confirmation (no spikes). None = use native bar resolution."
                ),
                "options": [None, "1h", "4h"],
            },
            "min_retracement": {
                "type": "float",
                "default": 0.3,
                "description": (
                    "Minimum retracement of previous day range (checked via H/L) before "
                    "retest signal fires. 0.3 = at least 30% retracement. 0 = disabled."
                ),
                "min": 0.0,
                "max": 0.9,
                "step": 0.1,
            },
            "breakout_threshold": {
                "type": "float",
                "default": 0.0,
                "description": (
                    "Minimum distance as fraction of range for breakout. "
                    "E.g. 0.05 = close must be at least 5% of range beyond boundary. "
                    "0 = disabled."
                ),
                "min": 0.0,
                "max": 0.5,
                "step": 0.01,
            },
            "breakout_threshold_abs": {
                "type": "float",
                "default": 0.0,
                "description": (
                    "Minimum distance in absolute terms (pips/points) for breakout. "
                    "Effective threshold = max(pct * range, abs). 0 = disabled."
                ),
                "min": 0.0,
                "max": 100.0,
                "step": 1.0,
            },
        }


__all__ = ["PreviousDayLevelsIndicator"]
