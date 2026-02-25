"""
Opening Range Breakout (ORB) Indicator Plugin.

Berechnet Features basierend auf dem Opening Range Konzept:
- Session ORB: Range/Position/Breakout für konfigurierbare Session-Stunden (forward-filled)
- Retest-Signale bei konfigurierbaren Retracement-Levels (rl{N}_ Prefix)

Timeframe-Kompatibilität: Intraday (M1-H4). Auf DAY-Bars → NaN.

range_bars akzeptiert int oder List[int]. Bei einer Liste werden für jeden
Wert eigene Feature-Spalten mit Präfix rb{n}_ generiert, z.B. rb1_orb_range
und rb2_orb_range. Das erlaubt dem ML-Modell, beide Varianten zu vergleichen.
"""
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide
from fwbg_sdk.retest import apply_breakout_threshold, compute_break_state, compute_retest_signals


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR via ta-lib, returns raw ATR (not percent)."""
    return ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=period)


def _session_orb_features(
    df: pd.DataFrame,
    sessions: List[int],
    range_bars: int,
    atr: pd.Series,
    enable_retracement: bool = False,
    retest_atr_width: float = 0.3,
    retest_zone_width: float = 0.5,
    carry_forward_days: int = 0,
    pre_range_bars: int = 0,
    candle_span: str = "hl",
    breakout_threshold: float = 0.0,
    breakout_threshold_abs: float = 0.0,
    retracement_levels: Union[float, List[float]] = 0.5,
    min_retracement: float = 0.0,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """
    Compute session-specific ORB features.

    For each configured session hour, compute the opening range at that hour
    and forward-fill until the next occurrence of that session hour.

    carry_forward_days: if no breakout occurs during a session, carry the range
        forward to the next N session occurrences. 0 = no carry (default).
    pre_range_bars: include N bars before the session start in the range
        calculation, expanding the opening range window backward. 0 = disabled.
    """
    features: Dict[str, Union[pd.Series, np.ndarray]] = {}

    hours = pd.Series(df.index.hour, index=df.index)
    prev_hour = hours.shift(1)

    for session_hour in sessions:
        prefix = f"orb_s{session_hour:02d}"

        # Boolean Series: is this bar within the session hour?
        is_session = hours == session_hour

        # Session start: first bar of a new session_hour block
        session_start = is_session & (prev_hour != session_hour)

        # Session ID: increments at each session start, groups all bars until next start
        session_id = session_start.cumsum()

        # Count bars within session hour per session_id (0-indexed)
        bar_in_block = is_session.astype(int).groupby(session_id).cumsum() - 1

        # Opening range computation
        if candle_span == "hl":
            # H/L of all bars in the range window
            range_mask = is_session & (bar_in_block < range_bars)
            or_high = df["H"].where(range_mask).groupby(session_id).transform("max")
            or_low = df["L"].where(range_mask).groupby(session_id).transform("min")
        else:
            # Body: O of first bar, C of last bar
            or_mask_first = is_session & (bar_in_block == 0)
            or_mask_last = is_session & (bar_in_block == range_bars - 1)
            or_open = df["O"].where(or_mask_first).groupby(session_id).transform("first")
            or_close = df["C"].where(or_mask_last).groupby(session_id).transform("last")
            or_high = pd.Series(np.maximum(or_open.values, or_close.values), index=df.index)
            or_low = pd.Series(np.minimum(or_open.values, or_close.values), index=df.index)

        # --- pre_range_bars: expand range to include pre-session bars ---
        if pre_range_bars > 0:
            start_positions = np.where(session_start.values)[0]
            arr_h = or_high.values.copy()
            arr_l = or_low.values.copy()
            sid_vals = session_id.values

            for pos in start_positions:
                sid = sid_vals[pos]
                if sid == 0:
                    continue
                pre_start = max(0, pos - pre_range_bars)
                if pre_start < pos:
                    if candle_span == "hl":
                        pre_h = df["H"].values[pre_start:pos].max()
                        pre_l = df["L"].values[pre_start:pos].min()
                    else:
                        pre_o = df["O"].values[pre_start:pos]
                        pre_c = df["C"].values[pre_start:pos]
                        pre_h = max(pre_o.max(), pre_c.max())
                        pre_l = min(pre_o.min(), pre_c.min())
                    sess_mask = sid_vals == sid
                    cur_h = arr_h[sess_mask][0]
                    cur_l = arr_l[sess_mask][0]
                    if not np.isnan(cur_h):
                        arr_h[sess_mask] = max(cur_h, pre_h)
                        arr_l[sess_mask] = min(cur_l, pre_l)

            or_high = pd.Series(arr_h, index=df.index)
            or_low = pd.Series(arr_l, index=df.index)

        # --- carry_forward_days: carry range from no-breakout sessions ---
        carried_session_ids = set()
        if carry_forward_days > 0:
            unique_sids = sorted(session_id.unique())
            arr_h = or_high.values.copy()
            arr_l = or_low.values.copy()
            sid_vals = session_id.values
            close_vals = df["C"].values

            carry_remaining = 0
            carry_h = None
            carry_l = None

            for sid in unique_sids:
                if sid == 0:
                    continue

                sess_mask = sid_vals == sid

                if carry_remaining > 0:
                    # Use carried range instead of this session's own range
                    arr_h[sess_mask] = carry_h
                    arr_l[sess_mask] = carry_l
                    carried_session_ids.add(sid)

                    # Check breakout against carried range
                    sess_close = close_vals[sess_mask]
                    had_breakout = (sess_close > carry_h).any() or (sess_close < carry_l).any()

                    if had_breakout:
                        carry_remaining = 0
                    else:
                        carry_remaining -= 1
                else:
                    # Use own range, check for breakout
                    own_h = arr_h[sess_mask][0]
                    own_l = arr_l[sess_mask][0]

                    if not np.isnan(own_h):
                        sess_close = close_vals[sess_mask]
                        had_breakout = (sess_close > own_h).any() or (sess_close < own_l).any()

                        if not had_breakout:
                            carry_remaining = carry_forward_days
                            carry_h = own_h
                            carry_l = own_l

            or_high = pd.Series(arr_h, index=df.index)
            or_low = pd.Series(arr_l, index=df.index)

        # Valid: after first session, not within the opening range bars themselves
        # For carried sessions, range is pre-established → no range period masking
        in_range_period = is_session & (bar_in_block < range_bars)
        if carried_session_ids:
            is_carried = session_id.isin(carried_session_ids)
            in_range_period = in_range_period & ~is_carried
        valid = (session_id > 0) & ~in_range_period

        or_range = or_high - or_low
        or_midpoint = (or_high + or_low) / 2

        features[f"{prefix}_range"] = safe_divide(or_range, df["C"]).where(valid, np.nan)
        features[f"{prefix}_position"] = safe_divide(
            df["C"] - or_low, or_range
        ).where(valid, np.nan)

        # Event feature: 1 only on the first bar crossing above/below per session.
        sess_above_int = (df["C"] > or_high).astype(np.int8)
        sess_below_int = (df["C"] < or_low).astype(np.int8)
        sess_prev_above = sess_above_int.groupby(session_id).shift(1).fillna(0).astype(np.int8)
        sess_prev_below = sess_below_int.groupby(session_id).shift(1).fillna(0).astype(np.int8)
        features[f"{prefix}_breakout_up"] = (
            (sess_above_int - sess_prev_above).clip(lower=0).where(valid, np.nan)
        )
        features[f"{prefix}_breakout_down"] = (
            (sess_below_int - sess_prev_below).clip(lower=0).where(valid, np.nan)
        )

        features[f"{prefix}_range_vs_atr"] = safe_divide(or_range, atr).where(
            valid, np.nan
        )

        # POC proxy: normalized distance from close to ORB midpoint
        features[f"{prefix}_poc_dist"] = safe_divide(
            df["C"] - or_midpoint, atr
        ).where(valid, np.nan)

        # SL distance: half body range (entry at midpoint → SL at body boundary)
        # When body range is 0 (O==C doji), sl_dist is invalid → NaN
        sl_dist = (or_range / 2).where(valid & (or_range > 0), np.nan)
        features[f"{prefix}_sl_dist"] = sl_dist

        # Post-breakout state via shared break detection (with threshold).
        # Only count breakouts on valid bars (after range period).
        close_vals = df["C"].values
        or_high_vals = or_high.values
        or_low_vals = or_low.values
        or_range_vals = or_range.values
        valid_vals = valid.values
        sid_vals = session_id.values

        above_raw, below_raw = apply_breakout_threshold(
            close_vals, or_high_vals, or_low_vals, or_range_vals,
            breakout_threshold, breakout_threshold_abs,
        )
        # Mask out range-period bars
        above_masked = above_raw & valid_vals
        below_masked = below_raw & valid_vals

        _, _, post_bull_arr, post_bear_arr, broke_high_arr, broke_low_arr = \
            compute_break_state(
                above_masked, below_masked, or_high_vals,
                sid_vals, len(df),
            )
        features[f"{prefix}_post_bull"] = pd.Series(
            post_bull_arr, index=df.index,
        ).where(valid, np.nan)
        features[f"{prefix}_post_bear"] = pd.Series(
            post_bear_arr, index=df.index,
        ).where(valid, np.nan)

        if enable_retracement:
            # ATR-based zone features (proximity to ORB boundaries)
            half_band_atr = retest_atr_width * atr
            near_high = (df["C"] >= or_high - half_band_atr) & (df["C"] <= or_high + half_band_atr)
            near_low = (df["C"] >= or_low - half_band_atr) & (df["C"] <= or_low + half_band_atr)
            features[f"{prefix}_retest_zone_up"] = near_high.astype(int).where(valid, np.nan)
            features[f"{prefix}_retest_zone_down"] = near_low.astype(int).where(valid, np.nan)

            # Retest signals per retracement level
            rl_list = [retracement_levels] if isinstance(retracement_levels, (int, float)) else list(retracement_levels)
            retest_half_band = retest_zone_width / 2 * or_range_vals

            for rl in rl_list:
                rl_pfx = f"rl{int(rl * 100)}_"
                entry_bull = or_high_vals - rl * or_range_vals
                entry_bear = or_low_vals + rl * or_range_vals

                retest_result = compute_retest_signals(
                    close=close_vals,
                    high=df["H"].values,
                    low=df["L"].values,
                    range_high=or_high_vals,
                    range_low=or_low_vals,
                    group_ids=sid_vals,
                    broke_high_arr=broke_high_arr,
                    broke_low_arr=broke_low_arr,
                    entry_bull=entry_bull,
                    entry_bear=entry_bear,
                    half_band=retest_half_band,
                    n=len(df),
                    min_retracement=min_retracement,
                )
                features[f"{prefix}_{rl_pfx}retest_bull"] = pd.Series(
                    retest_result["retest_bull"], index=df.index,
                ).where(valid, np.nan)
                features[f"{prefix}_{rl_pfx}retest_bear"] = pd.Series(
                    retest_result["retest_bear"], index=df.index,
                ).where(valid, np.nan)

                # SL: entry to opposite boundary
                rl_sl = ((1 - rl) * or_range).where(valid & (or_range > 0), np.nan)
                features[f"{prefix}_{rl_pfx}sl_dist"] = rl_sl

    return features


@register_indicator("opening_range")
class OpeningRangeIndicator(BaseIndicator):
    """
    Opening Range Breakout Features.

    Session ORB — für konfigurierbare Session-Stunden (forward-filled),
    mit Retest-Signalen bei konfigurierbaren Retracement-Levels.
    """

    name = "opening_range"
    version = "1.0.0"
    benefits_from_stationary = False
    group = "session"

    def compute(
        self,
        df: pd.DataFrame,
        range_bars: Union[int, List[int]] = 1,
        atr_period: int = 14,
        sessions: List[int] = None,
        enable_retracement: bool = True,
        retest_atr_width: float = 0.3,
        retest_zone_width: float = 0.5,
        carry_forward_days: Union[int, List[int]] = 0,
        pre_range_bars: Union[int, List[int]] = 0,
        candle_span: str = "hl",
        breakout_threshold: float = 0.0,
        breakout_threshold_abs: float = 0.0,
        retracement_levels: Union[float, List[float]] = 0.5,
        min_retracement: float = 0.0,
        **params,
    ) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame muss einen DatetimeIndex haben")

        # Skip daily data — ORB is intraday only
        if len(df) > 1:
            median_diff = df.index.to_series().diff().median()
            if median_diff >= pd.Timedelta(hours=20):
                return df

        if sessions is None:
            sessions = [8, 9, 14, 15]

        # Normalisiere range_bars zu einer Liste
        rb_list = [range_bars] if isinstance(range_bars, int) else list(range_bars)
        use_rb_prefix = len(rb_list) > 1

        # Normalisiere carry_forward_days / pre_range_bars zu Listen
        cf_list = [carry_forward_days] if isinstance(carry_forward_days, int) else list(carry_forward_days)
        prb_list = [pre_range_bars] if isinstance(pre_range_bars, int) else list(pre_range_bars)
        use_cf_prb_prefix = len(cf_list) > 1 or len(prb_list) > 1

        features: Dict[str, Union[pd.Series, np.ndarray]] = {}
        atr = _compute_atr(df, atr_period)

        for rb in rb_list:
            rb_pfx = f"rb{rb}_" if use_rb_prefix else ""

            for cf in cf_list:
                for prb in prb_list:
                    cf_prb_pfx = f"cf{cf}_prb{prb}_" if use_cf_prb_prefix else ""
                    session = _session_orb_features(
                        df, sessions, rb, atr, enable_retracement, retest_atr_width,
                        retest_zone_width, cf, prb, candle_span,
                        breakout_threshold, breakout_threshold_abs,
                        retracement_levels, min_retracement,
                    )
                    features.update({f"{rb_pfx}{cf_prb_pfx}{k}": v for k, v in session.items()})

        if not features:
            return df

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    # UTC sessions covered by the orb_scalping pipeline:
    # 0=Nikkei/ASX200 open, 1=Nikkei/HK50 morning, 2=All Asia morning,
    # 5=Nikkei/HK50 afternoon open, 6=DAX pre-market, 7=Xetra/DAX open,
    # 8=London open, 12=NY pre-dawn, 13=NY pre-market, 14=approaching NYSE open
    _PIPELINE_SESSIONS = [0, 1, 2, 5, 6, 7, 8, 12, 13, 14]

    def get_feature_columns(self) -> List[str]:
        session = []
        for h in self._PIPELINE_SESSIONS:
            pfx = f"orb_s{h:02d}"
            session.extend([
                f"{pfx}_range", f"{pfx}_position",
                f"{pfx}_breakout_up", f"{pfx}_breakout_down",
                f"{pfx}_range_vs_atr",
                f"{pfx}_poc_dist", f"{pfx}_sl_dist",
                f"{pfx}_post_bull", f"{pfx}_post_bear",
                f"{pfx}_retest_zone_up", f"{pfx}_retest_zone_down",
                f"{pfx}_rl50_retest_bull", f"{pfx}_rl50_retest_bear",
                f"{pfx}_rl50_sl_dist",
            ])
        return session

    def get_signal_columns(self) -> List[str]:
        signals = []
        for h in self._PIPELINE_SESSIONS:
            pfx = f"orb_s{h:02d}"
            signals.extend([
                f"{pfx}_breakout_up", f"{pfx}_breakout_down",
                f"{pfx}_rl50_retest_bull", f"{pfx}_rl50_retest_bear",
            ])
        return signals

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "range_bars": 1,
            "atr_period": 14,
            "sessions": [8, 9, 14, 15],
            "enable_retracement": True,
            "retest_atr_width": 0.3,
            "retest_zone_width": 0.5,
            "carry_forward_days": 0,
            "pre_range_bars": 0,
            "candle_span": "hl",
            "breakout_threshold": 0.0,
            "breakout_threshold_abs": 0.0,
            "retracement_levels": 0.5,
            "min_retracement": 0.0,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "range_bars": {
                "type": "list[int]",
                "default": 1,
                "description": "Number of bars defining the opening range after each hour boundary. At M15: 1 bar = 15min range, 2 bars = 30min range. At M5: 1 bar = 5min range. Can be a list (e.g. [1, 2]) to compute features for multiple range sizes simultaneously — each size gets its own prefixed columns (rb1_orb_*, rb2_orb_*) so the ML model can select the better variant.",
                "min": 1,
                "max": 12,
                "step": 1,
            },
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR period for normalizing the opening range size.",
                "min": 5,
                "max": 100,
                "step": 1,
            },
            "sessions": {
                "type": "list[int]",
                "default": [8, 9, 14, 15],
                "description": "Data-local hours for session-specific ORB features. Each session produces features that persist until the next occurrence of that session hour.",
                "min": 0,
                "max": 23,
            },
            "enable_retracement": {
                "type": "bool",
                "default": True,
                "description": "Enable retest zone features and retest signals at configured retracement levels.",
            },
            "retest_atr_width": {
                "type": "float",
                "default": 0.3,
                "description": "Half-bandwidth in ATR units around the ORB high/low that defines the boundary reload zone (retest_zone_up/down).",
                "min": 0.1,
                "max": 1.0,
                "step": 0.1,
            },
            "retest_zone_width": {
                "type": "float",
                "default": 0.5,
                "description": "Width of the retest zone as a fraction of the OR range (0-1). Range-based so the zone scales correctly regardless of ATR size.",
                "min": 0.1,
                "max": 1.0,
                "step": 0.1,
            },
            "carry_forward_days": {
                "type": "list[int]",
                "default": 0,
                "description": "If no breakout occurs during a session, carry the ORB range forward to the next N session occurrences. 0 = disabled.",
                "min": 0,
                "max": 5,
                "step": 1,
            },
            "pre_range_bars": {
                "type": "list[int]",
                "default": 0,
                "description": "Number of bars before the session start to include in the range calculation. Expands the opening range window backward.",
                "min": 0,
                "max": 8,
                "step": 1,
            },
            "candle_span": {
                "type": "choice",
                "default": "hl",
                "description": "Vertical extent of candles used for range. 'hl': full candle including wicks (H/L). 'body': candle body only (max/min of O/C, ignoring wicks).",
                "choices": ["hl", "body"],
            },
            "breakout_threshold": {
                "type": "float",
                "default": 0.0,
                "description": "Minimum distance as fraction of range for breakout. E.g. 0.05 = close must be at least 5% of range beyond boundary. 0 = disabled.",
                "min": 0.0,
                "max": 0.5,
                "step": 0.01,
            },
            "breakout_threshold_abs": {
                "type": "float",
                "default": 0.0,
                "description": "Minimum distance in absolute terms (pips/points) for breakout. Effective threshold = max(pct * range, abs). 0 = disabled.",
                "min": 0.0,
                "max": 100.0,
                "step": 1.0,
            },
            "retracement_levels": {
                "type": "list[float]",
                "default": 0.5,
                "description": (
                    "Retracement fraction(s) of the OR range for entry level. "
                    "0.5 = midpoint. 0 = at boundary. 0.382 = shallow retrace. "
                    "Always generates rl{N}_ prefixed columns (e.g. rl50_retest_bull). "
                    "Pass a list to precompute multiple variants for grid search."
                ),
                "min": 0.0,
                "max": 0.9,
                "step": 0.1,
            },
            "min_retracement": {
                "type": "float",
                "default": 0.0,
                "description": (
                    "Minimum retracement of OR range (checked via H/L) before "
                    "retest signal fires. 0.3 = price must retrace at least 30% from "
                    "breakout boundary. 0 = disabled."
                ),
                "min": 0.0,
                "max": 0.9,
                "step": 0.1,
            },
        }


__all__ = ["OpeningRangeIndicator"]
