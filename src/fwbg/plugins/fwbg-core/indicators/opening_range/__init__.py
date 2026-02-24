"""
Opening Range Breakout (ORB) Indicator Plugin.

Berechnet Features basierend auf dem Opening Range Konzept:
- Rolling ORB: Range/Position/Breakout relativ zur letzten vollen Stunde
- Session ORB: Dasselbe für konfigurierbare Session-Stunden (forward-filled)
- Statistik: Durchschnittliche Range, Breakout-Rate, Continuation-Rate

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


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR via ta-lib, returns raw ATR (not percent)."""
    return ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=period)


def _rolling_orb_features(
    df: pd.DataFrame,
    range_bars: int,
    atr: pd.Series,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """
    Compute rolling ORB features relative to the current hour boundary.

    For each full hour, the opening range is the high/low of the first
    `range_bars` bars. Features are only valid after the range is established.
    """
    features: Dict[str, Union[pd.Series, np.ndarray]] = {}

    hour_group = df.index.floor("h")
    bar_in_hour = df.groupby(hour_group).cumcount()

    # Opening range = H/L of first `range_bars` bars per hour
    or_mask = bar_in_hour < range_bars
    or_high = df["H"].where(or_mask).groupby(hour_group).transform("max")
    or_low = df["L"].where(or_mask).groupby(hour_group).transform("min")
    or_range = or_high - or_low
    or_midpoint = (or_high + or_low) / 2

    # Features only valid after range is established
    valid = bar_in_hour >= range_bars

    features["orb_range"] = safe_divide(or_range, df["C"]).where(valid, np.nan)
    features["orb_position"] = safe_divide(df["C"] - or_low, or_range).where(valid, np.nan)

    # Event feature: 1 only on the FIRST bar where C crosses above/below the range boundary.
    # Subsequent bars that remain above/below are 0 (transition detection, not state).
    above_int = (df["C"] > or_high).astype(np.int8)
    below_int = (df["C"] < or_low).astype(np.int8)
    prev_above = above_int.groupby(hour_group).shift(1).fillna(0).astype(np.int8)
    prev_below = below_int.groupby(hour_group).shift(1).fillna(0).astype(np.int8)
    features["orb_breakout_up"] = (above_int - prev_above).clip(lower=0).where(valid, np.nan)
    features["orb_breakout_down"] = (below_int - prev_below).clip(lower=0).where(valid, np.nan)

    features["orb_range_vs_atr"] = safe_divide(or_range, atr).where(valid, np.nan)

    # POC proxy: normalized distance from close to ORB midpoint (equilibrium without tick volume)
    features["orb_poc_dist"] = safe_divide(df["C"] - or_midpoint, atr).where(valid, np.nan)

    # SL distance: full ORB range (entry near ORB High → SL at ORB Low, or vice versa)
    features["orb_sl_dist"] = or_range.where(valid, np.nan)

    return features


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

        # Opening range: body of the combined reference candle (O of first bar, C of last bar)
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

        # Post-breakout STATE: 1 for all bars after first breakout in this session.
        # Resets at each new session start via groupby(session_id).cummax().
        above_cummax = sess_above_int.groupby(session_id).cummax()
        below_cummax = sess_below_int.groupby(session_id).cummax()
        features[f"{prefix}_post_bull"] = above_cummax.where(valid, np.nan)
        features[f"{prefix}_post_bear"] = below_cummax.where(valid, np.nan)

        if enable_retracement:
            # Reload zone: is price within retest_atr_width * ATR of the ORB boundary?
            # This captures the "test of the broken level" setup described by ICT/IVB traders.
            half_band = retest_atr_width * atr
            near_high = (df["C"] >= or_high - half_band) & (df["C"] <= or_high + half_band)
            near_low = (df["C"] >= or_low - half_band) & (df["C"] <= or_low + half_band)
            features[f"{prefix}_retest_zone_up"] = near_high.astype(int).where(valid, np.nan)
            features[f"{prefix}_retest_zone_down"] = near_low.astype(int).where(valid, np.nan)

            # Retest (reload) entry signal — fires on the FIRST bar price enters the midpoint
            # zone after a breakout (event signal, not sustained state).
            # Bull: broke above OR High AND retraced to midpoint AND no subsequent bear breakout
            # Bear: broke below OR Low AND retraced to midpoint AND no subsequent bull breakout
            #
            # near_poc zone: range-based (fraction of OR range centred on midpoint).
            # retest_zone_width=0.5  → covers the middle 50 % of the OR range (25 %–75 %).
            # This is independent of ATR so the zone scales correctly with every range size,
            # and fast retracement bars that skip an ATR-sized window are much less likely
            # to miss the trigger.
            zone_half = (retest_zone_width / 2) * or_range
            near_poc = (df["C"] >= or_midpoint - zone_half) & (df["C"] <= or_midpoint + zone_half)

            post_bull_flag = above_cummax.where(valid, 0).astype(bool)
            post_bear_flag = below_cummax.where(valid, 0).astype(bool)

            # Thesis still valid: had breakout in one direction, but NOT also in the other.
            # If price broke both sides the range is "used up" — no clean retest.
            still_valid_bull = above_cummax.astype(bool) & ~below_cummax.astype(bool)
            still_valid_bear = below_cummax.astype(bool) & ~above_cummax.astype(bool)

            bull_cond = post_bull_flag & near_poc & still_valid_bull
            bear_cond = post_bear_flag & near_poc & still_valid_bear

            # First-touch: fire only when condition transitions False → True within the session
            bull_int  = bull_cond.astype(np.int8)
            bear_int  = bear_cond.astype(np.int8)
            prev_bull = bull_int.groupby(session_id).shift(1).fillna(0).astype(np.int8)
            prev_bear = bear_int.groupby(session_id).shift(1).fillna(0).astype(np.int8)

            features[f"{prefix}_retest_bull"] = (
                (bull_int - prev_bull).clip(lower=0).astype(float).where(valid, np.nan)
            )
            features[f"{prefix}_retest_bear"] = (
                (bear_int - prev_bear).clip(lower=0).astype(float).where(valid, np.nan)
            )

    return features


def _stat_features(
    df: pd.DataFrame,
    stat_window: int,
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """
    Compute rolling statistics over hourly opening ranges.

    Uses the first bar of each hour as the "opening range" reference, then
    checks whether later bars in that hour broke out. Aggregates over a
    rolling window of bars.
    """
    features: Dict[str, Union[pd.Series, np.ndarray]] = {}

    hour_group = df.index.floor("h")
    bar_in_hour = df.groupby(hour_group).cumcount()

    # Opening range: H/L of the first bar of each hour
    or_high = df["H"].where(bar_in_hour == 0).groupby(hour_group).transform("max")
    or_low = df["L"].where(bar_in_hour == 0).groupby(hour_group).transform("min")
    or_range = safe_divide(or_high - or_low, df["C"])

    # Breakout: close outside opening range (only for bars after the first)
    valid = bar_in_hour > 0
    broke_up = (df["C"] > or_high) & valid
    broke_down = (df["C"] < or_low) & valid
    broke_any = (broke_up | broke_down).astype(float).where(valid, np.nan)

    # Rolling window in bars
    win = stat_window * 4
    min_periods = max(1, win // 2)

    features["orb_stat_avg_range"] = or_range.rolling(
        win, min_periods=min_periods
    ).mean()

    features["orb_stat_breakout_rate"] = broke_any.rolling(
        win, min_periods=min_periods
    ).mean()

    # Continuation: did breakout direction persist to the next hour?
    direction = np.sign(df["C"] - df["C"].shift(4))
    continued = (direction == direction.shift(4)).astype(float)
    features["orb_stat_continuation_rate"] = continued.rolling(
        win, min_periods=min_periods
    ).mean()

    return features


@register_indicator("opening_range")
class OpeningRangeIndicator(BaseIndicator):
    """
    Opening Range Breakout Features.

    Berechnet drei Feature-Gruppen:
    1. Rolling ORB — relativ zur letzten vollen Stunde
    2. Session ORB — für konfigurierbare Session-Stunden (forward-filled)
    3. Statistik — Durchschnittliche Range, Breakout-Rate, Continuation-Rate
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
        stat_window: int = 20,
        enable_rolling: bool = True,
        enable_session: bool = True,
        enable_stats: bool = True,
        enable_retracement: bool = True,
        retest_atr_width: float = 0.3,
        retest_zone_width: float = 0.5,
        carry_forward_days: Union[int, List[int]] = 0,
        pre_range_bars: Union[int, List[int]] = 0,
        **params,
    ) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame muss einen DatetimeIndex haben")

        # Skip daily data — ORB is intraday only
        if len(df) > 1:
            median_diff = df.index.to_series().diff().median()
            if median_diff >= pd.Timedelta(hours=20):
                # Daily data — return df unchanged
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

            if enable_rolling:
                rolling = _rolling_orb_features(df, rb, atr)
                features.update({f"{rb_pfx}{k}": v for k, v in rolling.items()})

            if enable_session:
                for cf in cf_list:
                    for prb in prb_list:
                        cf_prb_pfx = f"cf{cf}_prb{prb}_" if use_cf_prb_prefix else ""
                        session = _session_orb_features(
                            df, sessions, rb, atr, enable_retracement, retest_atr_width,
                            retest_zone_width, cf, prb,
                        )
                        features.update({f"{rb_pfx}{cf_prb_pfx}{k}": v for k, v in session.items()})

        if enable_stats:
            features.update(_stat_features(df, stat_window))

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
        # Default: range_bars=1 (no prefix), sessions from orb_scalping pipeline
        rolling = [
            "orb_range", "orb_position", "orb_breakout_up",
            "orb_breakout_down", "orb_range_vs_atr",
            "orb_poc_dist", "orb_sl_dist",
        ]
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
                f"{pfx}_retest_bull", f"{pfx}_retest_bear",
            ])
        stats = [
            "orb_stat_avg_range", "orb_stat_breakout_rate",
            "orb_stat_continuation_rate",
        ]
        return rolling + session + stats

    def get_signal_columns(self) -> List[str]:
        signals = ["orb_breakout_up", "orb_breakout_down"]
        for h in self._PIPELINE_SESSIONS:
            pfx = f"orb_s{h:02d}"
            signals.extend([
                f"{pfx}_breakout_up", f"{pfx}_breakout_down",
                f"{pfx}_retest_bull", f"{pfx}_retest_bear",
            ])
        return signals

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "range_bars": 1,
            "atr_period": 14,
            "sessions": [8, 9, 14, 15],
            "stat_window": 20,
            "enable_rolling": True,
            "enable_session": True,
            "enable_stats": True,
            "enable_retracement": True,
            "retest_atr_width": 0.3,
            "retest_zone_width": 0.5,
            "carry_forward_days": 0,
            "pre_range_bars": 0,
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
                "description": "ATR period for normalizing the opening range size. The orb_range_vs_atr feature divides the opening range by ATR to identify unusually narrow (potential breakout) or wide (potential mean-reversion) ranges.",
                "min": 5,
                "max": 100,
                "step": 1,
            },
            "sessions": {
                "type": "list[int]",
                "default": [8, 9, 14, 15],
                "description": "Data-local hours for session-specific ORB features. Default: 8/9 (DAX pre-open + Xetra open), 14/15 (US pre-market + NY open). Each session produces 5 features that persist until the next occurrence of that session hour.",
                "min": 0,
                "max": 23,
            },
            "stat_window": {
                "type": "int",
                "default": 20,
                "description": "Rolling window (in hours) for statistical features: average range, breakout rate, and continuation rate. Larger windows give more stable statistics but react slower to regime changes.",
                "min": 5,
                "max": 200,
                "step": 5,
            },
            "enable_rolling": {
                "type": "bool",
                "default": True,
                "description": "Enable rolling ORB features computed relative to the last full-hour boundary. These 5 features capture the current hour's price action dynamics.",
            },
            "enable_session": {
                "type": "bool",
                "default": True,
                "description": "Enable session-specific ORB features for each configured session hour. Produces 5 features per session that persist until the next session start.",
            },
            "enable_stats": {
                "type": "bool",
                "default": True,
                "description": "Enable statistical features: rolling average range, breakout rate, and continuation rate over the stat_window.",
            },
            "enable_retracement": {
                "type": "bool",
                "default": True,
                "description": "Enable reload-zone features: binary signal when price is within retest_atr_width * ATR of the session ORB high or low. Captures the IVB/ICT 'test of the broken level' setup where smart money reloads after a breakout.",
            },
            "retest_atr_width": {
                "type": "float",
                "default": 0.3,
                "description": "Half-bandwidth in ATR units around the ORB high/low that defines the boundary reload zone (retest_zone_up/down). A value of 0.3 means the zone extends 0.3 * ATR above and below the OR boundary.",
                "min": 0.1,
                "max": 1.0,
                "step": 0.1,
            },
            "retest_zone_width": {
                "type": "float",
                "default": 0.5,
                "description": "Width of the midpoint retest zone as a fraction of the OR range (0–1). Default 0.5 means the zone covers the middle 50 % of the range (25 %–75 % from OR low). Range-based so the zone scales correctly regardless of ATR size — prevents fast retracement bars from skipping a fixed ATR-width window.",
                "min": 0.1,
                "max": 1.0,
                "step": 0.1,
            },
            "carry_forward_days": {
                "type": "list[int]",
                "default": 0,
                "description": "If no breakout occurs during a session, carry the ORB range forward to the next N session occurrences. 0 = disabled (each session uses its own range). Can be a list (e.g. [0, 1, 2]) to compute features for multiple carry durations — each gets prefixed columns (cf0_prb0_orb_*, cf1_prb0_orb_*) so the optimizer can select the best variant.",
                "min": 0,
                "max": 5,
                "step": 1,
            },
            "pre_range_bars": {
                "type": "list[int]",
                "default": 0,
                "description": "Number of bars before the session start to include in the range calculation. Expands the opening range window backward. Can be a list (e.g. [0, 1]) to compute features for multiple pre-range sizes — each gets prefixed columns (cf0_prb0_orb_*, cf0_prb1_orb_*) so the optimizer can select the best variant.",
                "min": 0,
                "max": 8,
                "step": 1,
            },
        }


__all__ = ["OpeningRangeIndicator"]
