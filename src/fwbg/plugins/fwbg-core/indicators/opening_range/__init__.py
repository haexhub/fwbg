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
) -> Dict[str, Union[pd.Series, np.ndarray]]:
    """
    Compute session-specific ORB features.

    For each configured session hour, compute the opening range at that hour
    and forward-fill until the next occurrence of that session hour.
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

        # Opening range: first range_bars bars of each session hour block
        or_mask = is_session & (bar_in_block < range_bars)
        or_high = df["H"].where(or_mask).groupby(session_id).transform("max")
        or_low = df["L"].where(or_mask).groupby(session_id).transform("min")

        # Valid: after first session, not within the opening range bars themselves
        in_range_period = is_session & (bar_in_block < range_bars)
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

        # SL distance: full ORB range (entry near breakout side → SL at opposite boundary)
        features[f"{prefix}_sl_dist"] = or_range.where(valid, np.nan)

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

            # Retest (reload) entry signal — recommended ORB entry at ORB midpoint (POC proxy).
            # Fires when: post-breakout AND price near midpoint AND thesis still valid.
            # Bull: broken up AND retrace to midpoint AND still above ORB Low
            # Bear: broken down AND retrace to midpoint AND still below ORB High
            near_poc = (df["C"] >= or_midpoint - half_band) & (df["C"] <= or_midpoint + half_band)
            still_valid_bull = df["C"] > or_low
            still_valid_bear = df["C"] < or_high

            post_bull_flag = above_cummax.where(valid, 0).astype(bool)
            post_bear_flag = below_cummax.where(valid, 0).astype(bool)

            features[f"{prefix}_retest_bull"] = (
                (post_bull_flag & near_poc & still_valid_bull).astype(float).where(valid, np.nan)
            )
            features[f"{prefix}_retest_bear"] = (
                (post_bear_flag & near_poc & still_valid_bear).astype(float).where(valid, np.nan)
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
        use_prefix = len(rb_list) > 1

        features: Dict[str, Union[pd.Series, np.ndarray]] = {}
        atr = _compute_atr(df, atr_period)

        for rb in rb_list:
            pfx = f"rb{rb}_" if use_prefix else ""

            if enable_rolling:
                rolling = _rolling_orb_features(df, rb, atr)
                features.update({f"{pfx}{k}": v for k, v in rolling.items()})

            if enable_session:
                session = _session_orb_features(
                    df, sessions, rb, atr, enable_retracement, retest_atr_width
                )
                features.update({f"{pfx}{k}": v for k, v in session.items()})

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
                "description": "Half-bandwidth in ATR units around the ORB high/low that defines the reload zone. A value of 0.3 means the zone extends 0.3 * ATR above and below the boundary.",
                "min": 0.1,
                "max": 1.0,
                "step": 0.1,
            },
        }


__all__ = ["OpeningRangeIndicator"]
