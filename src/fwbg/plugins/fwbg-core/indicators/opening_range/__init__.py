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

    # Features only valid after range is established
    valid = bar_in_hour >= range_bars

    features["orb_range"] = safe_divide(or_range, df["C"]).where(valid, np.nan)
    features["orb_position"] = safe_divide(df["C"] - or_low, or_range).where(valid, np.nan)
    features["orb_breakout_up"] = (df["C"] > or_high).astype(int).where(valid, np.nan)
    features["orb_breakout_down"] = (df["C"] < or_low).astype(int).where(valid, np.nan)
    features["orb_range_vs_atr"] = safe_divide(or_range, atr).where(valid, np.nan)

    return features


def _session_orb_features(
    df: pd.DataFrame,
    sessions: List[int],
    range_bars: int,
    atr: pd.Series,
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

        features[f"{prefix}_range"] = safe_divide(or_range, df["C"]).where(valid, np.nan)
        features[f"{prefix}_position"] = safe_divide(
            df["C"] - or_low, or_range
        ).where(valid, np.nan)
        features[f"{prefix}_breakout_up"] = (
            (df["C"] > or_high).astype(int).where(valid, np.nan)
        )
        features[f"{prefix}_breakout_down"] = (
            (df["C"] < or_low).astype(int).where(valid, np.nan)
        )
        features[f"{prefix}_range_vs_atr"] = safe_divide(or_range, atr).where(
            valid, np.nan
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
                session = _session_orb_features(df, sessions, rb, atr)
                features.update({f"{pfx}{k}": v for k, v in session.items()})

        if enable_stats:
            features.update(_stat_features(df, stat_window))

        if not features:
            return df

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        # Für den Default-Fall (range_bars=1, kein Präfix)
        rolling = [
            "orb_range", "orb_position", "orb_breakout_up",
            "orb_breakout_down", "orb_range_vs_atr",
        ]
        session = []
        for h in [8, 9, 14, 15]:
            pfx = f"orb_s{h:02d}"
            session.extend([
                f"{pfx}_range", f"{pfx}_position",
                f"{pfx}_breakout_up", f"{pfx}_breakout_down",
                f"{pfx}_range_vs_atr",
            ])
        stats = [
            "orb_stat_avg_range", "orb_stat_breakout_rate",
            "orb_stat_continuation_rate",
        ]
        return rolling + session + stats

    def get_signal_columns(self) -> List[str]:
        signals = ["orb_breakout_up", "orb_breakout_down"]
        for h in [8, 9, 14, 15]:
            pfx = f"orb_s{h:02d}"
            signals.extend([f"{pfx}_breakout_up", f"{pfx}_breakout_down"])
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
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "range_bars": {
                "type": "int | list[int]",
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
        }


__all__ = ["OpeningRangeIndicator"]
