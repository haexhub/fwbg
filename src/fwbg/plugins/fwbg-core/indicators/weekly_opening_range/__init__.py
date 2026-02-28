"""
Weekly Opening Range Breakout (WOR) Indicator Plugin.

Berechnet Features basierend auf dem wöchentlichen Opening Range Konzept:
- WOR Range/Position/Breakout relativ zur ersten N Bars der Handelswoche (Montag)
- Statistiken: durchschnittliche Wochenrange, Breakout-Rate, Continuation-Rate
- Reload-Zone: Preis kehrt nach Breakout zur WOR-Grenze zurück

Hintergrund:
Wöchentliche Opening Ranges werden von institutionellen Marktteilnehmern
intensiv beobachtet. Der erste Kurs nach dem Wochenende konzentriert akkumulierten
Overnight-/Wochenend-Orderflow. Breakouts über WOR High/Low haben statistisch
höhere Continuation-Wahrscheinlichkeit als zufällige Intraday-Breakouts.

Unterschied zu Opening Range (ORB):
- ORB: erste N Bars der Handelsstunde/-session (täglich)
- WOR: erste N Bars der Handelswoche (wöchentlich, Monday-Open)
- WOR-Level werden vom Markt wochenlang als Referenz genutzt

Timeframe-Kompatibilität: Intraday (M1-H4). Auf DAY-Bars oder WEEK-Bars → NaN.
"""
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide


# ── Naming helpers ──────────────────────────────────────────────────────────


def wor_col(rb: int, feature: str) -> str:
    """Build WOR column name: wor_rb{N}_{feature}."""
    return f"wor_rb{rb}_{feature}"


WOR_SIGNAL_SUFFIXES = ("_breakout_up", "_breakout_down")


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    return ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=period)


def _get_week_id(df: pd.DataFrame) -> pd.Series:
    """Create a unique week identifier per bar (year * 100 + ISO week number)."""
    iso = df.index.isocalendar()
    return pd.Series(
        iso.year.values * 100 + iso.week.values,
        index=df.index,
        dtype=np.int64,
    )


def _weekly_orb_features(
    df: pd.DataFrame,
    range_bars: int,
    atr: pd.Series,
) -> Dict[str, pd.Series]:
    """
    Compute weekly opening range features.

    The weekly range is defined by the first `range_bars` bars of each
    trading week (starting from Monday). Features are valid from bar
    `range_bars` onward and persist for the entire week.
    """
    features: Dict[str, pd.Series] = {}

    week_id = _get_week_id(df)
    bar_in_week = df.groupby(week_id).cumcount()

    # Opening range: H/L of first `range_bars` bars of each week
    or_mask = bar_in_week < range_bars
    or_high = df["H"].where(or_mask).groupby(week_id).transform("max")
    or_low = df["L"].where(or_mask).groupby(week_id).transform("min")
    or_range = or_high - or_low

    # Only valid after the opening range is fully established
    valid = bar_in_week >= range_bars

    features["wor_range"] = safe_divide(or_range, df["C"]).where(valid, np.nan)
    features["wor_position"] = safe_divide(df["C"] - or_low, or_range).where(valid, np.nan)
    features["wor_breakout_up"] = (df["C"] > or_high).astype(int).where(valid, np.nan)
    features["wor_breakout_down"] = (df["C"] < or_low).astype(int).where(valid, np.nan)
    features["wor_range_vs_atr"] = safe_divide(or_range, atr).where(valid, np.nan)

    # Distance from WOR High/Low (signed, ATR-normalized)
    features["wor_dist_to_high"] = safe_divide(or_high - df["C"], atr).where(valid, np.nan)
    features["wor_dist_to_low"] = safe_divide(df["C"] - or_low, atr).where(valid, np.nan)

    # SL distance for orb_based exit strategy.
    # Entry near WOR boundary (retest zone), SL at opposite boundary.
    # SL distance = full WOR range.
    features["wor_sl_dist"] = or_range.where(valid, np.nan)

    return features


def _weekly_stat_features(
    df: pd.DataFrame,
    week_id: pd.Series,
    stat_window: int,
) -> Dict[str, pd.Series]:
    """
    Compute rolling statistics over past weekly opening ranges.

    Uses only past weeks' data (shifted by 1 week) to avoid lookahead.
    """
    features: Dict[str, pd.Series] = {}

    bar_in_week = df.groupby(week_id).cumcount()
    valid = bar_in_week >= 1  # at least 1 bar into the week

    # Per-week opening range (first bar only)
    or_high_first = df["H"].where(bar_in_week == 0).groupby(week_id).transform("max")
    or_low_first = df["L"].where(bar_in_week == 0).groupby(week_id).transform("min")
    or_range_first = safe_divide(or_high_first - or_low_first, df["C"])

    # Aggregate one value per week (the first bar's range)
    # Using shift to get previous week's stats (no lookahead)
    _unique_weeks = week_id.drop_duplicates().sort_values()
    week_range_map = (
        or_range_first.where(bar_in_week == 0)
        .groupby(week_id)
        .first()
    )
    week_range_map.index = week_range_map.index.astype(np.int64)

    # Rolling stats over last stat_window weeks (shifted by 1 to exclude current week)
    week_range_rolling_avg = week_range_map.rolling(stat_window, min_periods=max(1, stat_window // 2)).mean().shift(1)
    week_range_rolling_std = week_range_map.rolling(stat_window, min_periods=max(1, stat_window // 2)).std().shift(1)

    avg_map = week_range_rolling_avg.to_dict()
    _std_map = week_range_rolling_std.to_dict()

    features["wor_stat_avg_range"] = week_id.map(avg_map).where(valid, np.nan)
    # Normalized: how wide is this week's range vs historical average?
    current_range_norm = safe_divide(
        or_range_first,
        week_id.map(avg_map).replace(0, np.nan),
    )
    features["wor_stat_range_vs_avg"] = current_range_norm.where(valid, np.nan)

    # Breakout rate: fraction of past weeks where price broke out of WOR
    or_high_week = df["H"].where(bar_in_week == 0).groupby(week_id).transform("max")
    or_low_week = df["L"].where(bar_in_week == 0).groupby(week_id).transform("min")
    _broke_out = ((df["H"].max() > or_high_week) | (df["L"].min() < or_low_week)).astype(float)

    # Per-week breakout flag (any bar in week broke out of first-bar range)
    week_breakout = (
        (df["C"].where(bar_in_week > 0) > or_high_week)
        | (df["C"].where(bar_in_week > 0) < or_low_week)
    ).groupby(week_id).transform("any").astype(float)

    week_breakout_map = (
        week_breakout.where(bar_in_week == 0)
        .groupby(week_id)
        .first()
    )
    week_breakout_map.index = week_breakout_map.index.astype(np.int64)
    breakout_rate = week_breakout_map.rolling(stat_window, min_periods=max(1, stat_window // 2)).mean().shift(1)
    breakout_rate_map = breakout_rate.to_dict()
    features["wor_stat_breakout_rate"] = week_id.map(breakout_rate_map).where(valid, np.nan)

    return features


@register_indicator("weekly_opening_range")
class WeeklyOpeningRangeIndicator(BaseIndicator):
    """
    Weekly Opening Range Breakout Features.

    Berechnet Features relativ zur wöchentlichen Opening Range:
    - WOR Range/Position: Wo steht der Preis relativ zur Wochenrange?
    - WOR Breakout: Hat der Preis die Wochenrange gebrochen?
    - WOR Reload Zone: Kehrt der Preis nach Breakout zur WOR-Grenze zurück?
    - WOR Statistiken: Historische Wochenrange-Daten
    """

    name = "weekly_opening_range"
    version = "1.0.0"
    benefits_from_stationary = False
    group = "session"

    def compute(
        self,
        df: pd.DataFrame,
        range_bars: Union[int, List[int]] = 2,
        atr_period: int = 14,
        stat_window: int = 12,
        enable_stats: bool = True,
        **params,
    ) -> pd.DataFrame:
        """
        Compute weekly opening range features.

        Args:
            df: OHLC DataFrame mit DatetimeIndex
            range_bars: Anzahl Bars die die Weekly Range definieren.
                        M15: 2 = erste 30 Min Montag, 4 = erste Stunde
            atr_period: ATR-Periode für Normalisierung
            stat_window: Rollendes Fenster (Wochen) für Statistik-Features
            enable_stats: Statistik-Features aktivieren
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame muss einen DatetimeIndex haben")

        # Skip daily/weekly data — WOR is intraday only
        if len(df) > 1:
            median_diff = df.index.to_series().diff().median()
            if median_diff >= pd.Timedelta(hours=20):
                return df

        rb_list = [range_bars] if isinstance(range_bars, int) else list(range_bars)

        atr = _compute_atr(df, atr_period)
        week_id = _get_week_id(df)
        features: Dict[str, pd.Series] = {}

        for rb in rb_list:
            wor = _weekly_orb_features(df, rb, atr)
            # Always use wor_rb{N}_ prefix (replace leading "wor_")
            wor = {f"wor_rb{rb}_{k[4:]}": v for k, v in wor.items()}
            features.update(wor)

        if enable_stats:
            stats = _weekly_stat_features(df, week_id, stat_window)
            features.update(stats)

        if not features:
            return df

        self._feature_columns = list(features.keys())
        self._signal_columns = [
            k for k in features if any(k.endswith(s) for s in WOR_SIGNAL_SUFFIXES)
        ]

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        if self._feature_columns:
            return self._feature_columns
        rb = self.get_default_params()["range_bars"]
        base_feats = [
            "range", "position", "breakout_up", "breakout_down",
            "range_vs_atr", "dist_to_high", "dist_to_low", "sl_dist",
        ]
        return [wor_col(rb, f) for f in base_feats] + [
            "wor_stat_avg_range", "wor_stat_range_vs_avg", "wor_stat_breakout_rate",
        ]

    def get_signal_columns(self) -> List[str]:
        if self._signal_columns:
            return self._signal_columns
        rb = self.get_default_params()["range_bars"]
        return [
            wor_col(rb, "breakout_up"), wor_col(rb, "breakout_down"),
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "range_bars": 2,
            "atr_period": 14,
            "stat_window": 12,
            "enable_stats": True,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "range_bars": {
                "type": "list[int]",
                "default": 2,
                "description": (
                    "Anzahl Bars die die Weekly Opening Range definieren. "
                    "Bei M15: 2 = erste 30 Min Montag (empfohlen für Scalping), "
                    "4 = erste Stunde (mehr Stabilität, weniger Trades). "
                    "Kann auch Liste sein [2, 4] für beide Varianten."
                ),
                "min": 1,
                "max": 16,
                "step": 1,
            },
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR-Periode für Normalisierung der WOR-Größe und Reload-Zone.",
                "min": 5,
                "max": 100,
                "step": 1,
            },
            "stat_window": {
                "type": "int",
                "default": 12,
                "description": (
                    "Rollendes Fenster (Wochen) für Statistik-Features. "
                    "12 = ~3 Monate, 26 = ~6 Monate. "
                    "Nur vergangene Wochen fließen ein (kein Lookahead)."
                ),
                "min": 4,
                "max": 52,
                "step": 4,
            },
            "enable_stats": {
                "type": "bool",
                "default": True,
                "description": "Statistik-Features aktivieren: durchschnittliche Wochenrange, Breakout-Rate.",
            },
        }


__all__ = ["WeeklyOpeningRangeIndicator"]
