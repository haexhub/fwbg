"""
Weekly Opening Range Breakout (WOR) Indicator Plugin.

Berechnet Features basierend auf dem wöchentlichen Opening Range Konzept:
- WOR Range/Position/Breakout relativ zum Zeitfenster der ersten N Bars der Handelswoche
  (die Woche beginnt mit ihrer ersten Bar: Sonntagabend bei FX, Montag an Börsen)
- Statistiken: durchschnittliche Wochenrange, Breakout-Rate, Continuation-Rate
- Reload-Zone: Preis kehrt nach Breakout zur WOR-Grenze zurück

Hintergrund:
Wöchentliche Opening Ranges werden von institutionellen Marktteilnehmern
intensiv beobachtet. Der erste Kurs nach dem Wochenende konzentriert akkumulierten
Overnight-/Wochenend-Orderflow. Breakouts über WOR High/Low haben statistisch
höhere Continuation-Wahrscheinlichkeit als zufällige Intraday-Breakouts.

Unterschied zu Opening Range (ORB):
- ORB: erste N Bars der Handelsstunde/-session (täglich)
- WOR: erste N Bars der Handelswoche (wöchentlich, erste Bar der Woche)
- WOR-Level werden vom Markt wochenlang als Referenz genutzt

Timeframe-Kompatibilität: Intraday (M1-H4). Auf DAY-Bars oder WEEK-Bars → NaN.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide


# ── Naming helpers ──────────────────────────────────────────────────────────


def wor_col(rb: int, feature: str) -> str:
    """Build WOR column name: wor_rb{N}_{feature}."""
    return f"wor_rb{rb}_{feature}"


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    return ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=period)


# A trading week starts at its first bar.  FX/CFD data reopens on Sunday evening,
# which is ISO week N-1 by the calendar but belongs to the trading week that starts
# on Monday.  Shifting the clock by 12 hours moves every bar from Sunday 12:00 on
# into the following ISO week while Friday (even late Friday in any time zone
# used for the data) stays in its own week.
_WEEK_ROLLOVER_SHIFT = pd.Timedelta(hours=12)


def _get_week_id(df: pd.DataFrame) -> pd.Series:
    """Trading-week identifier per bar (ISO year * 100 + ISO week of ``ts + 12h``)."""
    iso = (df.index + _WEEK_ROLLOVER_SHIFT).isocalendar()
    return pd.Series(
        iso.year.values.astype(np.int64) * 100 + iso.week.values.astype(np.int64),
        index=df.index,
        dtype=np.int64,
    )


def _bar_step(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Typical bar spacing (median index difference, 1 minute fallback)."""
    if len(index) >= 2:
        step = index.to_series().diff().median()
        if pd.notna(step) and step > pd.Timedelta(0):
            return step
    return pd.Timedelta(minutes=1)


def _range_window(
    df: pd.DataFrame, week_id: pd.Series, range_bars: int, step: pd.Timedelta
) -> tuple[pd.Series, pd.Series]:
    """(in_range, after_range) masks for the weekly opening range.

    The range is the time window ``[first bar of the week, + range_bars * step)``,
    so a missing bar shrinks the range instead of stretching it to later bars.
    """
    ts = pd.Series(df.index, index=df.index)
    week_open = ts.groupby(week_id).transform("min")
    range_end = week_open + range_bars * step
    return ts < range_end, ts >= range_end


def _weekly_orb_features(
    df: pd.DataFrame,
    range_bars: int,
    atr: pd.Series,
    week_id: Optional[pd.Series] = None,
    step: Optional[pd.Timedelta] = None,
) -> Dict[str, pd.Series]:
    """
    Compute weekly opening range features.

    The weekly range is defined by the first `range_bars` bar periods of each
    trading week (starting at the week's first bar, Sunday evening for FX).
    Features are valid from the end of that window onward and persist for the
    entire week.
    """
    features: Dict[str, pd.Series] = {}

    if week_id is None:
        week_id = _get_week_id(df)
    if step is None:
        step = _bar_step(df.index)
    or_mask, valid = _range_window(df, week_id, range_bars, step)

    # Opening range: H/L of the bars in the range window of each week
    or_high = df["H"].where(or_mask).groupby(week_id).transform("max")
    or_low = df["L"].where(or_mask).groupby(week_id).transform("min")
    or_range = or_high - or_low

    # Only valid after the opening range is fully established
    features["wor_range"] = safe_divide(or_range, df["C"]).where(valid, np.nan)
    features["wor_position"] = safe_divide(df["C"] - or_low, or_range).where(
        valid, np.nan
    )
    features["wor_breakout_up"] = (df["C"] > or_high).astype(int).where(valid, np.nan)
    features["wor_breakout_down"] = (df["C"] < or_low).astype(int).where(valid, np.nan)
    features["wor_range_vs_atr"] = safe_divide(or_range, atr).where(valid, np.nan)

    # Distance from WOR High/Low (signed, ATR-normalized)
    features["wor_dist_to_high"] = safe_divide(or_high - df["C"], atr).where(
        valid, np.nan
    )
    features["wor_dist_to_low"] = safe_divide(df["C"] - or_low, atr).where(
        valid, np.nan
    )

    # SL distance for orb_based exit strategy.
    # Entry near WOR boundary (retest zone), SL at opposite boundary.
    # SL distance = full WOR range.
    features["wor_sl_dist"] = or_range.where(valid, np.nan)

    return features


def _weekly_stat_features(
    df: pd.DataFrame,
    week_id: pd.Series,
    stat_window: int,
    range_bars: int = 1,
    step: Optional[pd.Timedelta] = None,
) -> Dict[str, pd.Series]:
    """
    Compute rolling statistics over past weekly opening ranges.

    The weekly range is the same ``range_bars`` window as the ``wor_rb{N}_*``
    features.  Uses only past weeks' data (shifted by 1 week) to avoid lookahead.
    """
    features: Dict[str, pd.Series] = {}

    if step is None:
        step = _bar_step(df.index)
    in_range, after_range = _range_window(df, week_id, range_bars, step)
    bar_in_week = df.groupby(week_id).cumcount()
    valid = bar_in_week >= 1  # at least 1 bar into the week

    # Per-week opening range (bars inside the range window)
    or_high_week = df["H"].where(in_range).groupby(week_id).transform("max")
    or_low_week = df["L"].where(in_range).groupby(week_id).transform("min")
    or_range_week = safe_divide(or_high_week - or_low_week, df["C"])

    # Aggregate one value per week; rolling windows are shifted by one week
    # so the current week never enters its own statistics (no lookahead).
    week_range_map = or_range_week.groupby(week_id).first()
    week_range_map.index = week_range_map.index.astype(np.int64)

    # Rolling stats over last stat_window weeks (shifted by 1 to exclude current week)
    week_range_rolling_avg = (
        week_range_map.rolling(stat_window, min_periods=max(1, stat_window // 2))
        .mean()
        .shift(1)
    )
    avg_map = week_range_rolling_avg.to_dict()

    features["wor_stat_avg_range"] = week_id.map(avg_map).where(valid, np.nan)
    # Normalized: how wide is this week's range vs historical average?
    # The current range is only known once the range window is complete.
    current_range_norm = safe_divide(
        or_range_week,
        week_id.map(avg_map).replace(0, np.nan),
    )
    features["wor_stat_range_vs_avg"] = current_range_norm.where(after_range, np.nan)

    # Per-week breakout flag (any close after the range window left the range)
    week_breakout = (
        (
            (df["C"].where(after_range) > or_high_week)
            | (df["C"].where(after_range) < or_low_week)
        )
        .groupby(week_id)
        .transform("any")
        .astype(float)
    )

    week_breakout_map = week_breakout.groupby(week_id).first()
    week_breakout_map.index = week_breakout_map.index.astype(np.int64)
    breakout_rate = (
        week_breakout_map.rolling(stat_window, min_periods=max(1, stat_window // 2))
        .mean()
        .shift(1)
    )
    breakout_rate_map = breakout_rate.to_dict()
    features["wor_stat_breakout_rate"] = week_id.map(breakout_rate_map).where(
        valid, np.nan
    )

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
    version = "1.1.0"
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
                        M15: 2 = erste 30 Min der Handelswoche, 4 = erste Stunde
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
        step = _bar_step(df.index)
        features: Dict[str, pd.Series] = {}

        for rb in rb_list:
            wor = _weekly_orb_features(df, rb, atr, week_id, step)
            # Always use wor_rb{N}_ prefix (replace leading "wor_")
            wor = {f"wor_rb{rb}_{k[4:]}": v for k, v in wor.items()}
            features.update(wor)

        if enable_stats:
            # The week-level statistics are not rb-prefixed: they use the first
            # configured range_bars value.
            stats = _weekly_stat_features(df, week_id, stat_window, rb_list[0], step)
            features.update(stats)

        if not features:
            return df

        self._feature_columns = list(features.keys())

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def _resolve_range_bars(self, params=None) -> List[int]:
        """Extract range_bars from params or fall back to defaults."""
        p = params if params else self.get_default_params()
        rb = p.get("range_bars", self.get_default_params()["range_bars"])
        return [rb] if isinstance(rb, int) else list(rb)

    def get_feature_columns(self, params=None) -> List[str]:
        if self._feature_columns and not params:
            return self._feature_columns
        rb_list = self._resolve_range_bars(params)
        base_feats = [
            "range",
            "position",
            "breakout_up",
            "breakout_down",
            "range_vs_atr",
            "dist_to_high",
            "dist_to_low",
            "sl_dist",
        ]
        cols = []
        for rb in rb_list:
            cols.extend(wor_col(rb, f) for f in base_feats)
        p = params if params else self.get_default_params()
        if p.get("enable_stats", True):
            cols.extend(
                [
                    "wor_stat_avg_range",
                    "wor_stat_range_vs_avg",
                    "wor_stat_breakout_rate",
                ]
            )
        return cols

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
                    "Bei M15: 2 = erste 30 Min der Handelswoche (empfohlen für Scalping), "
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
