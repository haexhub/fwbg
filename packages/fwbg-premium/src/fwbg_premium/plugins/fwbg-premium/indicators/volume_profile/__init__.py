"""
Volume Profile Indicator Plugin.

Berechnet den Volumen-Profil der vorherigen Session und erzeugt daraus Features:
- POC (Point of Control): Preisniveau mit dem höchsten Volumen
- VAH (Value Area High): Obere Grenze der Value Area (70% des Volumens)
- VAL (Value Area Low): Untere Grenze der Value Area

Implementierung:
- Bar-basierte Annäherung (OHLCV): Volumen wird proportional über die H-L-Range verteilt
- Fallback auf TPO (Time Price Opportunity) wenn kein Volumen vorhanden
- Ausschließlich vorherige Session → kein Lookahead-Bias

Interpretation (aus IVB/ICT-Konzept):
- POC = "Reload Zone" nach ORB-Breakout (Markt kehrt oft zum Volumen-Schwerpunkt zurück)
- VAH/VAL = Grenzwerte der institutionellen Aktivität (Breakout darüber/darunter = echte Bewegung)
- Inside VA = Konsolidierungszone (Mean-Reversion-Bias)
- Outside VA = Direktionale Bewegung möglich

Timeframe-Kompatibilität: Intraday (M1-H4). Auf DAY-Bars → NaN.
"""
from typing import Dict, List, Union, Tuple

import numpy as np
import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, safe_divide


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    return ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=period)


def _compute_session_profile(
    day_df: pd.DataFrame,
    n_levels: int,
    value_area_pct: float,
    use_volume: bool,
) -> Tuple[float, float, float, float, float]:
    """
    Compute volume profile for a single session.

    Returns (poc, vah, val, session_high, session_low).
    Uses volume if available, otherwise TPO (1 unit per bar).
    """
    s_low = day_df["L"].min()
    s_high = day_df["H"].max()
    s_range = s_high - s_low

    if s_range == 0 or len(day_df) < 2:
        mid = (s_high + s_low) / 2.0
        return mid, mid, mid, s_high, s_low

    level_prices = np.linspace(s_low, s_high, n_levels + 1)
    level_vol = np.zeros(n_levels, dtype=np.float64)

    for _, bar in day_df.iterrows():
        vol = float(bar["V"]) if use_volume else 1.0
        if vol <= 0:
            vol = 1.0

        bar_range = bar["H"] - bar["L"]
        if bar_range == 0:
            # Single-price bar: all volume at the close level
            idx = int(np.clip(np.searchsorted(level_prices, bar["C"]) - 1, 0, n_levels - 1))
            level_vol[idx] += vol
        else:
            # Distribute volume proportionally across overlapping price levels
            for i in range(n_levels):
                overlap = min(bar["H"], level_prices[i + 1]) - max(bar["L"], level_prices[i])
                if overlap > 0:
                    level_vol[i] += vol * (overlap / bar_range)

    # POC = level with highest volume
    poc_idx = int(np.argmax(level_vol))
    poc = (level_prices[poc_idx] + level_prices[poc_idx + 1]) / 2.0

    # Value Area: expand from POC until value_area_pct of total volume is covered
    total = level_vol.sum()
    if total == 0:
        mid = (s_high + s_low) / 2.0
        return mid, mid, mid, s_high, s_low

    target = total * value_area_pct
    va_vol = level_vol[poc_idx]
    hi = poc_idx
    lo = poc_idx

    while va_vol < target:
        up_vol = level_vol[hi + 1] if hi < n_levels - 1 else 0.0
        dn_vol = level_vol[lo - 1] if lo > 0 else 0.0

        if up_vol == 0 and dn_vol == 0:
            break
        if up_vol >= dn_vol and hi < n_levels - 1:
            hi += 1
            va_vol += level_vol[hi]
        elif lo > 0:
            lo -= 1
            va_vol += level_vol[lo]
        else:
            hi += 1
            va_vol += level_vol[hi]

    vah = level_prices[hi + 1]
    val = level_prices[lo]

    return poc, vah, val, s_high, s_low


def _build_profiles(
    df: pd.DataFrame,
    n_levels: int,
    value_area_pct: float,
) -> Dict:
    """
    Compute volume profiles for all calendar days in the dataframe.

    Returns dict: date -> (poc, vah, val, session_high, session_low)
    """
    use_volume = "V" in df.columns and (df["V"] > 0).any()
    profiles = {}
    dates = df.index.date
    unique_dates = sorted(set(dates))

    for d in unique_dates:
        day_mask = dates == d
        day_df = df[day_mask]
        if len(day_df) < 2:
            continue
        profiles[d] = _compute_session_profile(day_df, n_levels, value_area_pct, use_volume)

    return profiles


@register_indicator("volume_profile")
class VolumeProfileIndicator(BaseIndicator):
    """
    Session Volume Profile Features.

    Berechnet POC, Value Area High/Low der vorherigen Session und
    erzeugt normalisierte Distanz-Features relativ zur aktuellen ATR.

    Features:
    - vp_poc_dist: (close - prev_POC) / ATR
    - vp_vah_dist: (close - prev_VAH) / ATR
    - vp_val_dist: (close - prev_VAL) / ATR
    - vp_inside_va: 1 wenn close zwischen VAL und VAH
    - vp_poc_rel_pos: POC-Position in der vorherigen Session-Range (0=bottom, 1=top)
    - vp_va_width_ratio: Breite der Value Area als Anteil der Session-Range
    """

    name = "volume_profile"
    version = "1.0.0"
    benefits_from_stationary = False
    group = "session"

    def compute(
        self,
        df: pd.DataFrame,
        atr_period: int = 14,
        n_levels: int = 50,
        value_area_pct: float = 0.70,
        **params,
    ) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame muss einen DatetimeIndex haben")

        # Skip daily data
        if len(df) > 1:
            median_diff = df.index.to_series().diff().median()
            if median_diff >= pd.Timedelta(hours=20):
                return df

        atr = _compute_atr(df, atr_period)
        profiles = _build_profiles(df, n_levels, value_area_pct)

        dates = df.index.date
        unique_dates = sorted(profiles.keys())

        poc_arr = np.full(len(df), np.nan)
        vah_arr = np.full(len(df), np.nan)
        val_arr = np.full(len(df), np.nan)
        sh_arr = np.full(len(df), np.nan)
        sl_arr = np.full(len(df), np.nan)

        # Assign PREVIOUS day's profile to each bar of the current day
        date_to_idx = {d: np.where(dates == d)[0] for d in unique_dates}

        for i, d in enumerate(unique_dates):
            if i == 0:
                continue
            prev_d = unique_dates[i - 1]
            if prev_d not in profiles:
                continue
            poc, vah, val, sh, sl = profiles[prev_d]
            idxs = date_to_idx.get(d, [])
            if len(idxs) == 0:
                continue
            poc_arr[idxs] = poc
            vah_arr[idxs] = vah
            val_arr[idxs] = val
            sh_arr[idxs] = sh
            sl_arr[idxs] = sl

        poc_s = pd.Series(poc_arr, index=df.index)
        vah_s = pd.Series(vah_arr, index=df.index)
        val_s = pd.Series(val_arr, index=df.index)
        sh_s = pd.Series(sh_arr, index=df.index)
        sl_s = pd.Series(sl_arr, index=df.index)

        close = df["C"]
        session_range = sh_s - sl_s

        features = {
            # Signed distance from close to profile levels, normalized by ATR
            "vp_poc_dist": safe_divide(close - poc_s, atr),
            "vp_vah_dist": safe_divide(close - vah_s, atr),
            "vp_val_dist": safe_divide(close - val_s, atr),
            # Is price currently inside the Value Area?
            "vp_inside_va": (
                ((close >= val_s) & (close <= vah_s))
                .astype(float)
                .where(vah_s.notna(), np.nan)
            ),
            # POC position within previous session range: >0.5 = bullish structure
            "vp_poc_rel_pos": safe_divide(poc_s - sl_s, session_range),
            # Value Area width relative to total session range: low = concentrated volume
            "vp_va_width_ratio": safe_divide(vah_s - val_s, session_range),
        }

        # No shift_features here: features are already from the PREVIOUS session (no lookahead).
        features_df = pd.DataFrame(features, index=df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return [
            "vp_poc_dist",
            "vp_vah_dist",
            "vp_val_dist",
            "vp_inside_va",
            "vp_poc_rel_pos",
            "vp_va_width_ratio",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "atr_period": 14,
            "n_levels": 50,
            "value_area_pct": 0.70,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR period for normalizing volume profile distances.",
                "min": 5,
                "max": 100,
                "step": 1,
            },
            "n_levels": {
                "type": "int",
                "default": 50,
                "description": "Number of price buckets for the volume histogram. Higher = more precise POC but slower computation.",
                "min": 20,
                "max": 200,
                "step": 10,
            },
            "value_area_pct": {
                "type": "float",
                "default": 0.70,
                "description": "Fraction of total session volume that defines the Value Area (standard: 70%). Expanding outward from the POC until this threshold is reached.",
                "min": 0.5,
                "max": 0.9,
                "step": 0.05,
            },
        }


__all__ = ["VolumeProfileIndicator"]
