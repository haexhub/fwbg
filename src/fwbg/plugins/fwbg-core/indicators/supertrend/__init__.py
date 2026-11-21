"""
Supertrend Indicator (Standard + Multi-Timeframe).

ATR-basierter Trend-Following-Indikator. Direction: +1 (Aufwärts) oder -1 (Abwärts).

Enthält zwei registrierte Indikatoren:
- "supertrend": Standard-Supertrend auf OHLC-Daten
- "supertrend_mtf": Supertrend auf abgeschlossenen, aggregierten Kerzen (z.B. D1)
"""

from typing import List

import numpy as np
import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, register_indicator, shift_features, safe_divide


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend Berechnung.

    Returns (direction, line):
    - direction: +1 (Aufwärts) oder -1 (Abwärts)
    - line: Supertrend-Preisniveau

    Beide Serien sind NaN, solange der ATR nicht gültig ist (die ersten
    ``period - 1`` Bars, bei weniger als ``period`` Bars komplett). Die Band-
    Rekursion startet erst auf der ersten Zeile mit gültigem ATR.
    """
    if period < 2:
        raise ValueError(f"Supertrend period must be >= 2, got {period}")

    n = len(close)
    direction = np.full(n, np.nan)
    supertrend_line = np.full(n, np.nan)
    start = period - 1

    if n > start:
        atr = ta.volatility.average_true_range(
            high, low, close, window=period
        ).to_numpy()
        hl2 = ((high + low) / 2).to_numpy()
        px = close.to_numpy()

        final_upper = hl2 + multiplier * atr
        final_lower = hl2 - multiplier * atr

        direction[start] = 1.0
        supertrend_line[start] = final_lower[start]

        for i in range(start + 1, n):
            if final_lower[i] < final_lower[i - 1] and px[i - 1] > final_lower[i - 1]:
                final_lower[i] = final_lower[i - 1]
            if final_upper[i] > final_upper[i - 1] and px[i - 1] < final_upper[i - 1]:
                final_upper[i] = final_upper[i - 1]

            if direction[i - 1] == 1:
                if px[i] < final_lower[i]:
                    direction[i] = -1
                    supertrend_line[i] = final_upper[i]
                else:
                    direction[i] = 1
                    supertrend_line[i] = final_lower[i]
            else:
                if px[i] > final_upper[i]:
                    direction[i] = 1
                    supertrend_line[i] = final_lower[i]
                else:
                    direction[i] = -1
                    supertrend_line[i] = final_upper[i]

    return (
        pd.Series(direction, index=close.index),
        pd.Series(supertrend_line, index=close.index),
    )


def _direction_flips(direction: pd.Series) -> pd.Series:
    """1.0 on a direction change, 0.0 otherwise, NaN until two valid bars exist."""
    previous = direction.shift(1)
    flips = (direction != previous).astype(float)
    return flips.where(direction.notna() & previous.notna())


@register_indicator("supertrend")
class SupertrendIndicator(BaseIndicator):
    """
    Standard-Supertrend auf OHLC-Daten.

    Features:
    - st_direction: Trend-Richtung (+1/-1)
    - st_flip: 1 wenn Richtungswechsel
    - _st_line: Supertrend-Preisniveau (Overlay)
    """

    name = "supertrend"
    version = "1.1.0"

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 14,
        multiplier: float = 3.0,
        **params,
    ) -> pd.DataFrame:
        direction, st_line = _supertrend(df["H"], df["L"], df["C"], period, multiplier)

        features = {
            "st_direction": direction,
            "st_flip": _direction_flips(direction),
            "_st_line": st_line,
        }

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        return ["st_direction", "st_flip"]

    def get_overlay_columns(self, params=None) -> List[str]:
        return ["_st_line"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"period": 14, "multiplier": 3.0}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "period": {
                "type": "int",
                "default": 14,
                "min": 2,
                "max": 500,
                "step": 1,
                "description": "ATR lookback period. Lower values = more responsive but noisier.",
            },
            "multiplier": {
                "type": "float",
                "default": 3.0,
                "min": 0.5,
                "max": 20.0,
                "step": 0.5,
                "description": "ATR multiplier for band width. Higher values = fewer trend flips.",
            },
        }

    def get_column_group_labels(self) -> dict:
        return {"st": "Supertrend"}


def _completed_bars(df: pd.DataFrame, span: pd.Timedelta) -> pd.DataFrame:
    """Aggregate OHLC into ``span`` buckets, available only once a bucket is over.

    Buckets are ``index.floor(span)`` (calendar-aligned for spans up to one day).
    The aggregated bar is placed on the last base bar of its bucket, which is the
    first moment its high, low and close are final. The still-open bucket at the
    end of the input is dropped, so a truncated input never reports a partial
    bar as finished.

    Returns a frame indexed by those last-base-bar timestamps with columns
    H, L, C.
    """
    bucket = df.index.floor(span)
    grouped = df.groupby(bucket, sort=True)
    bars = pd.DataFrame(
        {"H": grouped["H"].max(), "L": grouped["L"].min(), "C": grouped["C"].last()}
    )
    emit_at = pd.Series(df.index, index=df.index).groupby(bucket).last()
    bars.index = pd.DatetimeIndex(emit_at)
    return bars.iloc[:-1]


@register_indicator("supertrend_mtf")
class SupertrendMTFIndicator(BaseIndicator):
    """
    Supertrend auf einer höheren Zeitebene (z.B. D1) als Trend-Filter.

    Aggregiert die Basis-Bars zu abgeschlossenen Kerzen von ``d1_bars`` Basis-
    Bars Dauer (Bucket = ``Index.floor(d1_bars * Bar-Dauer)``, bis 1 Tag
    kalenderbündig), berechnet Supertrend/ATR auf diesen Kerzen und hält jeden
    Wert ab dem letzten Basis-Bar seiner Kerze (Forward-Fill). Eine noch laufende
    Kerze fließt nie ein. Die Bar-Dauer ist der Median der Index-Abstände.

    Bars pro Zeiteinheit (M15-Basis):
        H4 → d1_bars=16, D1 → d1_bars=96

    Wochenkerzen: ``d1_bars=480`` ist ein 5-Tage-Bucket relativ zur Epoche, kein
    Kalender-Wochenbucket. Für W1 besser ``indicator_timeframe`` mit echten
    W1-Daten verwenden.

    Features:
    - st_d1_direction: Richtung (+1/-1) der letzten abgeschlossenen Kerze
    - st_d1_dist_atr: (Close - ST-Level) / ATR der aggregierten Kerzen
    - _st_d1_line: Supertrend-Preisniveau der aggregierten Kerzen (Overlay)
    """

    name = "supertrend_mtf"
    version = "1.2.0"

    def compute(
        self,
        df: pd.DataFrame,
        period: int = 14,
        multiplier: float = 3.0,
        d1_bars: int = 96,
        **params,
    ) -> pd.DataFrame:
        if d1_bars < 2:
            raise ValueError(f"d1_bars must be >= 2, got {d1_bars}")

        nan = pd.Series(np.nan, index=df.index)
        direction, st_line, htf_atr = nan, nan, nan

        steps = df.index.to_series().diff().dropna()
        if len(steps) > 0 and steps.median() > pd.Timedelta(0):
            span = d1_bars * steps.median()
            bars = _completed_bars(df, span)
            if len(bars) >= period:
                htf_dir, htf_line = _supertrend(
                    bars["H"], bars["L"], bars["C"], period, multiplier
                )
                atr = ta.volatility.average_true_range(
                    bars["H"], bars["L"], bars["C"], window=period
                )
                atr.iloc[: period - 1] = np.nan
                # Place each completed-candle value on its last base bar, then hold it.
                direction, st_line, htf_atr = (
                    s.reindex(df.index).ffill() for s in (htf_dir, htf_line, atr)
                )

        st_dist_atr = safe_divide(df["C"] - st_line, htf_atr)

        features = {
            "st_d1_direction": direction,
            "st_d1_dist_atr": st_dist_atr,
            "_st_d1_line": st_line,
        }

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        return ["st_d1_direction", "st_d1_dist_atr"]

    def get_overlay_columns(self, params=None) -> List[str]:
        return ["_st_d1_line"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"period": 14, "multiplier": 3.0, "d1_bars": 96}

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "period": {
                "type": "int",
                "default": 14,
                "min": 5,
                "max": 50,
                "step": 1,
                "description": "ATR period for Supertrend.",
            },
            "multiplier": {
                "type": "float",
                "default": 3.0,
                "min": 1.0,
                "max": 6.0,
                "step": 0.5,
                "description": "ATR multiplier for band width.",
            },
            "d1_bars": {
                "type": "int",
                "default": 96,
                "min": 4,
                "max": 960,
                "step": 4,
                "description": "Number of base bars per aggregated candle. M15→D1: 96, M15→H4: 16, M15→W1: 480.",
            },
        }

    def get_column_group_labels(self) -> dict:
        return {"st": "Supertrend MTF"}


__all__ = ["SupertrendIndicator", "SupertrendMTFIndicator"]
