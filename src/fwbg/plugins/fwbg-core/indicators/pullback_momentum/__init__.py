"""
Trend + Pullback + Momentum Break Indicator.

Implements a three-step entry signal for trending markets:

  1. Trend filter:    Close > EMA(ema_period) for longs, Close < EMA for shorts.
  2. BOS:             Close breaks the rolling swing high (bullish BOS) or swing
                      low (bearish BOS), confirming the impulse.
  3. Pullback + entry: Price retraces at least min_pullback_pct of the impulse
                       range, then breaks the last local Lower High (long) or
                       local Higher Low (short) — the momentum entry trigger.

The structural stop is placed below the pullback low (long) or above the
pullback high (short), with an ATR buffer for noise tolerance.

Signal columns produced:
  tpm_entry_long / tpm_entry_short  — 1 at the entry bar
  tpm_sl_dist_long / _short         — SL distance in price units at entry bars

Feature columns useful for ML context:
  tpm_pullback_pct      — current retracement depth (0–2 range)
  tpm_in_pullback_long  — 1 while a valid long setup is being tracked
  tpm_in_pullback_short — 1 while a valid short setup is being tracked
  tpm_trend_ok_long     — 1 when close > EMA
  tpm_trend_ok_short    — 1 when close < EMA
"""
from typing import List

import numpy as np
import pandas as pd
import ta

from fwbg_sdk import BaseIndicator, EPSILON, register_indicator, shift_features


@register_indicator("pullback_momentum")
class PullbackMomentumIndicator(BaseIndicator):
    """Trend + Pullback + Momentum Break entry signals (Long and Short)."""

    name = "pullback_momentum"
    version = "1.0.0"

    _FEATURES = [
        "tpm_entry_long",
        "tpm_entry_short",
        "tpm_sl_dist_long",
        "tpm_sl_dist_short",
        "tpm_pullback_pct",
        "tpm_in_pullback_long",
        "tpm_in_pullback_short",
        "tpm_trend_ok_long",
        "tpm_trend_ok_short",
    ]

    def compute(
        self,
        df: pd.DataFrame,
        ema_period: int = 200,
        swing_lookback: int = 20,
        min_pullback_pct: float = 0.382,
        local_high_lookback: int = 3,
        sl_buffer_mult: float = 0.25,
        atr_period: int = 14,
        **params,
    ) -> pd.DataFrame:
        n = len(df)
        highs = df["H"].values.astype(np.float64)
        lows = df["L"].values.astype(np.float64)
        closes = df["C"].values.astype(np.float64)

        # EMA for trend filter
        ema = ta.trend.ema_indicator(pd.Series(closes), window=ema_period).values

        # ATR for SL buffer (handles edge case of atr=0 bars)
        atr = ta.volatility.average_true_range(
            pd.Series(highs), pd.Series(lows), pd.Series(closes), window=atr_period
        ).values
        atr = np.nan_to_num(atr, nan=0.0)

        # Rolling swing extremes — no lookahead (window covers only past bars)
        swing_high = pd.Series(highs).rolling(swing_lookback, min_periods=1).max().values
        swing_low = pd.Series(lows).rolling(swing_lookback, min_periods=1).min().values

        # Output arrays
        entry_long = np.zeros(n)
        entry_short = np.zeros(n)
        sl_dist_long = np.full(n, np.nan)
        sl_dist_short = np.full(n, np.nan)
        pullback_pct = np.zeros(n)
        in_pullback_long = np.zeros(n)
        in_pullback_short = np.zeros(n)
        trend_ok_long = np.zeros(n)
        trend_ok_short = np.zeros(n)

        # ── Long state machine ────────────────────────────────────────────────
        long_state = 0      # 0 = SCANNING, 1 = PULLBACK
        imp_high_l = 0.0    # price of the broken swing high (impulse top)
        imp_low_l = 0.0     # rolling swing low at BOS time (impulse base)
        pb_low_l = np.inf   # running minimum low during pullback

        # ── Short state machine ───────────────────────────────────────────────
        short_state = 0     # 0 = SCANNING, 1 = PULLBACK
        imp_high_s = 0.0    # rolling swing high at BOS time (impulse top)
        imp_low_s = 0.0     # price of the broken swing low (impulse base)
        pb_high_s = -np.inf # running maximum high during pullback

        for i in range(1, n):
            c = closes[i]
            h = highs[i]
            l = lows[i]
            ema_i = ema[i]
            atr_i = atr[i] if atr[i] > EPSILON else 1e-5

            # Previous rolling extremes (strictly no lookahead)
            prev_sh = swing_high[i - 1]
            prev_sl = swing_low[i - 1]

            ema_valid = not np.isnan(ema_i)
            trend_ok_long[i] = 1.0 if (ema_valid and c > ema_i) else 0.0
            trend_ok_short[i] = 1.0 if (ema_valid and c < ema_i) else 0.0

            # ── LONG ──────────────────────────────────────────────────────────
            if long_state == 0:  # SCANNING
                if ema_valid and c > prev_sh and c > ema_i:
                    long_state = 1
                    imp_high_l = prev_sh
                    imp_low_l = prev_sl
                    pb_low_l = l

            else:  # PULLBACK
                if c < imp_low_l:
                    # Setup failed: price fell below the impulse swing low
                    long_state = 0

                elif ema_valid and c > prev_sh and c > ema_i:
                    # New BOS while waiting for pullback → refresh impulse
                    imp_high_l = prev_sh
                    imp_low_l = prev_sl
                    pb_low_l = l

                else:
                    if l < pb_low_l:
                        pb_low_l = l

                    impulse_range = imp_high_l - imp_low_l
                    pb = 0.0
                    if impulse_range > EPSILON:
                        pb = (imp_high_l - c) / impulse_range

                    in_pullback_long[i] = 1.0

                    # Local lower high: max of highs in the prior local_high_lookback bars
                    lh_start = max(0, i - local_high_lookback)
                    local_lh = np.max(highs[lh_start:i]) if lh_start < i else imp_high_l

                    if pb >= min_pullback_pct and ema_valid and c > ema_i and c > local_lh:
                        entry_long[i] = 1.0
                        sl_d = (c - pb_low_l) + atr_i * sl_buffer_mult
                        sl_dist_long[i] = sl_d if sl_d > EPSILON else atr_i * 0.5
                        long_state = 0
                        in_pullback_long[i] = 0.0
                    else:
                        pullback_pct[i] = max(pullback_pct[i], min(pb, 2.0))

            # ── SHORT ─────────────────────────────────────────────────────────
            if short_state == 0:  # SCANNING
                if ema_valid and c < prev_sl and c < ema_i:
                    short_state = 1
                    imp_high_s = prev_sh
                    imp_low_s = prev_sl
                    pb_high_s = h

            else:  # PULLBACK
                if c > imp_high_s:
                    # Setup failed: price rose above the impulse swing high
                    short_state = 0

                elif ema_valid and c < prev_sl and c < ema_i:
                    # New BOS down while waiting for pullback → refresh impulse
                    imp_high_s = prev_sh
                    imp_low_s = prev_sl
                    pb_high_s = h

                else:
                    if h > pb_high_s:
                        pb_high_s = h

                    impulse_range = imp_high_s - imp_low_s
                    pb = 0.0
                    if impulse_range > EPSILON:
                        pb = (c - imp_low_s) / impulse_range

                    in_pullback_short[i] = 1.0

                    # Local higher low: min of lows in the prior local_high_lookback bars
                    hl_start = max(0, i - local_high_lookback)
                    local_hl = np.min(lows[hl_start:i]) if hl_start < i else imp_low_s

                    if pb >= min_pullback_pct and ema_valid and c < ema_i and c < local_hl:
                        entry_short[i] = 1.0
                        sl_d = (pb_high_s - c) + atr_i * sl_buffer_mult
                        sl_dist_short[i] = sl_d if sl_d > EPSILON else atr_i * 0.5
                        short_state = 0
                        in_pullback_short[i] = 0.0
                    else:
                        pullback_pct[i] = max(pullback_pct[i], min(pb, 2.0))

        features = {
            "tpm_entry_long": entry_long,
            "tpm_entry_short": entry_short,
            "tpm_sl_dist_long": sl_dist_long,
            "tpm_sl_dist_short": sl_dist_short,
            "tpm_pullback_pct": pullback_pct,
            "tpm_in_pullback_long": in_pullback_long,
            "tpm_in_pullback_short": in_pullback_short,
            "tpm_trend_ok_long": trend_ok_long,
            "tpm_trend_ok_short": trend_ok_short,
        }

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    def get_signal_columns(self) -> List[str]:
        return ["tpm_entry_long", "tpm_entry_short"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "ema_period": 200,
            "swing_lookback": 20,
            "min_pullback_pct": 0.382,
            "local_high_lookback": 3,
            "sl_buffer_mult": 0.25,
            "atr_period": 14,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "ema_period": {
                "type": "int",
                "default": 200,
                "description": (
                    "EMA period for trend filter. Long entries require close > EMA, "
                    "short entries require close < EMA."
                ),
                "min": 20,
                "max": 500,
                "step": 10,
            },
            "swing_lookback": {
                "type": "int",
                "default": 20,
                "description": (
                    "Rolling window for detecting swing highs/lows. A BOS fires when "
                    "close breaks the rolling extreme of this window."
                ),
                "min": 5,
                "max": 100,
                "step": 5,
            },
            "min_pullback_pct": {
                "type": "float",
                "default": 0.382,
                "description": (
                    "Minimum pullback depth as a fraction of the impulse range. "
                    "0.382 = 38.2% Fibonacci retracement."
                ),
                "min": 0.2,
                "max": 0.8,
                "step": 0.05,
            },
            "local_high_lookback": {
                "type": "int",
                "default": 3,
                "description": (
                    "Bars to look back when identifying the local Lower High (long) "
                    "or Higher Low (short). Price must break this level to trigger entry."
                ),
                "min": 1,
                "max": 20,
                "step": 1,
            },
            "sl_buffer_mult": {
                "type": "float",
                "default": 0.25,
                "description": (
                    "ATR multiplier added to the structural SL distance as a noise buffer "
                    "below the pullback low (long) or above the pullback high (short)."
                ),
                "min": 0.0,
                "max": 2.0,
                "step": 0.25,
            },
            "atr_period": {
                "type": "int",
                "default": 14,
                "description": "ATR lookback period for computing the SL buffer.",
                "min": 5,
                "max": 50,
                "step": 1,
            },
        }


__all__ = ["PullbackMomentumIndicator"]
