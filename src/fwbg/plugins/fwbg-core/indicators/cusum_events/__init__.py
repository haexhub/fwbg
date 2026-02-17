"""
CUSUM Structural Break Detection (de Prado, AFML Ch. 2).

Detects structural breaks in the price process using the symmetric CUSUM
filter. Instead of hard-filtering bars, produces ML features that encode:
- Whether a positive/negative structural break occurred
- The magnitude of the cumulative deviation
- Time since the last event

The ML model learns when these events are predictive in combination with
other indicators, rather than us imposing hard rules.
"""
from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, shift_features, register_indicator


def _compute_cusum(
    returns: np.ndarray,
    expected: np.ndarray,
    threshold: np.ndarray,
) -> tuple:
    """Symmetric CUSUM filter (de Prado AFML Ch. 2).

    Tracks positive and negative cumulative deviations of returns from
    their expected value. Fires an event when the cumulative sum exceeds
    the threshold, then resets.

    Args:
        returns: Log returns array.
        expected: Expected return per bar (rolling mean).
        threshold: Per-bar threshold (h * rolling_std).

    Returns:
        (pos_events, neg_events, pos_values, neg_values, intensity, bars_since)
    """
    n = len(returns)
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    pos_events = np.zeros(n)
    neg_events = np.zeros(n)
    intensity = np.zeros(n)
    bars_since = np.zeros(n)

    last_event_bar = -1

    for i in range(1, n):
        deviation = returns[i] - expected[i]

        # Accumulate positive/negative deviations
        s_pos[i] = max(0.0, s_pos[i - 1] + deviation)
        s_neg[i] = min(0.0, s_neg[i - 1] + deviation)

        h = threshold[i]

        # Positive structural break
        if s_pos[i] > h and h > 0:
            pos_events[i] = 1.0
            intensity[i] = s_pos[i] / h  # overshoot ratio
            s_pos[i] = 0.0  # reset
            last_event_bar = i

        # Negative structural break
        if s_neg[i] < -h and h > 0:
            neg_events[i] = 1.0
            intensity[i] = abs(s_neg[i]) / h  # overshoot ratio
            s_neg[i] = 0.0  # reset
            last_event_bar = i

        # Track bars since last event
        if last_event_bar >= 0:
            bars_since[i] = i - last_event_bar

    return pos_events, neg_events, s_pos, s_neg, intensity, bars_since


@register_indicator("cusum_events")
class CusumEventsIndicator(BaseIndicator):
    """CUSUM structural break features for ML trading."""

    name = "cusum_events"
    version = "1.0.0"

    _FEATURES = [
        "cusum_pos_event",
        "cusum_neg_event",
        "cusum_pos_value",
        "cusum_neg_value",
        "cusum_intensity",
        "cusum_bars_since",
    ]

    def compute(
        self,
        df: pd.DataFrame,
        threshold: float = 1.5,
        lookback: int = 100,
        **params,
    ) -> pd.DataFrame:
        close = df["C"].values
        log_returns = np.diff(np.log(close), prepend=np.log(close[0]))

        # Rolling expected return and volatility
        ret_series = pd.Series(log_returns)
        expected = ret_series.rolling(lookback, min_periods=20).mean().values
        rolling_std = ret_series.rolling(lookback, min_periods=20).std().values

        # Fill NaN warmup with global estimates
        global_mean = np.nanmean(log_returns[1:])
        global_std = np.nanstd(log_returns[1:])
        expected = np.where(np.isnan(expected), global_mean, expected)
        rolling_std = np.where(np.isnan(rolling_std), global_std, rolling_std)

        # Threshold = h * rolling_std
        h = threshold * rolling_std

        pos_events, neg_events, s_pos, s_neg, intensity, bars_since = (
            _compute_cusum(log_returns, expected, h)
        )

        # Normalize cumulative values by threshold for scale-independence
        with np.errstate(divide="ignore", invalid="ignore"):
            pos_norm = np.where(h > 0, s_pos / h, 0.0)
            neg_norm = np.where(h > 0, s_neg / -h, 0.0)  # flip sign: 0..1

        # Normalize bars_since by lookback for scale-independence
        bars_norm = bars_since / lookback

        features = {
            "cusum_pos_event": pos_events,
            "cusum_neg_event": neg_events,
            "cusum_pos_value": pos_norm,
            "cusum_neg_value": neg_norm,
            "cusum_intensity": intensity,
            "cusum_bars_since": bars_norm,
        }

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return self._FEATURES

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "threshold": 1.5,
            "lookback": 100,
        }


__all__ = ["CusumEventsIndicator"]
