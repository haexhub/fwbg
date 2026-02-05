"""
Regime Indicator Plugin.

Enthält:
- Hurst-Exponent (Trending vs Mean-Reverting Detection)
- Regime-Filter Features

Interpretation Hurst:
- H > 0.5: Trending/Persistent (gut für Trend-Following)
- H = 0.5: Random Walk (schwierig zu traden)
- H < 0.5: Mean-Reverting (gut für Mean-Reversion)
"""
from typing import List
import numpy as np
import pandas as pd

from fwbg.plugins import BaseIndicator
from fwbg.core import register_indicator


def _compute_hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    """
    Berechnet den Hurst-Exponenten mittels R/S (Rescaled Range) Analyse.
    """
    if len(series) < max_lag * 2:
        return 0.5

    # Log-Returns für bessere Skalierung
    returns = np.diff(np.log(series + 1e-10))
    returns = returns[~np.isnan(returns)]

    if len(returns) < max_lag:
        return 0.5

    lags = range(10, min(max_lag, len(returns) // 4))
    rs_values = []
    lag_values = []

    for lag in lags:
        n_subseries = len(returns) // lag
        if n_subseries < 2:
            continue

        rs_lag = []
        for i in range(n_subseries):
            subseries = returns[i * lag:(i + 1) * lag]
            if len(subseries) < 2:
                continue

            mean_val = np.mean(subseries)
            cumdev = np.cumsum(subseries - mean_val)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(subseries, ddof=1)

            if s > 1e-10:
                rs_lag.append(r / s)

        if rs_lag:
            rs_values.append(np.mean(rs_lag))
            lag_values.append(lag)

    if len(lag_values) < 3:
        return 0.5

    log_lags = np.log(lag_values)
    log_rs = np.log(rs_values)
    slope, _ = np.polyfit(log_lags, log_rs, 1)

    return float(np.clip(slope, 0.0, 1.0))


def _compute_rolling_hurst(
    series: np.ndarray,
    window: int = 100,
    step: int = 10
) -> np.ndarray:
    """Berechnet Rolling Hurst-Exponent."""
    result = np.full(len(series), np.nan)

    for i in range(window, len(series), step):
        window_data = series[i - window:i]
        h = _compute_hurst_exponent(window_data, max_lag=min(50, window // 4))
        end_idx = min(i + step, len(series))
        result[i:end_idx] = h

    # Forward-fill für Lücken
    for i in range(1, len(result)):
        if np.isnan(result[i]) and not np.isnan(result[i - 1]):
            result[i] = result[i - 1]

    return result


@register_indicator("regime")
class RegimeIndicators(BaseIndicator):
    """
    Regime-Detection Features.

    Features:
    - Hurst-Exponent (100, 200, 500 Fenster)
    - Hurst-Änderung (Regime-Shift Detection)
    - Hurst-Divergenz (kurzfristig vs langfristig)
    """

    group = "regime"

    def compute(
        self,
        df: pd.DataFrame,
        hurst_windows: List[int] = None,
        step: int = 10,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet Regime-Features.

        Args:
            df: DataFrame mit OHLC-Daten
            hurst_windows: Fenstergrößen für Hurst (default: [100, 200, 500])
            step: Schrittgröße für Rolling-Berechnung

        Returns:
            DataFrame mit Regime-Features
        """
        if hurst_windows is None:
            hurst_windows = [100, 200, 500]

        features = {}

        # Verwende Original-Close falls Frac-Diff aktiv
        close_for_hurst = (
            df["_original_close"].values
            if "_original_close" in df.columns
            else df["C"].values
        )

        # Hurst für verschiedene Fenstergrößen
        for window in hurst_windows:
            hurst_values = _compute_rolling_hurst(close_for_hurst, window=window, step=step)
            features[f"regime_hurst_{window}"] = hurst_values

        # Hurst-Änderung (Regime-Shift Detection)
        if 100 in hurst_windows:
            hurst_100 = pd.Series(features["regime_hurst_100"], index=df.index)
            features["regime_hurst_100_chg"] = hurst_100 - hurst_100.shift(24)

        if 200 in hurst_windows:
            hurst_200 = pd.Series(features["regime_hurst_200"], index=df.index)
            features["regime_hurst_200_chg"] = hurst_200 - hurst_200.shift(48)

        # Hurst-Divergenz zwischen Zeitskalen
        if 100 in hurst_windows and 500 in hurst_windows:
            features["regime_hurst_divergence"] = (
                pd.Series(features["regime_hurst_100"], index=df.index) -
                pd.Series(features["regime_hurst_500"], index=df.index)
            )

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = pd.DataFrame(features, index=df.index)
        for col in features_df.columns:
            features_df[col] = features_df[col].shift(1)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return [
            "regime_hurst_100", "regime_hurst_200", "regime_hurst_500",
            "regime_hurst_100_chg", "regime_hurst_200_chg",
            "regime_hurst_divergence",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "hurst_windows": [100, 200, 500],
            "step": 10,
        }


__all__ = ["RegimeIndicators"]
