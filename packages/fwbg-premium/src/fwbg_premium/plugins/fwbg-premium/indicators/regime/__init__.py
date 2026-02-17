"""
Regime Indicator Plugin.

Enthält:
- Hurst-Exponent (Trending vs Mean-Reverting Detection)
- Shannon Entropy (Markt-Komplexität/Vorhersagbarkeit)
- Variance Ratio (Random Walk Test)

Interpretation:
- Hurst > 0.5: Trending/Persistent, = 0.5: Random Walk, < 0.5: Mean-Reverting
- Entropy hoch: Komplexer/unvorhersagbarer Markt, niedrig: vorhersagbar
- VR > 1: Momentum, = 1: Random Walk, < 1: Mean-Reversion
"""
from typing import List
import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, shift_features, EPSILON, register_indicator


def _compute_hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    """
    Berechnet den Hurst-Exponenten mittels R/S (Rescaled Range) Analyse.
    """
    if len(series) < max_lag * 2:
        return 0.5

    # Log-Returns für bessere Skalierung
    returns = np.diff(np.log(series + EPSILON))
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

            if s > EPSILON:
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


def _compute_shannon_entropy(returns: np.ndarray, n_bins: int = 10) -> float:
    """
    Shannon Entropy der Return-Verteilung.

    Hohe Entropy = hohe Unsicherheit/Komplexität (schwer vorhersagbar).
    Niedrige Entropy = vorhersagbares Muster (gut für ML).
    """
    returns = returns[~np.isnan(returns)]
    if len(returns) < n_bins:
        return np.nan

    counts, _ = np.histogram(returns, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _compute_rolling_entropy(
    returns: np.ndarray,
    window: int,
    n_bins: int = 10,
    step: int = 1,
) -> np.ndarray:
    """Rolling Shannon Entropy über Returns."""
    result = np.full(len(returns), np.nan)

    for i in range(window, len(returns), step):
        window_data = returns[i - window:i]
        result[i] = _compute_shannon_entropy(window_data, n_bins)

    # Forward-fill bei step > 1
    if step > 1:
        for i in range(1, len(result)):
            if np.isnan(result[i]) and not np.isnan(result[i - 1]):
                result[i] = result[i - 1]

    return result


def _compute_variance_ratio(returns: np.ndarray, q: int) -> float:
    """
    Lo & MacKinlay Variance Ratio Test.

    VR(q) = Var(r_q) / (q * Var(r_1))

    VR > 1: Momentum/Trending (positive Autokorrelation)
    VR = 1: Random Walk
    VR < 1: Mean-Reverting (negative Autokorrelation)
    """
    returns = returns[~np.isnan(returns)]
    if len(returns) < q * 2:
        return np.nan

    var_1 = np.var(returns, ddof=1)
    if var_1 < EPSILON:
        return np.nan

    # q-period returns
    q_returns = np.array([
        returns[i:i + q].sum()
        for i in range(len(returns) - q + 1)
    ])
    var_q = np.var(q_returns, ddof=1)

    return float(var_q / (q * var_1))


def _compute_rolling_variance_ratio(
    returns: np.ndarray,
    window: int,
    q: int,
    step: int = 1,
) -> np.ndarray:
    """Rolling Variance Ratio."""
    result = np.full(len(returns), np.nan)

    for i in range(window, len(returns), step):
        window_data = returns[i - window:i]
        result[i] = _compute_variance_ratio(window_data, q)

    # Forward-fill bei step > 1
    if step > 1:
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
    - Shannon Entropy (Markt-Komplexität)
    - Variance Ratio (Random Walk Test)
    """

    name = "regime"
    version = "3.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        hurst_windows: List[int] = None,
        step: int = 10,
        entropy_windows: List[int] = None,
        vr_windows: List[int] = None,
        vr_lags: List[int] = None,
        **params
    ) -> pd.DataFrame:
        """
        Berechnet Regime-Features.

        Args:
            df: DataFrame mit OHLC-Daten
            hurst_windows: Fenstergrößen für Hurst (default: [100, 200, 500])
            step: Schrittgröße für Rolling-Berechnung
            entropy_windows: Fenstergrößen für Shannon Entropy (default: [50, 100])
            vr_windows: Fenstergrößen für Variance Ratio (default: [100, 200])
            vr_lags: Lag-Perioden für VR (default: [5, 10])
        """
        if hurst_windows is None:
            hurst_windows = [100, 200, 500]
        if entropy_windows is None:
            entropy_windows = [50, 100]
        if vr_windows is None:
            vr_windows = [100, 200]
        if vr_lags is None:
            vr_lags = [5, 10]

        features = {}

        # Verwende Original-Close falls Frac-Diff aktiv
        close_for_hurst = (
            df["_original_close"].values
            if "_original_close" in df.columns
            else df["C"].values
        )

        # === Hurst Exponent ===
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

        # === Shannon Entropy ===
        log_returns = np.diff(np.log(close_for_hurst + EPSILON))
        log_returns = np.concatenate([[np.nan], log_returns])

        for window in entropy_windows:
            entropy = _compute_rolling_entropy(log_returns, window, step=step)
            features[f"regime_entropy_{window}"] = entropy

        # Entropy Change: fallende Entropy = zunehmende Vorhersagbarkeit
        if 100 in entropy_windows:
            ent_100 = pd.Series(features["regime_entropy_100"], index=df.index)
            features["regime_entropy_100_chg"] = ent_100 - ent_100.shift(24)

        # === Variance Ratio ===
        for window in vr_windows:
            for q in vr_lags:
                vr = _compute_rolling_variance_ratio(log_returns, window, q, step=step)
                features[f"regime_vr_{window}_{q}"] = vr

        # VR Deviation from 1 (= Abweichung von Random Walk)
        if 100 in vr_windows and 5 in vr_lags:
            vr_100_5 = pd.Series(features["regime_vr_100_5"], index=df.index)
            features["regime_vr_deviation"] = vr_100_5 - 1.0

        # CRITICAL: Shift all features by 1 to prevent lookahead bias
        features_df = shift_features(features, df.index)

        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return [
            # Hurst
            "regime_hurst_100", "regime_hurst_200", "regime_hurst_500",
            "regime_hurst_100_chg", "regime_hurst_200_chg",
            "regime_hurst_divergence",
            # Shannon Entropy
            "regime_entropy_50", "regime_entropy_100",
            "regime_entropy_100_chg",
            # Variance Ratio
            "regime_vr_100_5", "regime_vr_100_10",
            "regime_vr_200_5", "regime_vr_200_10",
            "regime_vr_deviation",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "hurst_windows": [100, 200, 500],
            "step": 10,
            "entropy_windows": [50, 100],
            "vr_windows": [100, 200],
            "vr_lags": [5, 10],
        }


__all__ = ["RegimeIndicators"]
