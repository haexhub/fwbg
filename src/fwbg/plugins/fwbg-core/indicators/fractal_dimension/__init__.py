"""
Higuchi Fractal Dimension (HFD) indicator plugin.

Measures the complexity and roughness of a price series using the Higuchi
algorithm. The fractal dimension D ranges from 1.0 (smooth/trending) to
2.0 (random/complex), complementing the Hurst exponent.

Algorithm:
  For a time series X of length N, with k_max intervals:
  1. For each k in 1..k_max, construct k sub-series starting at m=1..k
  2. Compute the length L_m(k) of each sub-series
  3. Average over m: L(k) = mean(L_m(k))
  4. D = slope of log(L(k)) vs log(1/k)
"""

from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, shift_features, register_indicator


def _higuchi_fd(series: np.ndarray, k_max: int) -> float:
    """Compute the Higuchi Fractal Dimension of a 1-D time series.

    Args:
        series: 1-D array of values (e.g. close prices).
        k_max: Maximum interval parameter. Higher = more accurate but slower.

    Returns:
        Fractal dimension estimate (typically between 1.0 and 2.0).
    """
    n = len(series)
    if n < k_max + 1:
        return np.nan

    # Clamp k_max so we always have at least 2 points per sub-series
    k_max = min(k_max, n - 1)

    lk = np.zeros(k_max)

    for k in range(1, k_max + 1):
        lengths = np.zeros(k)
        for m in range(1, k + 1):
            # Sub-series indices: m-1, m-1+k, m-1+2k, ...
            idx = np.arange(m - 1, n, k)
            if len(idx) < 2:
                lengths[m - 1] = 0.0
                continue
            sub = series[idx]
            # Sum of absolute differences
            abs_diffs = np.abs(np.diff(sub))
            num_segments = len(idx) - 1
            # Normalisation factor: (N-1) / (num_segments * k)
            norm = (n - 1) / (num_segments * k)
            lengths[m - 1] = abs_diffs.sum() * norm / k
        lk[k - 1] = np.mean(lengths)

    # Fit log(L(k)) vs log(1/k) via least-squares
    ks = np.arange(1, k_max + 1, dtype=np.float64)
    valid = lk > 0
    if valid.sum() < 2:
        return np.nan

    x = np.log(1.0 / ks[valid])
    y = np.log(lk[valid])

    # Linear regression slope
    n_pts = len(x)
    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    denom = n_pts * sxx - sx * sx
    if abs(denom) < 1e-15:
        return np.nan
    slope = (n_pts * sxy - sx * sy) / denom
    return slope


def _rolling_higuchi(close: np.ndarray, window: int, k_max: int) -> np.ndarray:
    """Compute Higuchi FD over a rolling window.

    Args:
        close: 1-D price array.
        window: Rolling window size.
        k_max: Higuchi k_max parameter.

    Returns:
        Array of same length as close, with NaN for warmup period.
    """
    n = len(close)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        segment = close[i - window + 1 : i + 1]
        result[i] = _higuchi_fd(segment, k_max)
    return result


# Higuchi FD of a pure random walk is unbiased around 1.5 but noisy: Monte
# Carlo over independent Gaussian random walks gives a standard deviation of
# roughly _SD_COEF * k_max**_SD_EXP / sqrt(window) (e.g. 0.13 for window=50,
# k_max=10 and 0.04 for window=400). A fixed band such as 1.4..1.6 therefore
# labels about 45 % of random-walk windows at window=50 as trending or
# mean-reverting.
_RW_FD = 1.5
_SD_COEF = 0.486
_SD_EXP = 0.27


def rw_fd_std(window: int, k_max: int) -> float:
    """Approximate std-dev of the Higuchi FD estimate for a random walk.

    Calibrated by simulation for window >= 50 and 5 <= k_max <= 20; used to size
    the dead band around FD = 1.5 in which the regime is "random".
    """
    k_eff = max(2, min(k_max, window - 1))
    return _SD_COEF * k_eff**_SD_EXP / np.sqrt(window)


@register_indicator("fractal_dimension")
class FractalDimensionIndicator(BaseIndicator):
    """Higuchi Fractal Dimension features for ML trading."""

    name = "fractal_dimension"
    version = "1.1.0"

    def compute(
        self,
        df: pd.DataFrame,
        windows: List[int] | None = None,
        k_max: int = 10,
        regime_z: float = 1.96,
        **params,
    ) -> pd.DataFrame:
        if windows is None:
            windows = [50, 100, 200]

        close = df["C"].values
        features: dict = {}

        for w in windows:
            fd = _rolling_higuchi(close, w, k_max)

            # Raw Higuchi FD
            features[f"fd_higuchi_{w}"] = fd

            # Change in FD vs previous window (regime transition detection)
            fd_series = pd.Series(fd)
            fd_change = fd_series.diff(w).values
            features[f"fd_higuchi_change_{w}"] = fd_change

            # Complexity ratio: how far from 1.5 (random walk)
            # Values near 0 = random, near 1 = highly structured
            deviation = np.abs(fd - 1.5)
            features[f"fd_complexity_ratio_{w}"] = deviation * 2.0

            # Regime: -1 (trending, FD below the band), 0 (indistinguishable
            # from a random walk), 1 (mean-reverting, FD above the band). The
            # dead band is regime_z estimator standard deviations wide around
            # FD = 1.5, so a pure random walk is labelled 0 in ~95 % of windows
            # for the default regime_z=1.96.
            band = regime_z * rw_fd_std(w, k_max)
            regime = np.full_like(fd, np.nan)
            valid = ~np.isnan(fd)
            regime[valid & (fd < _RW_FD - band)] = -1.0
            regime[valid & (fd >= _RW_FD - band) & (fd <= _RW_FD + band)] = 0.0
            regime[valid & (fd > _RW_FD + band)] = 1.0
            features[f"fd_regime_{w}"] = regime

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self, params=None) -> List[str]:
        windows = {**self.get_default_params(), **(params or {})}["windows"]
        cols = []
        for w in windows:
            cols.extend(
                [
                    f"fd_higuchi_{w}",
                    f"fd_higuchi_change_{w}",
                    f"fd_complexity_ratio_{w}",
                    f"fd_regime_{w}",
                ]
            )
        return cols

    def get_signal_columns(self, params=None) -> List[str]:
        windows = {**self.get_default_params(), **(params or {})}["windows"]
        return [f"fd_regime_{w}" for w in windows]

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "windows": [50, 100, 200],
            "k_max": 10,
            "regime_z": 1.96,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "windows": {
                "type": "list[int]",
                "default": [50, 100, 200],
                "description": "Rolling window sizes for Higuchi Fractal Dimension computation. The fractal dimension (1.0 = smooth/trending, 2.0 = random/complex) is calculated over each window. Shorter windows capture local regime transitions, longer windows provide a more stable complexity measure.",
                "min": 20,
                "max": 1000,
            },
            "k_max": {
                "type": "int",
                "default": 10,
                "description": "Maximum interval parameter for the Higuchi algorithm. Controls the number of sub-series scales used to estimate the fractal dimension. Higher values improve accuracy but increase computation time. Must be less than the smallest window size.",
                "min": 2,
                "max": 50,
                "step": 1,
            },
            "regime_z": {
                "type": "float",
                "default": 1.96,
                "description": "Half-width of the 'random' dead band around FD = 1.5 for fd_regime_*, in standard deviations of the Higuchi estimator on a random walk (which shrink with window size). 1.96 labels ~95% of pure random-walk windows as 0; lower values make the regime flag more sensitive and noisier.",
                "min": 0.5,
                "max": 4.0,
                "step": 0.05,
            },
        }


__all__ = ["FractalDimensionIndicator"]
