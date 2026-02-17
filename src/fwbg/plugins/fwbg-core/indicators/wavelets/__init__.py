"""
Wavelet Decomposition Features for ML Trading.

Uses the Discrete Wavelet Transform (DWT) to decompose log returns into
frequency bands. Unlike FFT, wavelets are localized in both time and
frequency — ideal for non-stationary financial signals.

Each decomposition level captures a different frequency band:
- Detail level 1: highest frequency (noise, microstructure)
- Detail level 2: medium-high frequency
- Detail level N: lowest detail frequency
- Approximation: trend/lowest frequency component

Features encode energy distribution across frequency bands, enabling the
ML model to distinguish trending vs choppy market regimes.
"""
from typing import List

import numpy as np
import pandas as pd
import pywt

from fwbg_sdk import BaseIndicator, shift_features, safe_divide, register_indicator


def _reconstruct_level(
    coeffs: list,
    level_idx: int,
    wavelet: str,
) -> np.ndarray:
    """Reconstruct a single decomposition level to full signal length.

    Zeros out all coefficient arrays except the target level, then
    applies inverse DWT to get the time-domain contribution of that level.

    Args:
        coeffs: Full coefficient list from pywt.wavedec [cA, cD_n, ..., cD_1].
        level_idx: Index into coeffs (0 = approx, 1..N = detail levels).
        wavelet: Wavelet name for reconstruction.

    Returns:
        Reconstructed signal array at the original signal length.
    """
    zeroed = [np.zeros_like(c) for c in coeffs]
    zeroed[level_idx] = coeffs[level_idx]
    return pywt.waverec(zeroed, wavelet)


def _rolling_energy(signal: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum-of-squares (energy) over a window."""
    sq = signal ** 2
    s = pd.Series(sq)
    return s.rolling(window, min_periods=1).mean().values


def _rolling_mean(signal: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean over a window."""
    s = pd.Series(signal)
    return s.rolling(window, min_periods=1).mean().values


@register_indicator("wavelets")
class WaveletsIndicator(BaseIndicator):
    """Wavelet decomposition features for ML trading."""

    name = "wavelets"
    version = "1.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        wavelet: str = "db4",
        levels: int = 3,
        windows: List[int] = None,
        **params,
    ) -> pd.DataFrame:
        """Compute wavelet decomposition features from close prices.

        Args:
            df: DataFrame with OHLC data (columns: O, H, L, C).
            wavelet: Wavelet family (default: 'db4').
            levels: Number of decomposition levels (default: 3).
            windows: Rolling windows for energy/mean stats (default: [10, 20, 50]).

        Returns:
            DataFrame with original columns plus wavelet features.
        """
        if windows is None:
            windows = [10, 20, 50]

        close = df["C"].values
        log_returns = np.diff(np.log(close), prepend=np.log(close[0]))

        # Decompose log returns via DWT
        # pywt.wavedec returns [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        # coeffs[0] = approximation (lowest freq)
        # coeffs[1] = cD_n (coarsest detail)
        # coeffs[-1] = cD_1 (finest detail, highest freq)
        coeffs = pywt.wavedec(log_returns, wavelet, level=levels)

        # Reconstruct each level to full length.
        # Map detail_1 = highest freq (cD_1 = coeffs[levels]),
        #      detail_N = lowest freq  (cD_N = coeffs[1]).
        n = len(log_returns)
        detail_signals = {}
        for lvl in range(1, levels + 1):
            # detail level lvl -> coeffs index (levels + 1 - lvl)
            coeffs_idx = levels + 1 - lvl
            rec = _reconstruct_level(coeffs, coeffs_idx, wavelet)
            detail_signals[lvl] = rec[:n]

        approx_signal = _reconstruct_level(coeffs, 0, wavelet)[:n]

        # Build features
        features = {}

        for w in windows:
            # Approximation energy
            features[f"wt_approx_energy_{w}"] = _rolling_energy(approx_signal, w)

            for lvl in range(1, levels + 1):
                sig = detail_signals[lvl]
                features[f"wt_detail_{lvl}_energy_{w}"] = _rolling_energy(sig, w)
                features[f"wt_detail_{lvl}_mean_{w}"] = _rolling_mean(sig, w)

        # Ratio features (window-independent, use instantaneous squared values
        # smoothed over a moderate window)
        ratio_window = min(20, n)
        total_energy = _rolling_energy(approx_signal, ratio_window).copy()
        detail_energies = {}
        for lvl in range(1, levels + 1):
            e = _rolling_energy(detail_signals[lvl], ratio_window)
            detail_energies[lvl] = e
            total_energy = total_energy + e

        for lvl in range(1, levels + 1):
            features[f"wt_detail_ratio_{lvl}"] = safe_divide(
                detail_energies[lvl], total_energy
            )

        # High-frequency to low-frequency ratio
        for w in windows:
            hf_energy = _rolling_energy(detail_signals[1], w)
            lf_energy = _rolling_energy(detail_signals[levels], w)
            features[f"wt_high_freq_ratio_{w}"] = safe_divide(hf_energy, lf_energy)

        # Cache feature columns for get_feature_columns
        self._feature_columns = sorted(features.keys())

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        if self._feature_columns:
            return self._feature_columns
        # Return default feature names for default params
        return self._build_feature_names(levels=3, windows=[10, 20, 50])

    @staticmethod
    def _build_feature_names(levels: int, windows: List[int]) -> List[str]:
        """Build expected feature column names for given params."""
        names = []
        for w in windows:
            names.append(f"wt_approx_energy_{w}")
            for lvl in range(1, levels + 1):
                names.append(f"wt_detail_{lvl}_energy_{w}")
                names.append(f"wt_detail_{lvl}_mean_{w}")
        for lvl in range(1, levels + 1):
            names.append(f"wt_detail_ratio_{lvl}")
        for w in windows:
            names.append(f"wt_high_freq_ratio_{w}")
        return sorted(names)

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "wavelet": "db4",
            "levels": 3,
            "windows": [10, 20, 50],
        }


__all__ = ["WaveletsIndicator"]
