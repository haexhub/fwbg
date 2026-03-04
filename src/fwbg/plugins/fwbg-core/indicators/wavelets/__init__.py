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

Implementation note:
DWT is applied in a rolling window (causal) to avoid lookahead bias.
A global DWT would use future bars to compute past features.
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


def _compute_causal_dwt_signals(
    log_returns: np.ndarray,
    wavelet: str,
    levels: int,
    dwt_window: int,
) -> tuple:
    """Compute DWT signals causally via a rolling window.

    For each bar i, DWT is applied only to log_returns[max(0, i-dwt_window+1):i+1].
    The last element of each reconstructed level is used as the signal value at bar i.
    This guarantees that no future data influences past feature values.

    Args:
        log_returns: Log return series (length n).
        wavelet: Wavelet family name.
        levels: Number of decomposition levels requested.
        dwt_window: Maximum number of past bars used per DWT computation.

    Returns:
        Tuple (approx_signal, detail_signals) where approx_signal is an array of
        length n and detail_signals is a dict {level: array of length n}.
        Values are NaN until enough bars have accumulated (2**levels minimum).
    """
    n = len(log_returns)
    min_samples = 2 ** levels

    approx_signal = np.full(n, np.nan)
    detail_signals = {lvl: np.full(n, np.nan) for lvl in range(1, levels + 1)}

    for i in range(n):
        start = max(0, i - dwt_window + 1)
        segment = log_returns[start : i + 1]
        seg_len = len(segment)

        if seg_len < min_samples:
            continue

        max_lvl = pywt.dwt_max_level(seg_len, wavelet)
        actual_lvl = min(levels, max_lvl)

        coeffs = pywt.wavedec(segment, wavelet, level=actual_lvl)

        # Take the last reconstructed value = contribution at bar i
        rec = _reconstruct_level(coeffs, 0, wavelet)
        approx_signal[i] = rec[min(seg_len - 1, len(rec) - 1)]

        for lvl in range(1, levels + 1):
            if lvl > actual_lvl:
                continue
            coeffs_idx = actual_lvl + 1 - lvl
            rec_d = _reconstruct_level(coeffs, coeffs_idx, wavelet)
            detail_signals[lvl][i] = rec_d[min(seg_len - 1, len(rec_d) - 1)]

    return approx_signal, detail_signals


@register_indicator("wavelets")
class WaveletsIndicator(BaseIndicator):
    """Wavelet decomposition features for ML trading."""

    name = "wavelets"
    version = "2.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        wavelet: str = "db4",
        levels: int = 3,
        windows: List[int] = None,
        dwt_window: int = 256,
        **params,
    ) -> pd.DataFrame:
        """Compute wavelet decomposition features from close prices.

        Args:
            df: DataFrame with OHLC data (columns: O, H, L, C).
            wavelet: Wavelet family (default: 'db4').
            levels: Number of decomposition levels (default: 3).
            windows: Rolling windows for energy/mean stats (default: [10, 20, 50]).
            dwt_window: Number of past bars used per DWT computation (default: 256).
                        Larger values capture lower-frequency structure but are slower.

        Returns:
            DataFrame with original columns plus wavelet features.
        """
        if windows is None:
            windows = [10, 20, 50]

        close = df["C"].values
        log_returns = np.diff(np.log(close), prepend=np.log(close[0]))
        n = len(log_returns)

        approx_signal, detail_signals = _compute_causal_dwt_signals(
            log_returns, wavelet, levels, dwt_window
        )

        features = {}

        for w in windows:
            features[f"wt_approx_energy_{w}"] = _rolling_energy(approx_signal, w)

            for lvl in range(1, levels + 1):
                sig = detail_signals[lvl]
                features[f"wt_detail_{lvl}_energy_{w}"] = _rolling_energy(sig, w)
                features[f"wt_detail_{lvl}_mean_{w}"] = _rolling_mean(sig, w)

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

        for w in windows:
            hf_energy = _rolling_energy(detail_signals[1], w)
            lf_energy = _rolling_energy(detail_signals[levels], w)
            features[f"wt_high_freq_ratio_{w}"] = safe_divide(hf_energy, lf_energy)

        self._feature_columns = sorted(features.keys())

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        if self._feature_columns:
            return self._feature_columns
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
            "dwt_window": 256,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "wavelet": {
                "type": "choice",
                "default": "db4",
                "description": "Wavelet family for the Discrete Wavelet Transform. Daubechies wavelets (db1-db20) are most common for financial data. db4 provides a good balance between time and frequency localization. Haar (db1) is simplest, higher-order wavelets are smoother.",
                "choices": [
                    "db1", "db2", "db3", "db4", "db5", "db6", "db8", "db10",
                    "sym2", "sym3", "sym4", "sym5",
                    "coif1", "coif2", "coif3",
                ],
            },
            "levels": {
                "type": "int",
                "default": 3,
                "description": "Number of DWT decomposition levels. Each level halves the frequency band: level 1 captures highest frequencies (noise/microstructure), level N captures lowest detail frequencies. The approximation captures the remaining trend component. More levels separate more frequency bands but require longer input series.",
                "min": 1,
                "max": 12,
                "step": 1,
            },
            "windows": {
                "type": "list[int]",
                "default": [10, 20, 50],
                "description": "Rolling window sizes for computing energy (mean squared amplitude) and mean statistics of each wavelet decomposition level. Shorter windows capture recent energy shifts, longer windows provide more stable regime characterization.",
                "min": 2,
                "max": 5000,
            },
            "dwt_window": {
                "type": "int",
                "default": 256,
                "description": "Number of past bars used per causal DWT computation. Larger values capture lower-frequency structure with better resolution but increase computation time linearly. Must be >= 2**levels.",
                "min": 8,
                "max": 2048,
                "step": 64,
            },
        }


__all__ = ["WaveletsIndicator"]
