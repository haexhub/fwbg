"""
Topological Data Analysis (TDA) features via Persistent Homology.

Uses Takens time-delay embedding to convert 1D price returns into a point
cloud in higher-dimensional space, then applies persistent homology (ripser)
to extract topological features that capture the shape of market dynamics.

Features encode:
- H0 (connected components): fragmentation of price structure
- H1 (loops/cycles): cyclical patterns and mean-reversion signals
- Persistence statistics: strength and distribution of topological features
"""
from typing import List

import numpy as np
import pandas as pd
import ripser

from fwbg_sdk import BaseIndicator, shift_features, safe_divide, register_indicator


def _takens_embedding(
    series: np.ndarray, embedding_dim: int, time_delay: int
) -> np.ndarray | None:
    """Convert a 1D series into a point cloud via Takens time-delay embedding.

    Args:
        series: 1-D array of values (e.g. log returns).
        embedding_dim: Dimension of the embedding space.
        time_delay: Time delay between coordinates.

    Returns:
        2-D array of shape (n_points, embedding_dim), or None if too short.
    """
    n = len(series)
    n_points = n - (embedding_dim - 1) * time_delay
    if n_points <= 0:
        return None
    embedded = np.zeros((n_points, embedding_dim))
    for i in range(embedding_dim):
        embedded[:, i] = series[i * time_delay : i * time_delay + n_points]
    return embedded


def _finite_persistence(diagram: np.ndarray) -> np.ndarray:
    """Extract finite persistence values (death - birth) from a diagram."""
    if len(diagram) == 0:
        return np.array([])
    finite_mask = np.isfinite(diagram[:, 1])
    if not finite_mask.any():
        return np.array([])
    return diagram[finite_mask, 1] - diagram[finite_mask, 0]


def _persistence_entropy(persistence: np.ndarray) -> float:
    """Compute persistence entropy: -sum(p_i * log(p_i))."""
    total = persistence.sum()
    if total <= 0:
        return 0.0
    p = persistence / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def _extract_features(diagrams: list) -> dict:
    """Extract topological features from persistence diagrams.

    Args:
        diagrams: List of persistence diagrams from ripser (H0, H1, ...).

    Returns:
        Dict of feature name -> value for a single window position.
    """
    feats = {}
    for dim, label in [(0, "h0"), (1, "h1")]:
        if dim >= len(diagrams):
            feats[f"{label}_count"] = 0.0
            feats[f"{label}_max_pers"] = 0.0
            feats[f"{label}_mean_pers"] = 0.0
            continue
        pers = _finite_persistence(diagrams[dim])
        feats[f"{label}_count"] = float(len(pers))
        feats[f"{label}_max_pers"] = float(pers.max()) if len(pers) > 0 else 0.0
        feats[f"{label}_mean_pers"] = float(pers.mean()) if len(pers) > 0 else 0.0

    # Combine all finite persistence values for entropy and amplitude
    all_pers = np.concatenate(
        [_finite_persistence(diagrams[d]) for d in range(min(2, len(diagrams)))]
    )
    feats["persistence_entropy"] = (
        _persistence_entropy(all_pers) if len(all_pers) > 0 else 0.0
    )
    feats["wasserstein_amp"] = (
        float(np.sqrt(np.sum(all_pers**2))) if len(all_pers) > 0 else 0.0
    )
    feats["h1_ratio"] = None  # computed via safe_divide later
    feats["max_loop_persistence"] = None  # computed via safe_divide later
    return feats


def _rolling_tda(
    close: np.ndarray,
    window: int,
    embedding_dim: int,
    time_delay: int,
    maxdim: int,
) -> dict:
    """Compute TDA features over a rolling window.

    Returns:
        Dict of feature_name -> np.ndarray of length len(close).
    """
    n = len(close)
    log_returns = np.diff(np.log(close), prepend=np.log(close[0]))

    feature_names = [
        "h0_count", "h1_count", "h0_max_pers", "h1_max_pers",
        "h0_mean_pers", "h1_mean_pers", "persistence_entropy",
        "wasserstein_amp", "h1_ratio", "max_loop_persistence",
    ]
    arrays = {name: np.full(n, np.nan) for name in feature_names}

    for i in range(window - 1, n):
        segment = log_returns[i - window + 1 : i + 1]
        cloud = _takens_embedding(segment, embedding_dim, time_delay)
        if cloud is None or len(cloud) < 3:
            continue
        try:
            result = ripser.ripser(cloud, maxdim=maxdim)
        except Exception:
            continue
        feats = _extract_features(result["dgms"])
        for name in feature_names:
            if name in ("h1_ratio", "max_loop_persistence"):
                continue
            arrays[name][i] = feats[name]

    # Compute ratio features via safe_divide
    arrays["h1_ratio"] = safe_divide(arrays["h1_count"], arrays["h0_count"])
    arrays["max_loop_persistence"] = safe_divide(
        arrays["h1_max_pers"], arrays["h0_max_pers"]
    )
    return arrays


@register_indicator("topological_features")
class TopologicalFeaturesIndicator(BaseIndicator):
    """Topological Data Analysis features for ML trading."""

    name = "topological_features"
    version = "1.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        windows: List[int] | None = None,
        embedding_dim: int = 3,
        time_delay: int = 1,
        maxdim: int = 1,
        **params,
    ) -> pd.DataFrame:
        defaults = self.get_default_params()
        if windows is None:
            windows = defaults["windows"]

        close = df["C"].values
        features: dict = {}
        feature_names = [
            "h0_count", "h1_count", "h0_max_pers", "h1_max_pers",
            "h0_mean_pers", "h1_mean_pers", "persistence_entropy",
            "wasserstein_amp", "h1_ratio", "max_loop_persistence",
        ]

        for w in windows:
            arrays = _rolling_tda(close, w, embedding_dim, time_delay, maxdim)
            for name in feature_names:
                features[f"tda_{name}_{w}"] = arrays[name]

        self._feature_columns = sorted(features.keys())
        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        if self._feature_columns:
            return self._feature_columns
        params = self.get_default_params()
        cols = []
        names = [
            "h0_count", "h1_count", "h0_max_pers", "h1_max_pers",
            "h0_mean_pers", "h1_mean_pers", "persistence_entropy",
            "wasserstein_amp", "h1_ratio", "max_loop_persistence",
        ]
        for w in params["windows"]:
            for name in names:
                cols.append(f"tda_{name}_{w}")
        return sorted(cols)

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "windows": [50, 100],
            "embedding_dim": 3,
            "time_delay": 1,
            "maxdim": 1,
        }


__all__ = ["TopologicalFeaturesIndicator"]
