"""
Latent feature extraction via PCA.

Compresses all numeric indicator features into low-dimensional latent
representations using Principal Component Analysis. PCA is equivalent to
a linear autoencoder's bottleneck layer but is deterministic and fast.

Features produced:
- ae_latent_{i}: PCA component i (captures main modes of variation)
- ae_reconstruction_error: Per-row reconstruction error (anomaly signal)
- ae_explained_variance: Cumulative explained variance ratio
"""
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from fwbg_sdk import BaseIndicator, shift_features, register_indicator

# Columns to always exclude from PCA input
_OHLCV = {"O", "H", "L", "C", "V"}


def _select_feature_columns(df: pd.DataFrame, exclude_prefixes: List[str]) -> List[str]:
    """Select numeric feature columns, excluding OHLCV and specified prefixes."""
    cols = []
    for col in df.columns:
        if col in _OHLCV:
            continue
        if any(col.startswith(p) for p in exclude_prefixes):
            continue
        if df[col].dtype in (np.float64, np.float32, np.int64, np.int32):
            cols.append(col)
    return cols


@register_indicator("autoencoder_features")
class AutoencoderFeaturesIndicator(BaseIndicator):
    """PCA-based latent feature extraction for ML trading."""

    name = "autoencoder_features"
    version = "1.0.0"

    def compute(
        self,
        df: pd.DataFrame,
        n_components: int = 8,
        exclude_prefixes: List[str] | None = None,
        **params,
    ) -> pd.DataFrame:
        if exclude_prefixes is None:
            exclude_prefixes = ["ae_"]

        # Select numeric feature columns
        feature_cols = _select_feature_columns(df, exclude_prefixes)

        if len(feature_cols) == 0:
            return df

        X = df[feature_cols].values.astype(np.float64)

        # Replace Inf/-Inf with NaN first, then impute all NaN
        X[~np.isfinite(X)] = np.nan

        # Fill NaN with column median (robust to outliers)
        medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = medians[j] if np.isfinite(medians[j]) else 0.0

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Adjust n_components if fewer features available
        effective_components = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
        if effective_components < 1:
            return df

        # Fit PCA
        pca = PCA(n_components=effective_components)
        latent = pca.fit_transform(X_scaled)

        # Reconstruction error: ||x - x_reconstructed||^2
        X_reconstructed = pca.inverse_transform(latent)
        recon_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

        # Cumulative explained variance
        cumulative_var = np.sum(pca.explained_variance_ratio_)

        # Build feature dict
        features = {}
        for i in range(effective_components):
            features[f"ae_latent_{i}"] = latent[:, i]
        features["ae_reconstruction_error"] = recon_error
        features["ae_explained_variance"] = np.full(len(df), cumulative_var)

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        # Default feature list for n_components=8
        cols = [f"ae_latent_{i}" for i in range(8)]
        cols.append("ae_reconstruction_error")
        cols.append("ae_explained_variance")
        return cols

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "n_components": 8,
            "exclude_prefixes": ["ae_"],
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "n_components": {
                "type": "int",
                "default": 8,
                "description": "Number of PCA components (latent dimensions) to extract from all numeric indicator features. Each component captures an orthogonal mode of variation. More components preserve more information but increase dimensionality. The reconstruction error feature acts as an anomaly detector regardless of this setting.",
                "min": 1,
                "max": 500,
                "step": 1,
            },
            "exclude_prefixes": {
                "type": "list[string]",
                "default": ["ae_"],
                "description": "Column name prefixes to exclude from PCA input. By default excludes the autoencoder's own output columns (ae_*) to prevent circular dependencies. Add other prefixes to exclude specific indicator groups from the latent representation.",
            },
        }


__all__ = ["AutoencoderFeaturesIndicator"]
