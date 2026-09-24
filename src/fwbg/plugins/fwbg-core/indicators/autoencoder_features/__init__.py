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


def _prepare_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    medians: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a finite feature matrix and return it with its imputation medians."""
    X = df.reindex(columns=feature_cols).to_numpy(dtype=np.float64, copy=True)
    X[~np.isfinite(X)] = np.nan

    if medians is None:
        medians = np.nanmedian(X, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)

    for j in range(X.shape[1]):
        X[np.isnan(X[:, j]), j] = medians[j]
    return X, medians


@register_indicator("autoencoder_features")
class AutoencoderFeaturesIndicator(BaseIndicator):
    """PCA-based latent feature extraction for ML trading."""

    name = "autoencoder_features"
    version = "1.0.0"
    stateful = True
    cacheable = False

    def __init__(self) -> None:
        super().__init__()
        self._feature_cols: List[str] = []
        self._medians: np.ndarray | None = None
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._effective_components = 0
        self._fit_params: dict = {}
        self._lazy_fit = False

    def fit(self, ctx, **params) -> None:
        """Fit imputation, scaling, and PCA state on training data only."""
        self.reset()
        exclude_prefixes = params.get("exclude_prefixes")
        if exclude_prefixes is None:
            exclude_prefixes = ["ae_"]
        n_components = params.get("n_components", 8)

        self._feature_cols = _select_feature_columns(ctx.df, exclude_prefixes)
        if not self._feature_cols:
            self._fitted = True
            self._fit_params = {
                "n_components": n_components,
                "exclude_prefixes": list(exclude_prefixes),
            }
            return

        X, self._medians = _prepare_matrix(ctx.df, self._feature_cols)
        self._effective_components = min(
            n_components, X.shape[1] - 1, X.shape[0] - 1
        )
        if self._effective_components < 1:
            self._fitted = True
            self._fit_params = {
                "n_components": n_components,
                "exclude_prefixes": list(exclude_prefixes),
            }
            return

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._pca = PCA(n_components=self._effective_components)
        self._pca.fit(X_scaled)
        self._fit_params = {
            "n_components": n_components,
            "exclude_prefixes": list(exclude_prefixes),
        }
        self._lazy_fit = False
        self._fitted = True

    def reset(self) -> None:
        super().reset()
        self._feature_cols = []
        self._medians = None
        self._scaler = None
        self._pca = None
        self._effective_components = 0
        self._fit_params = {}
        self._lazy_fit = False

    def compute(
        self,
        df: pd.DataFrame,
        n_components: int = 8,
        exclude_prefixes: List[str] | None = None,
        **params,
    ) -> pd.DataFrame:
        # Direct callers historically used compute() without an explicit fit.
        # Preserve that API by fitting on the supplied frame; the pipeline
        # runner always calls fit() with the fold's training frame first.
        requested_exclude_prefixes = (
            ["ae_"] if exclude_prefixes is None else list(exclude_prefixes)
        )
        requested_params = {
            "n_components": n_components,
            "exclude_prefixes": requested_exclude_prefixes,
        }
        if not self._fitted or (
            self._lazy_fit and requested_params != self._fit_params
        ):
            self.fit(
                type("FitContext", (), {"df": df})(),
                n_components=n_components,
                exclude_prefixes=requested_exclude_prefixes,
                **params,
            )
            self._lazy_fit = True

        if self._pca is None or self._scaler is None or self._medians is None:
            return df

        X, _ = _prepare_matrix(df, self._feature_cols, self._medians)
        X_scaled = self._scaler.transform(X)
        latent = self._pca.transform(X_scaled)

        # Reconstruction error: ||x - x_reconstructed||^2
        X_reconstructed = self._pca.inverse_transform(latent)
        recon_error = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

        # Cumulative explained variance
        cumulative_var = np.sum(self._pca.explained_variance_ratio_)

        # Build feature dict
        features = {}
        for i in range(self._effective_components):
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
