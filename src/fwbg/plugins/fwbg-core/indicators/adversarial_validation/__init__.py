"""
Adversarial Validation indicator plugin.

Detects distribution shift between recent and older market data by training
a classifier to distinguish "old" from "new" observations within a sliding
window.  High AUC signals a regime change; values near 0.5 indicate a
stable distribution.
"""
from typing import List

import numpy as np
import pandas as pd

from fwbg_sdk import BaseIndicator, shift_features, safe_divide, register_indicator


def _select_feature_columns(df: pd.DataFrame, exclude_prefixes: List[str]) -> List[str]:
    """Select numeric columns, excluding OHLCV and specified prefixes."""
    ohlcv = {"O", "H", "L", "C", "V"}
    cols = []
    for c in df.columns:
        if c in ohlcv:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _compute_adversarial_auc(old_data: np.ndarray, new_data: np.ndarray):
    """Fit LogisticRegression to distinguish old from new data, return AUC and max importance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([old_data, new_data])
    y = np.concatenate([np.zeros(len(old_data)), np.ones(len(new_data))])

    # Replace Inf/-Inf with NaN, then remove rows with NaN
    X[~np.isfinite(X)] = np.nan
    valid = ~np.isnan(X).any(axis=1)
    X, y = X[valid], y[valid]

    if len(X) < 10 or len(np.unique(y)) < 2:
        return 0.5, 0.0

    # Impute remaining NaN with median
    medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = medians[j] if np.isfinite(medians[j]) else 0.0

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    try:
        model = LogisticRegression(max_iter=100, solver="liblinear", C=1.0)
        model.fit(X, y)
        proba = model.decision_function(X)
        auc = roc_auc_score(y, proba)
        max_importance = float(np.max(np.abs(model.coef_[0])))
    except Exception:
        auc = 0.5
        max_importance = 0.0

    return auc, max_importance


def _compute_window_features(
    data: np.ndarray,
    window: int,
    step: int,
    max_features: int,
    n_rows: int,
) -> dict:
    """Compute adversarial validation features for a single window size."""
    rng = np.random.RandomState(42)
    n_cols = data.shape[1]

    auc_arr = np.full(n_rows, np.nan)
    importance_arr = np.full(n_rows, np.nan)

    for i in range(window, n_rows, step):
        half = window // 2
        old_slice = data[i - window: i - half]
        new_slice = data[i - half: i]

        # Feature subsampling for performance
        if n_cols > max_features:
            col_idx = rng.choice(n_cols, max_features, replace=False)
            old_slice = old_slice[:, col_idx]
            new_slice = new_slice[:, col_idx]

        auc, imp = _compute_adversarial_auc(old_slice, new_slice)
        auc_arr[i] = auc
        importance_arr[i] = imp

    # Forward-fill between steps (copy to make arrays writable)
    auc_s = pd.Series(auc_arr).ffill().to_numpy(copy=True)
    importance_s = pd.Series(importance_arr).ffill().to_numpy(copy=True)

    drift = np.clip(2.0 * (auc_s - 0.5), 0.0, 1.0)
    stability = 1.0 - drift
    acceleration = pd.Series(drift).diff().to_numpy(copy=True)

    # Preserve NaN for warmup (before first computed point)
    warmup = np.isnan(auc_s)
    for arr in [auc_s, drift, stability, importance_s, acceleration]:
        arr[warmup] = np.nan

    w = window
    return {
        f"adv_auc_{w}": auc_s,
        f"adv_drift_score_{w}": drift,
        f"adv_stability_{w}": stability,
        f"adv_max_feature_importance_{w}": importance_s,
        f"adv_drift_acceleration_{w}": acceleration,
    }


@register_indicator("adversarial_validation")
class AdversarialValidationIndicator(BaseIndicator):
    """Adversarial Validation features for ML trading."""

    name = "adversarial_validation"
    version = "1.0.0"

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        defaults = self.get_default_params()
        windows = params.get("windows", defaults["windows"])
        step = params.get("step", defaults["step"])
        max_features = params.get("max_features", defaults["max_features"])
        exclude_prefixes = params.get("exclude_prefixes", defaults["exclude_prefixes"])

        feat_cols = _select_feature_columns(df, exclude_prefixes)
        if len(feat_cols) < 3:
            return df

        data = df[feat_cols].values
        features: dict = {}

        for w in windows:
            feats = _compute_window_features(data, w, step, max_features, len(df))
            features.update(feats)

        self._feature_columns = list(features.keys())
        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        if self._feature_columns:
            return self._feature_columns
        params = self.get_default_params()
        cols = []
        for w in params["windows"]:
            cols.extend([
                f"adv_auc_{w}",
                f"adv_drift_score_{w}",
                f"adv_stability_{w}",
                f"adv_max_feature_importance_{w}",
                f"adv_drift_acceleration_{w}",
            ])
        return cols

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "windows": [100, 200],
            "step": 10,
            "max_features": 30,
            "exclude_prefixes": ["adv_"],
        }


__all__ = ["AdversarialValidationIndicator"]
