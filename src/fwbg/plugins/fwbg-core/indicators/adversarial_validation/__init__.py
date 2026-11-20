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

from fwbg_sdk import BaseIndicator, shift_features, register_indicator


# Columns that are never covariates: chart overlays (``_`` prefix, absolute price
# levels that trivially separate old from new bars) and target / label columns.
_ALWAYS_EXCLUDED_PREFIXES = ("_", "target_", "label_", "y_")
_LABEL_COLUMNS = {"target", "label", "y", "y_true"}


def _select_feature_columns(df: pd.DataFrame, exclude_prefixes: List[str]) -> List[str]:
    """Select numeric covariates for the adversarial classifier.

    Excludes OHLCV, ``_``-prefixed overlay columns, target / label columns and
    anything starting with one of ``exclude_prefixes``.
    """
    ohlcv = {"O", "H", "L", "C", "V"}
    cols = []
    for c in df.columns:
        if c in ohlcv:
            continue
        if c.startswith(_ALWAYS_EXCLUDED_PREFIXES) or c.lower() in _LABEL_COLUMNS:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _compute_adversarial_auc(
    old_data: np.ndarray, new_data: np.ndarray, n_folds: int = 4
):
    """Out-of-sample AUC of a classifier separating old from new observations.

    The AUC is cross-validated with *contiguous* blocks: each fold holds out one
    consecutive block of the old rows and one of the new rows (so autocorrelated
    neighbours do not leak across the split), the scaler and the classifier are
    fit on the remaining rows only, and the fold AUCs are averaged. Scoring on
    the training data would reach AUC ~ 1 for pure noise whenever the number of
    columns approaches the number of rows.

    Returns:
        ``(auc, max_importance)``: mean out-of-fold AUC and the mean over folds
        of the largest absolute standardised coefficient. ``(0.5, 0.0)`` when
        there are too few usable rows.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([old_data, new_data])
    y = np.concatenate([np.zeros(len(old_data)), np.ones(len(new_data))])

    # Replace Inf/-Inf with NaN, then remove rows with NaN
    X[~np.isfinite(X)] = np.nan
    valid = ~np.isnan(X).any(axis=1)
    X, y = X[valid], y[valid]

    old_idx = np.flatnonzero(y == 0)
    new_idx = np.flatnonzero(y == 1)
    if len(X) < 10 or min(len(old_idx), len(new_idx)) < 2 * n_folds:
        return 0.5, 0.0

    aucs = []
    importances = []
    try:
        for old_test, new_test in zip(
            np.array_split(old_idx, n_folds), np.array_split(new_idx, n_folds)
        ):
            test = np.concatenate([old_test, new_test])
            train = np.ones(len(X), dtype=bool)
            train[test] = False

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train])
            X_test = scaler.transform(X[test])

            model = LogisticRegression(max_iter=100, solver="liblinear", C=1.0)
            model.fit(X_train, y[train])
            aucs.append(roc_auc_score(y[test], model.decision_function(X_test)))
            importances.append(float(np.max(np.abs(model.coef_[0]))))
    except ValueError:
        return 0.5, 0.0

    return float(np.mean(aucs)), float(np.mean(importances))


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
        old_slice = data[i - window : i - half]
        new_slice = data[i - half : i]

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
    version = "1.1.0"

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
            cols.extend(
                [
                    f"adv_auc_{w}",
                    f"adv_drift_score_{w}",
                    f"adv_stability_{w}",
                    f"adv_max_feature_importance_{w}",
                    f"adv_drift_acceleration_{w}",
                ]
            )
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
