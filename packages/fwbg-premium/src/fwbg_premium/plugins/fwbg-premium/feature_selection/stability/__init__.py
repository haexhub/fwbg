"""
Stability Selection Feature Selection Plugin.

Runs an inner feature selector (e.g., Boruta) multiple times with
bootstrap resampling. Only features selected in >= threshold% of
runs are kept.

Reference: Meinshausen & Bühlmann (2010) "Stability Selection"
"""
from typing import List, Tuple
import numpy as np
import pandas as pd

from fwbg_sdk import BaseFeatureSelector, register_feature_selector
from fwbg.core import get_feature_selector


@register_feature_selector("stability")
class StabilitySelector(BaseFeatureSelector):
    """Stability Selection via bootstrap resampling of an inner selector."""

    name = "stability"

    def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        max_features: int = None,
        inner_selector: str = "boruta",
        inner_params: dict = None,
        n_bootstrap: int = 10,
        threshold: float = 0.6,
        bootstrap_ratio: float = 0.8,
        **params,
    ) -> Tuple[List[str], dict]:
        """
        Run inner selector n_bootstrap times with bootstrap samples.
        Keep features selected in >= threshold fraction of runs.
        """
        rng = np.random.default_rng()
        n_samples = len(X)
        n_per_sample = int(n_samples * bootstrap_ratio)
        feature_votes = {}

        selector_cls = get_feature_selector(inner_selector)

        for _ in range(n_bootstrap):
            indices = rng.choice(n_samples, n_per_sample, replace=True)
            X_boot = X.iloc[indices].reset_index(drop=True)
            y_boot = y[indices]

            selector = selector_cls()
            selected, _ = selector.select_features(
                X_boot, y_boot, **(inner_params or {})
            )

            for feat in (selected or []):
                feature_votes[feat] = feature_votes.get(feat, 0) + 1

        # Filter by threshold
        threshold_count = threshold * n_bootstrap
        stable = [
            f for f, count in feature_votes.items()
            if count >= threshold_count
        ]

        # Sort by vote count (most stable first)
        stable.sort(key=lambda f: feature_votes.get(f, 0), reverse=True)

        if max_features and len(stable) > max_features:
            stable = stable[:max_features]

        metadata = {
            "feature_votes": feature_votes,
            "n_bootstrap": n_bootstrap,
            "threshold": threshold,
            "n_selected": len(stable),
        }

        return stable, metadata

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "inner_selector": "boruta",
            "inner_params": {"n_iter": 5, "n_estimators": 30, "min_z_score": 0.5},
            "n_bootstrap": 10,
            "threshold": 0.6,
            "bootstrap_ratio": 0.8,
        }
