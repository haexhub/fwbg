"""
Correlation Filter Feature Selection Plugin.

Removes redundant features by filtering out highly correlated pairs.
Designed to run AFTER an importance-based selector (e.g., Stability Boruta)
so that input features are already sorted by importance.

Algorithm (greedy, O(n²)):
1. Compute absolute pairwise correlation matrix
2. Iterate features in input order (most important first)
3. Keep feature only if |corr| < threshold with all already-kept features
4. Apply optional max_features limit
"""
from typing import List, Tuple
import numpy as np
import pandas as pd

from fwbg_sdk import BaseFeatureSelector, register_feature_selector


@register_feature_selector("correlation_filter")
class CorrelationFilter(BaseFeatureSelector):
    """Greedy correlation-based redundancy filter."""

    def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        max_features: int = None,
        max_correlation: float = 0.7,
        **params,
    ) -> Tuple[List[str], dict]:
        """
        Filter highly correlated features, keeping the most important.

        Features arrive pre-sorted by importance (from upstream selector).
        For each feature, check if it's too correlated with any already-selected
        feature. If so, drop it.

        Args:
            X: Feature DataFrame (columns = pre-selected features)
            y: Target array (unused, required by interface)
            max_features: Optional hard cap on output features
            max_correlation: Maximum absolute correlation allowed (default 0.7)

        Returns:
            (selected_features, metadata)
        """
        if len(X.columns) <= 1:
            return list(X.columns), {"n_dropped": 0, "dropped": []}

        # Compute correlation matrix once
        corr_matrix = X.corr().abs()

        selected = []
        dropped = []
        drop_reasons = {}

        for feat in X.columns:
            if max_features and len(selected) >= max_features:
                break

            # Check correlation with all already-selected features
            too_correlated = False
            for kept in selected:
                corr_val = corr_matrix.loc[feat, kept]
                if corr_val >= max_correlation:
                    too_correlated = True
                    drop_reasons[feat] = f"{kept} (r={corr_val:.2f})"
                    break

            if too_correlated:
                dropped.append(feat)
            else:
                selected.append(feat)

        metadata = {
            "n_input": len(X.columns),
            "n_selected": len(selected),
            "n_dropped": len(dropped),
            "dropped": dropped,
            "drop_reasons": drop_reasons,
        }

        return selected, metadata

    @classmethod
    def get_default_params(cls) -> dict:
        return {
            "max_correlation": 0.7,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "max_correlation": {
                "type": "float",
                "default": 0.7,
                "description": "Maximum absolute pairwise correlation allowed between kept features; higher-correlated features are dropped",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
            },
        }
