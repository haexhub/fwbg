"""
Tests for plugin-driven feature selection.

Verifies:
1. No plugins configured = all features returned
2. Selectors registered with the plugin registry are used
3. Unknown plugins raise ValueError
4. Plugin params are passed through correctly
5. Multiple plugins can be chained
"""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.registry import FEATURE_SELECTOR_REGISTRY
from fwbg.optimization.nested_cv import select_features_from_fold
from fwbg_sdk import BaseFeatureSelector, register_feature_selector


@pytest.fixture(autouse=True)
def corr_selectors():
    """Register two minimal selectors (no feature selector ships in the core package)."""

    @register_feature_selector("_test_corr_rank")
    class CorrRankSelector(BaseFeatureSelector):
        """Keep features whose |corr| with the target is at least ``min_corr``."""

        name = "_test_corr_rank"

        def select_features(self, X, y, max_features=None, min_corr=0.0, **params):
            corr = {c: abs(np.corrcoef(X[c].to_numpy(), y)[0, 1]) for c in X.columns}
            ranked = sorted(
                (c for c in X.columns if corr[c] >= min_corr), key=lambda c: -corr[c]
            )
            if max_features is not None:
                ranked = ranked[:max_features]
            return ranked, {"corr": {c: corr[c] for c in ranked}}

    @register_feature_selector("_test_drop_last")
    class DropLastSelector(BaseFeatureSelector):
        """Drop the weakest feature of its input."""

        name = "_test_drop_last"

        def select_features(self, X, y, max_features=None, **params):
            corr = {c: abs(np.corrcoef(X[c].to_numpy(), y)[0, 1]) for c in X.columns}
            kept = sorted(X.columns, key=lambda c: -corr[c])[:-1]
            return kept, {"dropped_last": True}

    yield
    FEATURE_SELECTOR_REGISTRY.pop("_test_corr_rank", None)
    FEATURE_SELECTOR_REGISTRY.pop("_test_drop_last", None)


@pytest.fixture
def feature_df():
    np.random.seed(42)
    n = 500
    y = np.random.randint(0, 2, n)
    df = pd.DataFrame(
        {
            "relevant_1": y + np.random.normal(0, 0.3, n),
            "relevant_2": y * 0.8 + np.random.normal(0, 0.4, n),
            "noise_1": np.random.normal(0, 1, n),
            "noise_2": np.random.normal(0, 1, n),
        }
    )
    return df, y


class TestPluginDrivenFeatureSelection:
    """Feature selection through plugin registry."""

    def test_no_plugins_returns_all_features(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        selected, metadata = select_features_from_fold(
            df,
            targets,
            features,
            min_trades=10,
            feature_selection_plugins=None,
        )
        assert selected == features
        assert metadata == {}

    def test_empty_plugin_list_returns_all_features(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        selected, metadata = select_features_from_fold(
            df,
            targets,
            features,
            min_trades=10,
            feature_selection_plugins=[],
        )
        assert selected == features

    def test_selector_via_registry(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        plugins = [
            {"name": "_test_corr_rank", "params": {"min_corr": 0.3, "max_features": 10}}
        ]
        selected, metadata = select_features_from_fold(
            df,
            targets,
            features,
            min_trades=10,
            feature_selection_plugins=plugins,
        )
        assert selected is not None
        assert set(selected) == {"relevant_1", "relevant_2"}
        assert selected[0] == "relevant_1"  # ranked by correlation
        assert set(metadata["corr"]) == {"relevant_1", "relevant_2"}

    def test_max_features_is_applied_by_the_caller(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        plugins = [
            {"name": "_test_corr_rank", "params": {"min_corr": 0.0, "max_features": 3}}
        ]
        selected, _ = select_features_from_fold(
            df,
            targets,
            features,
            min_trades=10,
            feature_selection_plugins=plugins,
        )
        assert selected is not None
        assert len(selected) == 3

    def test_unknown_plugin_raises(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        plugins = [{"name": "nonexistent_selector", "params": {}}]
        with pytest.raises(ValueError, match="Unknown feature selector"):
            select_features_from_fold(
                df,
                targets,
                features,
                min_trades=10,
                feature_selection_plugins=plugins,
            )

    def test_plugin_params_passed_through(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        plugins = [
            {
                "name": "_test_corr_rank",
                "params": {
                    "min_corr": 100.0,  # Impossibly high -> no features pass
                    "max_features": 1,
                },
            }
        ]
        selected, metadata = select_features_from_fold(
            df,
            targets,
            features,
            min_trades=10,
            feature_selection_plugins=plugins,
        )
        assert selected is None


class TestPluginChaining:
    """Multiple plugins are chained: each operates on previous output."""

    def test_two_plugins_chain(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        plugins = [
            {"name": "_test_corr_rank", "params": {"min_corr": 0.0, "max_features": 3}},
            {"name": "_test_drop_last", "params": {}},
        ]
        selected, metadata = select_features_from_fold(
            df,
            targets,
            features,
            min_trades=10,
            feature_selection_plugins=plugins,
        )
        # The second selector only sees the three features the first one kept.
        assert selected is not None
        assert len(selected) == 2
        assert metadata["dropped_last"] is True
        assert "corr" in metadata
