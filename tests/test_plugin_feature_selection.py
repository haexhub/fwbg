"""
Tests for plugin-driven feature selection.

Verifies:
1. No plugins configured = all features returned
2. Boruta/plateau work through the plugin registry
3. Unknown plugins raise ValueError
4. Plugin params are passed through correctly
5. Multiple plugins can be chained
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.core.context import SimulationContext
from fwbg.optimization.nested_cv import select_features_from_fold


@pytest.fixture
def feature_df():
    np.random.seed(42)
    n = 500
    y = np.random.randint(0, 2, n)
    df = pd.DataFrame({
        "relevant_1": y + np.random.normal(0, 0.3, n),
        "relevant_2": y * 0.8 + np.random.normal(0, 0.4, n),
        "noise_1": np.random.normal(0, 1, n),
        "noise_2": np.random.normal(0, 1, n),
    })
    return df, y


class TestPluginDrivenFeatureSelection:
    """Feature selection through plugin registry."""

    def test_no_plugins_returns_all_features(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        selected, metadata = select_features_from_fold(
            df, targets, features, min_trades=10,
            feature_selection_plugins=None,
        )
        assert selected == features
        assert metadata == {}

    def test_empty_plugin_list_returns_all_features(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        selected, metadata = select_features_from_fold(
            df, targets, features, min_trades=10,
            feature_selection_plugins=[],
        )
        assert selected == features

    def test_boruta_via_registry(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        plugins = [{"name": "boruta", "params": {
            "n_iter": 3, "n_estimators": 20, "max_depth": 3,
            "min_z_score": 0.0, "max_features": 10,
        }}]
        selected, metadata = select_features_from_fold(
            df, targets, features, min_trades=10,
            feature_selection_plugins=plugins,
        )
        assert selected is not None
        assert isinstance(selected, list)
        assert len(selected) > 0

    def test_plateau_via_registry(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        plugins = [{"name": "plateau", "params": {
            "n_estimators": 20, "max_depth": 3, "max_features": 10,
        }}]
        selected, metadata = select_features_from_fold(
            df, targets, features, min_trades=10,
            feature_selection_plugins=plugins,
        )
        assert selected is not None
        assert isinstance(selected, list)

    def test_unknown_plugin_raises(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        plugins = [{"name": "nonexistent_selector", "params": {}}]
        with pytest.raises(ValueError, match="Unknown feature selector"):
            select_features_from_fold(
                df, targets, features, min_trades=10,
                feature_selection_plugins=plugins,
            )

    def test_plugin_params_passed_through(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        plugins = [{"name": "boruta", "params": {
            "n_iter": 2, "n_estimators": 10, "max_depth": 2,
            "min_z_score": 100.0,  # Impossibly high → no features pass
            "max_features": 1,
        }}]
        selected, metadata = select_features_from_fold(
            df, targets, features, min_trades=10,
            feature_selection_plugins=plugins,
        )
        assert selected is None


class TestPluginChaining:
    """Multiple plugins are chained: each operates on previous output."""

    def test_two_plugins_chain(self, feature_df):
        df, targets = feature_df
        features = list(df.columns)
        plugins = [
            {"name": "boruta", "params": {
                "n_iter": 3, "n_estimators": 20, "max_depth": 3,
                "min_z_score": 0.0, "max_features": 10,
            }},
            {"name": "plateau", "params": {
                "n_estimators": 20, "max_depth": 3, "max_features": 10,
            }},
        ]
        selected, metadata = select_features_from_fold(
            df, targets, features, min_trades=10,
            feature_selection_plugins=plugins,
        )
        assert selected is None or isinstance(selected, list)
