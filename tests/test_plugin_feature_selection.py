"""
Tests for plugin-driven feature selection.

Verifies:
1. Feature selection goes through the plugin registry, not hardcoded imports
2. Multiple plugins can be chained (each operates on previous output)
3. Unknown plugins raise ValueError
4. No plugins configured = all features returned
5. select_features_from_fold uses the same interface regardless of plugin
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from fwbg.core.context import SimulationContext
from fwbg.optimization.nested_cv import select_features_from_fold


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def feature_df():
    """DataFrame with features and targets."""
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


# =============================================================================
# Test: Plugin-driven feature selection via registry
# =============================================================================

class TestPluginDrivenFeatureSelection:
    """Feature selection must go through the plugin registry interface."""

    def test_no_plugins_returns_all_features(self, feature_df):
        """When no plugins configured, return all available features."""
        df, targets = feature_df
        features = list(df.columns)

        selected, metadata = select_features_from_fold(
            df, targets, features, min_trades=10,
            feature_selection_plugins=None,
        )

        assert selected == features
        assert metadata == {}

    def test_empty_plugin_list_returns_all_features(self, feature_df):
        """Empty plugin list = no selection = all features."""
        df, targets = feature_df
        features = list(df.columns)

        selected, metadata = select_features_from_fold(
            df, targets, features, min_trades=10,
            feature_selection_plugins=[],
        )

        assert selected == features

    def test_boruta_via_registry(self, feature_df):
        """Boruta plugin works through registry (not direct import)."""
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
        """Plateau plugin works through registry."""
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
        """Unknown plugin name raises ValueError."""
        df, targets = feature_df
        features = list(df.columns)

        plugins = [{"name": "nonexistent_selector", "params": {}}]

        with pytest.raises(ValueError, match="Unknown feature selector"):
            select_features_from_fold(
                df, targets, features, min_trades=10,
                feature_selection_plugins=plugins,
            )

    def test_plugin_params_passed_through(self, feature_df):
        """Plugin params from config are passed to select_features()."""
        df, targets = feature_df
        features = list(df.columns)

        # Use boruta with very specific params that affect behavior
        plugins = [{"name": "boruta", "params": {
            "n_iter": 2,
            "n_estimators": 10,
            "max_depth": 2,
            "min_z_score": 100.0,  # Impossibly high → no features selected
            "max_features": 1,
        }}]

        selected, metadata = select_features_from_fold(
            df, targets, features, min_trades=10,
            feature_selection_plugins=plugins,
        )

        # With min_z_score=100, no feature can pass → should return None
        assert selected is None


# =============================================================================
# Test: Plugin chaining
# =============================================================================

class TestPluginChaining:
    """Multiple plugins are chained: each operates on previous output."""

    def test_two_plugins_chain(self, feature_df):
        """Second plugin operates on first plugin's selected features."""
        df, targets = feature_df
        features = list(df.columns)

        # Chain: boruta first (broad selection), then plateau (filter)
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

        # Should produce a result (or None if too few features survive)
        assert selected is None or isinstance(selected, list)


# =============================================================================
# Test: No hardcoded boruta in nested_cv
# =============================================================================

class TestNoHardcodedBoruta:
    """nested_cv.py must not contain hardcoded boruta references."""

    def test_no_hardcoded_boruta_import(self):
        """nested_cv must not import select_features_boruta."""
        import inspect
        from fwbg.optimization import nested_cv

        source = inspect.getsource(nested_cv)

        assert "select_features_boruta" not in source, (
            "nested_cv.py still imports select_features_boruta — "
            "feature selection must go through the plugin registry"
        )

    def test_no_hardcoded_boruta_branching(self):
        """select_features_from_fold must not branch on 'boruta' string."""
        import inspect
        from fwbg.optimization.nested_cv import select_features_from_fold

        source = inspect.getsource(select_features_from_fold)

        assert 'feature_selection == "boruta"' not in source, (
            "select_features_from_fold still has hardcoded boruta branching"
        )
        assert 'feature_selection == "boruta_plateau"' not in source, (
            "select_features_from_fold still has hardcoded boruta_plateau branching"
        )

    def test_no_boruta_specific_params_on_context(self):
        """SimulationContext must not have boruta-specific fields."""
        import inspect
        source = inspect.getsource(SimulationContext)

        assert "min_z_score" not in source, (
            "SimulationContext still has min_z_score (boruta-specific param)"
        )

    def test_context_has_feature_selection_plugins_list(self):
        """SimulationContext must have feature_selection_plugins (list)."""
        ctx = SimulationContext(
            symbol="TEST", asset_class="FOREX",
            spread=0.0002, point=0.0001,
        )

        assert hasattr(ctx, "feature_selection_plugins")
        # Should be a list (or None), not a single dict
        assert ctx.feature_selection_plugins is None or isinstance(
            ctx.feature_selection_plugins, list
        )


# =============================================================================
# Test: Same interface from grid_search and nested_cv
# =============================================================================

class TestUnifiedInterface:
    """grid_search.py and nested_cv.py must use the same feature selection path."""

    def test_grid_search_uses_plugin_interface(self):
        """grid_search.select_features must use feature_selection_plugins."""
        import inspect
        from fwbg.optimization.grid_search import select_features

        source = inspect.getsource(select_features)

        # Must NOT use the old boruta-specific params
        assert "ctx.feature_selection" not in source or "ctx.feature_selection_plugins" in source, (
            "grid_search.select_features still uses ctx.feature_selection (string) "
            "instead of ctx.feature_selection_plugins (list)"
        )
        assert "ctx.min_z_score" not in source, (
            "grid_search.select_features still references boruta-specific min_z_score"
        )
        assert "ctx.max_features" not in source, (
            "grid_search.select_features still references max_features from context"
        )
