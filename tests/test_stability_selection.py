"""
Tests für Stability Selection Feature Selection Plugin.

Stability Selection runs an inner feature selector (e.g., Boruta)
multiple times with bootstrap resampling. Only features selected
in >= threshold% of runs are kept. This makes selection robust
against sampling variation.

Reference: Meinshausen & Bühlmann (2010) "Stability Selection"
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module


@pytest.fixture
def predictive_data():
    """Create data with known predictive vs noise features."""
    rng = np.random.default_rng(42)
    n = 500
    y = rng.integers(0, 2, n).astype(float)

    df = pd.DataFrame({
        # Strongly predictive (should survive stability selection)
        "strong_1": y + rng.normal(0, 0.2, n),
        "strong_2": y * 0.9 + rng.normal(0, 0.3, n),
        # Weakly predictive (might not survive)
        "weak_1": y * 0.3 + rng.normal(0, 0.8, n),
        "weak_2": y * 0.2 + rng.normal(0, 0.9, n),
        # Pure noise (should be filtered)
        "noise_1": rng.standard_normal(n),
        "noise_2": rng.standard_normal(n),
        "noise_3": rng.standard_normal(n),
        "noise_4": rng.standard_normal(n),
    })

    return df, y


class TestStabilitySelector:
    """Tests for the stability feature selection plugin."""

    def test_plugin_registered(self):
        """Stability selector should be discoverable via registry."""
        from fwbg.core import get_feature_selector

        cls = get_feature_selector("stability")
        assert cls is not None
        assert cls.name == "stability"

    def test_selects_stable_features(self, predictive_data):
        """Strongly predictive features should survive stability selection."""
        from fwbg.core import get_feature_selector

        selector = get_feature_selector("stability")()
        df, y = predictive_data

        selected, metadata = selector.select_features(
            df, y,
            inner_selector="boruta",
            inner_params={"n_iter": 3, "n_estimators": 20, "max_depth": 3, "min_z_score": 0.0},
            n_bootstrap=5,
            threshold=0.6,
            bootstrap_ratio=0.8,
        )

        # Strong features should be selected
        assert "strong_1" in selected
        assert "strong_2" in selected
        # Should return at least the strong features
        assert len(selected) >= 2

    def test_filters_noise_features(self, predictive_data):
        """Pure noise features should be filtered out by stability selection."""
        from fwbg.core import get_feature_selector

        selector = get_feature_selector("stability")()
        df, y = predictive_data

        selected, metadata = selector.select_features(
            df, y,
            inner_selector="boruta",
            inner_params={"n_iter": 3, "n_estimators": 20, "max_depth": 3, "min_z_score": 0.3},
            n_bootstrap=7,
            threshold=0.6,
            bootstrap_ratio=0.8,
        )

        # Noise features should NOT survive stability selection
        noise_in_selected = [f for f in selected if f.startswith("noise_")]
        assert len(noise_in_selected) == 0, (
            f"Noise features should be filtered: {noise_in_selected}"
        )

    def test_metadata_contains_vote_counts(self, predictive_data):
        """Metadata should contain per-feature vote counts and config."""
        from fwbg.core import get_feature_selector

        selector = get_feature_selector("stability")()
        df, y = predictive_data

        n_bootstrap = 5
        selected, metadata = selector.select_features(
            df, y,
            inner_selector="boruta",
            inner_params={"n_iter": 3, "n_estimators": 20, "max_depth": 3, "min_z_score": 0.0},
            n_bootstrap=n_bootstrap,
            threshold=0.6,
            bootstrap_ratio=0.8,
        )

        assert "feature_votes" in metadata
        assert "n_bootstrap" in metadata
        assert "threshold" in metadata
        assert metadata["n_bootstrap"] == n_bootstrap

        # Vote counts should be between 0 and n_bootstrap
        for feat, count in metadata["feature_votes"].items():
            assert 0 <= count <= n_bootstrap

    def test_respects_max_features(self, predictive_data):
        """max_features should cap the output list."""
        from fwbg.core import get_feature_selector

        selector = get_feature_selector("stability")()
        df, y = predictive_data

        selected, _ = selector.select_features(
            df, y,
            max_features=2,
            inner_selector="boruta",
            inner_params={"n_iter": 3, "n_estimators": 20, "max_depth": 3, "min_z_score": 0.0},
            n_bootstrap=5,
            threshold=0.4,  # Low threshold to get more candidates
            bootstrap_ratio=0.8,
        )

        assert len(selected) <= 2

    def test_high_threshold_fewer_features(self, predictive_data):
        """Higher threshold should produce fewer (but more stable) features."""
        from fwbg.core import get_feature_selector

        selector = get_feature_selector("stability")()
        df, y = predictive_data

        params = dict(
            inner_selector="boruta",
            inner_params={"n_iter": 3, "n_estimators": 20, "max_depth": 3, "min_z_score": 0.0},
            n_bootstrap=7,
            bootstrap_ratio=0.8,
        )

        selected_low, _ = selector.select_features(df, y, threshold=0.3, **params)
        selected_high, _ = selector.select_features(df, y, threshold=0.8, **params)

        # Higher threshold should select fewer or equal features
        assert len(selected_high) <= len(selected_low)

    def test_integration_with_select_features_from_fold(self, predictive_data):
        """Stability selector should work when chained via select_features_from_fold."""
        from fwbg.optimization.nested_cv import select_features_from_fold

        df, y = predictive_data

        plugins = [{
            "name": "stability",
            "params": {
                "inner_selector": "boruta",
                "inner_params": {"n_iter": 3, "n_estimators": 20, "max_depth": 3, "min_z_score": 0.0},
                "n_bootstrap": 5,
                "threshold": 0.6,
                "bootstrap_ratio": 0.8,
                "max_features": 10,
            },
        }]

        selected, metadata = select_features_from_fold(
            df, y, list(df.columns), min_trades=10,
            feature_selection_plugins=plugins,
        )

        assert selected is not None
        assert len(selected) >= 2
        # Strong features should be in the selection
        assert "strong_1" in selected or "strong_2" in selected
