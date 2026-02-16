"""
Tests for the regime_cluster indicator plugin.

Tests:
- Score computation from core inputs
- Cluster labels 0/1/2 via quantiles
- Missing optional input handled gracefully
- All outputs shifted by 1 bar (lookahead prevention)
- regime_cluster_n_inputs diagnostic correct
- Plugin registration and discovery
- Dependency validation (requires regime + volatility)
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_mod = import_plugin_module("fwbg-premium", "indicators", "regime_cluster")
Columns = _mod.Columns


@pytest.fixture
def df_with_regime_and_vol():
    """DataFrame with pre-computed regime and volatility columns."""
    np.random.seed(42)
    n = 800

    # Generate realistic OHLC
    returns = np.random.randn(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "O": close * (1 + np.random.randn(n) * 0.002),
        "H": close * (1 + np.abs(np.random.randn(n) * 0.005)),
        "L": close * (1 - np.abs(np.random.randn(n) * 0.005)),
        "C": close,
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

    # Add core input columns (simulated)
    df["regime_hurst_200"] = 0.5 + np.random.randn(n) * 0.1
    df["regime_entropy_100"] = 2.0 + np.random.randn(n) * 0.3
    df["regime_vr_200_5"] = 1.0 + np.random.randn(n) * 0.2
    df["vol_atr_pct_14_rank"] = np.random.rand(n)
    df["regime_hurst_divergence"] = np.random.randn(n) * 0.05

    return df


class TestRegimeClusterComputation:
    def test_score_computation(self, df_with_regime_and_vol):
        """Score is computed from z-scored inputs."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime_cluster")
        plugin = plugin_cls()

        result = plugin.compute(df_with_regime_and_vol)

        assert Columns.SCORE in result.columns
        # Score should have non-NaN values after warmup
        score = result[Columns.SCORE].dropna()
        assert len(score) > 0

    def test_cluster_labels(self, df_with_regime_and_vol):
        """Cluster labels are 0, 1, or 2."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime_cluster")
        plugin = plugin_cls()

        result = plugin.compute(df_with_regime_and_vol)

        cluster = result[Columns.LABEL].dropna()
        assert len(cluster) > 0
        assert set(cluster.unique()).issubset({0.0, 1.0, 2.0})

    def test_shift_features(self, df_with_regime_and_vol):
        """All outputs are shifted by 1 bar for lookahead prevention."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime_cluster")
        plugin = plugin_cls()

        result = plugin.compute(df_with_regime_and_vol)

        # First row should be NaN (shifted by 1)
        assert pd.isna(result[Columns.SCORE].iloc[0])
        assert pd.isna(result[Columns.LABEL].iloc[0])

    def test_n_inputs_core_only(self, df_with_regime_and_vol):
        """regime_cluster_n_inputs = 5 when only core inputs present."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime_cluster")
        plugin = plugin_cls()

        result = plugin.compute(df_with_regime_and_vol)

        n_inputs = result[Columns.N_INPUTS].dropna()
        assert len(n_inputs) > 0
        assert (n_inputs == 5).all()

    def test_optional_input_included(self, df_with_regime_and_vol):
        """regime_cluster_n_inputs = 6 when regime_risk_composite is present."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        df = df_with_regime_and_vol.copy()
        df["regime_risk_composite"] = np.random.randn(len(df)) * 0.5

        plugin_cls = registry.get("fwbg-premium:regime_cluster")
        plugin = plugin_cls()

        result = plugin.compute(df)

        n_inputs = result[Columns.N_INPUTS].dropna()
        assert (n_inputs == 6).all()

    def test_missing_core_input(self):
        """Score degrades gracefully when a core input is missing."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        np.random.seed(42)
        n = 800
        df = pd.DataFrame({
            "O": np.ones(n), "H": np.ones(n),
            "L": np.ones(n), "C": np.ones(n),
            # Only 3 of 5 core inputs
            "regime_hurst_200": np.random.randn(n),
            "regime_entropy_100": np.random.randn(n),
            "vol_atr_pct_14_rank": np.random.rand(n),
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        plugin_cls = registry.get("fwbg-premium:regime_cluster")
        plugin = plugin_cls()
        result = plugin.compute(df)

        n_inputs = result[Columns.N_INPUTS].dropna()
        assert (n_inputs == 3).all()

    def test_score_change_over_24_bars(self, df_with_regime_and_vol):
        """regime_cluster_score_chg is the 24-bar difference of the score."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime_cluster")
        plugin = plugin_cls()

        result = plugin.compute(df_with_regime_and_vol)

        assert Columns.SCORE_CHG in result.columns
        score_chg = result[Columns.SCORE_CHG].dropna()
        assert len(score_chg) > 0


class TestRegimeClusterPlugin:
    def test_plugin_discoverable(self):
        """Plugin is discoverable via registry."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime_cluster")
        assert plugin_cls is not None
        assert plugin_cls.name == "regime_cluster"

    def test_plugin_depends_on(self):
        """Plugin declares dependencies on regime and volatility."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime_cluster")
        assert "regime" in plugin_cls.depends_on
        assert "volatility" in plugin_cls.depends_on

    def test_plugin_feature_columns(self):
        """Plugin reports correct feature columns."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime_cluster")
        plugin = plugin_cls()
        cols = plugin.get_feature_columns()
        assert Columns.SCORE in cols
        assert Columns.LABEL in cols
        assert Columns.SCORE_CHG in cols
        assert Columns.N_INPUTS in cols

    def test_benefits_from_stationary_false(self):
        """regime_cluster should NOT benefit from stationary data."""
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()

        plugin_cls = registry.get("fwbg-premium:regime_cluster")
        assert plugin_cls.benefits_from_stationary is False


class TestRegimeClusterDependencyValidation:
    def test_pipeline_rejects_missing_dependency(self):
        """Pipeline should reject regime_cluster without regime or volatility."""
        from fwbg.pipeline import get_registry
        from fwbg.pipeline.config import PipelineConfig, PluginConfig
        from fwbg.pipeline.runner import PipelineRunner

        registry = get_registry()
        registry.auto_discover()

        # Only regime_cluster, missing both dependencies
        config = PipelineConfig(
            indicators=[
                PluginConfig(name="fwbg-premium:regime_cluster", params={}),
            ]
        )
        runner = PipelineRunner(registry, config)
        with pytest.raises(ValueError, match="depends on"):
            runner._initialize()

    def test_pipeline_accepts_with_dependencies(self):
        """Pipeline initializes when dependencies are present."""
        from fwbg.pipeline import get_registry
        from fwbg.pipeline.config import PipelineConfig, PluginConfig
        from fwbg.pipeline.runner import PipelineRunner

        registry = get_registry()
        registry.auto_discover()

        config = PipelineConfig(
            indicators=[
                PluginConfig(name="fwbg-core:volatility", params={}),
                PluginConfig(name="fwbg-premium:regime", params={}),
                PluginConfig(name="fwbg-premium:regime_cluster", params={}),
            ]
        )
        runner = PipelineRunner(registry, config)
        runner._initialize()

        # regime_cluster should come after both dependencies
        names = [c.name.split(":")[-1] for c, _ in runner._execution_order]
        assert names.index("regime_cluster") > names.index("regime")
        assert names.index("regime_cluster") > names.index("volatility")
