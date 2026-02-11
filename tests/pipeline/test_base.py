# tests/pipeline/test_base.py
"""Tests for BasePlugin abstract class and PluginPhase enum."""
import pytest
import pandas as pd
from typing import Dict, Any

from fwbg.pipeline.context import PipelineContext
from fwbg.pipeline.base import BasePlugin, PluginPhase


def test_plugin_phase_enum():
    """Verify all 7 phase values exist and are correct."""
    assert PluginPhase.DATA_LOADING.value == "data_loading"
    assert PluginPhase.PREPROCESSING.value == "preprocessing"
    assert PluginPhase.INDICATORS.value == "indicators"
    assert PluginPhase.FEATURE_SELECTION.value == "feature_selection"
    assert PluginPhase.LABELING.value == "labeling"
    assert PluginPhase.MODEL.value == "model"
    assert PluginPhase.VALIDATION.value == "validation"

    # Verify exactly 7 phases
    assert len(PluginPhase) == 7


def test_base_plugin_is_abstract():
    """BasePlugin() raises TypeError because it's abstract."""
    with pytest.raises(TypeError):
        BasePlugin()


def test_base_plugin_required_attributes():
    """Incomplete plugin (missing attrs) raises TypeError."""
    # Missing name
    with pytest.raises(TypeError, match="name"):

        class MissingName(BasePlugin):
            version = "1.0.0"
            phase = PluginPhase.PREPROCESSING

            def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
                return ctx

            def validate(self) -> bool:
                return True

    # Missing version
    with pytest.raises(TypeError, match="version"):

        class MissingVersion(BasePlugin):
            name = "test_plugin"
            phase = PluginPhase.PREPROCESSING

            def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
                return ctx

            def validate(self) -> bool:
                return True

    # Missing phase
    with pytest.raises(TypeError, match="phase"):

        class MissingPhase(BasePlugin):
            name = "test_plugin"
            version = "1.0.0"

            def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
                return ctx

            def validate(self) -> bool:
                return True


def test_concrete_plugin_implementation():
    """Complete plugin can be instantiated, verify defaults."""

    class ConcretePlugin(BasePlugin):
        name = "concrete_plugin"
        version = "1.0.0"
        phase = PluginPhase.INDICATORS

        def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
            return ctx

        def validate(self) -> bool:
            return True

    plugin = ConcretePlugin()

    # Verify class attributes
    assert plugin.name == "concrete_plugin"
    assert plugin.version == "1.0.0"
    assert plugin.phase == PluginPhase.INDICATORS

    # Verify defaults
    assert plugin.stateful is False
    assert plugin.cacheable is True

    # Verify instance state
    assert plugin._fitted is False

    # Verify default methods return expected values
    assert plugin.get_default_params() == {}
    assert plugin.get_feature_columns() == []


def test_plugin_execute():
    """Plugin execute processes context with params."""

    class ProcessingPlugin(BasePlugin):
        name = "processing_plugin"
        version = "1.0.0"
        phase = PluginPhase.PREPROCESSING

        def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
            # Process: multiply column by factor param
            factor = params.get("factor", 1)
            ctx.df["processed"] = ctx.df["value"] * factor
            ctx.metadata["processed_by"] = self.name
            return ctx

        def validate(self) -> bool:
            return True

    plugin = ProcessingPlugin()
    df = pd.DataFrame({"value": [1, 2, 3]})
    ctx = PipelineContext(df=df, symbol="TEST", asset_class="TEST")

    result = plugin.execute(ctx, factor=10)

    assert "processed" in result.df.columns
    assert result.df["processed"].tolist() == [10, 20, 30]
    assert result.metadata["processed_by"] == "processing_plugin"


def test_plugin_stateful_flag():
    """Stateful plugin with stateful=True and fit() method."""

    class StatefulPlugin(BasePlugin):
        name = "stateful_plugin"
        version = "1.0.0"
        phase = PluginPhase.MODEL
        stateful = True

        def __init__(self):
            super().__init__()
            self.learned_value = None

        def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
            if not self._fitted:
                raise RuntimeError("Plugin must be fitted first")
            ctx.df["prediction"] = self.learned_value
            return ctx

        def fit(self, ctx: PipelineContext, **params) -> None:
            self.learned_value = ctx.df["value"].mean()
            super().fit(ctx, **params)

        def validate(self) -> bool:
            return True

    plugin = StatefulPlugin()

    # Verify stateful flag
    assert plugin.stateful is True
    assert plugin._fitted is False

    # Fit the plugin
    df = pd.DataFrame({"value": [10, 20, 30]})
    ctx = PipelineContext(df=df, symbol="TEST", asset_class="TEST")
    plugin.fit(ctx)

    assert plugin._fitted is True
    assert plugin.learned_value == 20.0

    # Execute after fit
    result = plugin.execute(ctx)
    assert result.df["prediction"].tolist() == [20.0, 20.0, 20.0]

    # Reset and verify
    plugin.reset()
    assert plugin._fitted is False


def test_plugin_default_params():
    """Plugin get_default_params() returns custom defaults."""

    class ParameterizedPlugin(BasePlugin):
        name = "parameterized_plugin"
        version = "1.0.0"
        phase = PluginPhase.INDICATORS

        @classmethod
        def get_default_params(cls) -> Dict[str, Any]:
            return {
                "window": 14,
                "method": "sma",
                "threshold": 0.5,
            }

        def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
            defaults = self.get_default_params()
            merged = {**defaults, **params}
            ctx.metadata["params_used"] = merged
            return ctx

        def validate(self) -> bool:
            return True

    plugin = ParameterizedPlugin()

    # Verify custom defaults
    defaults = plugin.get_default_params()
    assert defaults["window"] == 14
    assert defaults["method"] == "sma"
    assert defaults["threshold"] == 0.5

    # Verify defaults can be overridden in execute
    df = pd.DataFrame({"value": [1, 2, 3]})
    ctx = PipelineContext(df=df, symbol="TEST", asset_class="TEST")
    result = plugin.execute(ctx, window=20)

    assert result.metadata["params_used"]["window"] == 20
    assert result.metadata["params_used"]["method"] == "sma"
