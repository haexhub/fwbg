"""Tests for PipelineRunner with phase ordering and stateful plugin support."""
import pandas as pd
import pytest

from fwbg_sdk import BasePlugin, PluginPhase, PipelineContext
from fwbg.pipeline.config import PipelineConfig, PluginConfig
from fwbg.pipeline.registry import PluginNotFoundError, PluginRegistry
from fwbg.pipeline.runner import PipelineRunner


# Test helper plugins
class AddColumnPlugin(BasePlugin):
    """Adds a column with a constant value (INDICATORS phase)."""

    name = "add_column"
    version = "1.0.0"
    phase = PluginPhase.INDICATORS

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        column_name = params.get("column_name", "new_column")
        value = params.get("value", 0)
        ctx.df[column_name] = value
        return ctx

    def validate(self) -> bool:
        return True

    @classmethod
    def get_default_params(cls):
        return {"column_name": "new_column", "value": 0}


class MultiplyPlugin(BasePlugin):
    """Multiplies a column by a factor (PREPROCESSING phase)."""

    name = "multiply"
    version = "1.0.0"
    phase = PluginPhase.PREPROCESSING

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        column = params.get("column", "value")
        factor = params.get("factor", 1)
        ctx.df[column] = ctx.df[column] * factor
        return ctx

    def validate(self) -> bool:
        return True

    @classmethod
    def get_default_params(cls):
        return {"column": "value", "factor": 1}


class StatefulPlugin(BasePlugin):
    """Subtracts the mean learned during fit (PREPROCESSING phase, stateful)."""

    name = "stateful_mean_subtractor"
    version = "1.0.0"
    phase = PluginPhase.PREPROCESSING
    stateful = True

    def __init__(self):
        super().__init__()
        self._mean = None

    def fit(self, ctx: PipelineContext, **params) -> None:
        column = params.get("column", "value")
        self._mean = ctx.df[column].mean()
        self._fitted = True

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        column = params.get("column", "value")
        if self._mean is not None:
            ctx.df[column] = ctx.df[column] - self._mean
        return ctx

    def validate(self) -> bool:
        return True

    def reset(self) -> None:
        super().reset()
        self._mean = None

    @classmethod
    def get_default_params(cls):
        return {"column": "value"}


class InvalidPlugin(BasePlugin):
    """A plugin that fails validation."""

    name = "invalid_plugin"
    version = "1.0.0"
    phase = PluginPhase.PREPROCESSING

    def execute(self, ctx: PipelineContext, **params) -> PipelineContext:
        return ctx

    def validate(self) -> bool:
        return False


# Namespace for test plugins
TEST_NS = "test"


@pytest.fixture
def registry():
    """Create a registry with test plugins registered using namespace."""
    reg = PluginRegistry()
    reg.register(AddColumnPlugin, namespace=TEST_NS)
    reg.register(MultiplyPlugin, namespace=TEST_NS)
    reg.register(StatefulPlugin, namespace=TEST_NS)
    reg.register(InvalidPlugin, namespace=TEST_NS)
    return reg


@pytest.fixture
def sample_context():
    """Create a sample PipelineContext with test data."""
    df = pd.DataFrame({"value": [10.0, 20.0, 30.0, 40.0, 50.0]})
    return PipelineContext(df=df, symbol="EURUSD", asset_class="FOREX")


class TestPipelineRunner:
    """Tests for PipelineRunner class."""

    def test_runner_execute_single_plugin(self, registry, sample_context):
        """Test running a single plugin through the runner."""
        config = PipelineConfig(
            indicators=[PluginConfig(name=f"{TEST_NS}:add_column", params={"column_name": "test_col", "value": 42})]
        )
        runner = PipelineRunner(registry, config)

        result = runner.run(sample_context)

        assert "test_col" in result.df.columns
        assert (result.df["test_col"] == 42).all()

    def test_runner_execute_multiple_plugins(self, registry, sample_context):
        """Test plugins execute in phase order (preprocessing before indicators)."""
        # Multiply is PREPROCESSING phase, AddColumn is INDICATORS phase
        # Even if listed in different order in config, should execute in phase order
        config = PipelineConfig(
            indicators=[PluginConfig(name=f"{TEST_NS}:add_column", params={"column_name": "indicator_col", "value": 100})],
            preprocessing=[PluginConfig(name=f"{TEST_NS}:multiply", params={"column": "value", "factor": 2})],
        )
        runner = PipelineRunner(registry, config)

        result = runner.run(sample_context)

        # Multiply should have run first (PREPROCESSING)
        # Original values [10, 20, 30, 40, 50] * 2 = [20, 40, 60, 80, 100]
        assert list(result.df["value"]) == [20.0, 40.0, 60.0, 80.0, 100.0]

        # AddColumn should have run second (INDICATORS)
        assert "indicator_col" in result.df.columns
        assert (result.df["indicator_col"] == 100).all()

    def test_runner_stateful_plugin_fit_transform(self, registry):
        """Test stateful plugins learn from training data and apply to test data."""
        # Training data: mean = 30
        train_df = pd.DataFrame({"value": [10.0, 20.0, 30.0, 40.0, 50.0]})
        train_ctx = PipelineContext(df=train_df, symbol="EURUSD", asset_class="FOREX")

        # Test data: values should have train mean (30) subtracted
        test_df = pd.DataFrame({"value": [100.0, 200.0, 300.0]})
        test_ctx = PipelineContext(df=test_df, symbol="EURUSD", asset_class="FOREX")

        config = PipelineConfig(
            preprocessing=[PluginConfig(name=f"{TEST_NS}:stateful_mean_subtractor", params={"column": "value"})]
        )
        runner = PipelineRunner(registry, config)

        # Fit on training data
        runner.fit(train_ctx)

        # Run on test data - should subtract train mean (30)
        result = runner.run(test_ctx)

        # 100-30=70, 200-30=170, 300-30=270
        assert list(result.df["value"]) == [70.0, 170.0, 270.0]

    def test_runner_validate_before_run(self, registry, sample_context):
        """Test validate returns validation results for all plugins."""
        config = PipelineConfig(
            preprocessing=[
                PluginConfig(name=f"{TEST_NS}:multiply"),
                PluginConfig(name=f"{TEST_NS}:invalid_plugin"),
            ],
            indicators=[PluginConfig(name=f"{TEST_NS}:add_column")],
        )
        runner = PipelineRunner(registry, config)

        results = runner.validate()

        # Should have results for all three plugins (keyed by FQN)
        assert f"{TEST_NS}:multiply" in results
        assert f"{TEST_NS}:invalid_plugin" in results
        assert f"{TEST_NS}:add_column" in results

        # Valid plugins should pass
        assert results[f"{TEST_NS}:multiply"]["valid"] is True
        assert results[f"{TEST_NS}:add_column"]["valid"] is True

        # Invalid plugin should fail
        assert results[f"{TEST_NS}:invalid_plugin"]["valid"] is False

    def test_runner_unknown_plugin_error(self, registry):
        """Test that unknown plugin names raise PluginNotFoundError."""
        config = PipelineConfig(
            preprocessing=[PluginConfig(name=f"{TEST_NS}:nonexistent_plugin")]
        )
        runner = PipelineRunner(registry, config)

        with pytest.raises(PluginNotFoundError) as exc_info:
            runner.run(PipelineContext(df=pd.DataFrame(), symbol="TEST", asset_class="TEST"))

        assert "nonexistent_plugin" in str(exc_info.value)

    def test_runner_reset_stateful_plugins(self, registry):
        """Test reset clears fitted state of stateful plugins."""
        train_df = pd.DataFrame({"value": [10.0, 20.0, 30.0, 40.0, 50.0]})
        train_ctx = PipelineContext(df=train_df, symbol="EURUSD", asset_class="FOREX")

        config = PipelineConfig(
            preprocessing=[PluginConfig(name=f"{TEST_NS}:stateful_mean_subtractor", params={"column": "value"})]
        )
        runner = PipelineRunner(registry, config)

        # Fit the stateful plugin
        runner.fit(train_ctx)

        # Get instance and verify it's fitted (use FQN)
        instance = runner.get_instance(f"{TEST_NS}:stateful_mean_subtractor")
        assert instance is not None
        assert instance._fitted is True
        assert instance._mean == 30.0

        # Reset the runner
        runner.reset()

        # Verify the plugin was reset
        assert instance._fitted is False
        assert instance._mean is None
