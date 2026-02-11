# tests/pipeline/test_fractional_diff_migration.py
"""Tests for FractionalDiffPreprocessor migration to new plugin system."""
import pytest
import pandas as pd
import numpy as np

from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext
from fwbg.builtins.preprocessing.fractional_diff import FractionalDiffPreprocessor


def test_fractional_diff_is_baseplugin():
    """FractionalDiffPreprocessor is BasePlugin subclass."""
    assert issubclass(FractionalDiffPreprocessor, BasePlugin)

    # Can be instantiated
    plugin = FractionalDiffPreprocessor()
    assert isinstance(plugin, BasePlugin)


def test_fractional_diff_attributes():
    """FractionalDiffPreprocessor has correct name, version, stateful=True, phase=PREPROCESSING."""
    plugin = FractionalDiffPreprocessor()

    # Required class attributes
    assert plugin.name == "fractional_diff"
    assert plugin.version == "2.0.0"
    assert plugin.phase == PluginPhase.PREPROCESSING

    # Stateful because it learns d and stores history
    assert plugin.stateful is True

    # Not cacheable because depends on fit state
    assert plugin.cacheable is False

    # Instance should start unfitted
    assert plugin._fitted is False


def test_fractional_diff_execute():
    """execute transforms DataFrame via PipelineContext."""
    plugin = FractionalDiffPreprocessor()

    # Create sample OHLC data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        "O": np.cumsum(np.random.randn(200)) + 100,
        "H": np.cumsum(np.random.randn(200)) + 102,
        "L": np.cumsum(np.random.randn(200)) + 98,
        "C": np.cumsum(np.random.randn(200)) + 100,
    }, index=dates)

    ctx = PipelineContext(
        df=df.copy(),
        symbol="TEST",
        asset_class="FOREX",
    )

    # Fit first (required for stateful plugin)
    plugin.fit(ctx, auto_d=False, default_d=0.4)
    assert plugin._fitted is True
    assert plugin.d_ == 0.4

    # Execute should transform the data
    result_ctx = plugin.execute(ctx, auto_d=False, default_d=0.4)

    # Result is a PipelineContext
    assert isinstance(result_ctx, PipelineContext)

    # DataFrame was transformed (rows may be dropped due to NaNs at start)
    assert len(result_ctx.df) <= len(df)
    assert len(result_ctx.df) > 0

    # Columns are still present
    assert all(col in result_ctx.df.columns for col in ["O", "H", "L", "C"])

    # Metadata should contain frac_diff_d
    assert result_ctx.df.attrs.get("frac_diff_d") == 0.4


def test_fractional_diff_validate():
    """validate returns True."""
    plugin = FractionalDiffPreprocessor()

    # Validation should pass
    assert plugin.validate() is True
