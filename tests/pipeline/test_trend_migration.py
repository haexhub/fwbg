# tests/pipeline/test_trend_migration.py
"""Tests for TrendIndicators migration to new plugin system."""
import pytest
import pandas as pd
import numpy as np

from fwbg.pipeline.base import BasePlugin, PluginPhase
from fwbg.pipeline.context import PipelineContext
from fwbg.builtins.indicators.trend import TrendIndicators


def test_trend_is_baseplugin():
    """TrendIndicators is BasePlugin subclass, phase is INDICATORS."""
    assert issubclass(TrendIndicators, BasePlugin)

    # Can be instantiated
    plugin = TrendIndicators()
    assert isinstance(plugin, BasePlugin)

    # Phase must be INDICATORS
    assert plugin.phase == PluginPhase.INDICATORS


def test_trend_attributes():
    """TrendIndicators has name='trend', version, stateful=False."""
    plugin = TrendIndicators()

    # Required class attributes
    assert plugin.name == "trend"
    assert plugin.version == "2.0.0"
    assert plugin.phase == PluginPhase.INDICATORS

    # Indicators are stateless
    assert plugin.stateful is False

    # Indicators are cacheable
    assert plugin.cacheable is True


def test_trend_execute():
    """execute adds indicator columns to ctx.df."""
    plugin = TrendIndicators()

    # Create sample OHLC data with enough history for indicators
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    df = pd.DataFrame({
        "O": np.cumsum(np.random.randn(300)) + 100,
        "H": np.cumsum(np.random.randn(300)) + 102,
        "L": np.cumsum(np.random.randn(300)) + 98,
        "C": np.cumsum(np.random.randn(300)) + 100,
    }, index=dates)
    # Ensure H > L for valid OHLC
    df["H"] = df[["O", "H", "C"]].max(axis=1) + 0.5
    df["L"] = df[["O", "L", "C"]].min(axis=1) - 0.5

    ctx = PipelineContext(
        df=df.copy(),
        symbol="TEST",
        asset_class="FOREX",
    )

    original_cols = set(ctx.df.columns)

    # Execute should add indicator columns
    result_ctx = plugin.execute(ctx)

    # Result is a PipelineContext
    assert isinstance(result_ctx, PipelineContext)

    # DataFrame should have new columns
    new_cols = set(result_ctx.df.columns) - original_cols
    assert len(new_cols) > 0

    # Should have trend indicator columns
    assert "trend_adx_14" in result_ctx.df.columns
    assert "trend_macd" in result_ctx.df.columns
    assert "trend_ema_dist_21" in result_ctx.df.columns


def test_trend_get_feature_columns():
    """get_feature_columns returns list of created columns."""
    plugin = TrendIndicators()

    feature_cols = plugin.get_feature_columns()

    # Should return a list
    assert isinstance(feature_cols, list)

    # Should have expected columns
    assert len(feature_cols) > 0
    assert "trend_adx_7" in feature_cols
    assert "trend_adx_14" in feature_cols
    assert "trend_adx_21" in feature_cols
    assert "trend_macd" in feature_cols
    assert "trend_macd_signal" in feature_cols
    assert "trend_ema_dist_8" in feature_cols
    assert "trend_aroon_up" in feature_cols
    assert "trend_aroon_down" in feature_cols


def test_trend_validate():
    """validate returns True."""
    plugin = TrendIndicators()

    # Validation should pass
    assert plugin.validate() is True
