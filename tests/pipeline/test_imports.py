# tests/pipeline/test_imports.py
"""Tests for pipeline module public API imports."""
import pytest


def test_pipeline_public_api():
    """Pipeline module should export main classes."""
    from fwbg.pipeline import (
        PipelineContext,
        BasePlugin,
        PluginPhase,
        PluginRegistry,
        PluginNotFoundError,
        PipelineConfig,
        PluginConfig,
        PipelineRunner,
        get_registry,
    )

    # All imports should work
    assert PipelineContext is not None
    assert BasePlugin is not None
    assert PluginPhase is not None
    assert PluginRegistry is not None
    assert PipelineRunner is not None
