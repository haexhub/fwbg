# tests/pipeline/test_context.py
import pytest
import pandas as pd
from fwbg_sdk import PipelineContext


def test_pipeline_context_creation():
    """PipelineContext should store DataFrame and metadata."""
    df = pd.DataFrame({"O": [1, 2], "H": [2, 3], "L": [0.5, 1.5], "C": [1.5, 2.5]})
    ctx = PipelineContext(df=df, symbol="EURUSD", asset_class="FOREX")

    assert ctx.df is not None
    assert len(ctx.df) == 2
    assert ctx.symbol == "EURUSD"
    assert ctx.asset_class == "FOREX"
    assert ctx.metadata == {}


def test_pipeline_context_metadata():
    """PipelineContext should allow storing arbitrary metadata."""
    df = pd.DataFrame({"C": [1, 2, 3]})
    ctx = PipelineContext(df=df, symbol="BTCUSD", asset_class="CRYPTO")

    ctx.metadata["fitted_d"] = 0.4
    ctx.metadata["selected_features"] = ["rsi_14", "ema_20"]

    assert ctx.metadata["fitted_d"] == 0.4
    assert len(ctx.metadata["selected_features"]) == 2


def test_pipeline_context_immutable_df_reference():
    """Updating df should create new reference, not mutate."""
    df1 = pd.DataFrame({"C": [1, 2]})
    ctx = PipelineContext(df=df1, symbol="TEST", asset_class="FOREX")

    df2 = pd.DataFrame({"C": [1, 2, 3, 4]})
    ctx.df = df2

    assert len(ctx.df) == 4
    assert len(df1) == 2  # Original unchanged


def test_pipeline_context_clone():
    """Clone should create independent copy."""
    df = pd.DataFrame({"C": [1, 2, 3]})
    ctx = PipelineContext(df=df, symbol="EURUSD", asset_class="FOREX")
    ctx.metadata["key"] = "value"
    ctx.fold_info = {"fold": 1}

    cloned = ctx.clone()

    # Verify cloned has same values
    assert len(cloned.df) == 3
    assert cloned.symbol == "EURUSD"
    assert cloned.asset_class == "FOREX"
    assert cloned.metadata["key"] == "value"
    assert cloned.fold_info["fold"] == 1

    # Verify modifications don't affect original
    cloned.df["C"] = [10, 20, 30]
    cloned.metadata["key"] = "modified"
    cloned.fold_info["fold"] = 99

    assert ctx.df["C"].tolist() == [1, 2, 3]
    assert ctx.metadata["key"] == "value"
    assert ctx.fold_info["fold"] == 1
