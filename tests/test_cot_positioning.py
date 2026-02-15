"""Tests for COT Positioning DataLoader."""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.registry import get_data_loader


def _make_df_with_cot(n=5000):
    """Create OHLC DataFrame with mock COT net position columns."""
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")

    df = pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.005,
        "L": close * 0.995,
        "C": close,
        # Mock COT net positions (weekly data forward-filled to hourly)
        "macro_cot_eurusd": (50000 + rng.normal(0, 5000, n).cumsum()).round(),
        "macro_cot_usdjpy": (-20000 + rng.normal(0, 3000, n).cumsum()).round(),
    }, index=idx)
    return df


class TestCOTPositioningLoader:
    def test_plugin_registered(self):
        """cot_positioning should be discoverable via registry."""
        cls = get_data_loader("cot_positioning")
        assert cls is not None

    def test_zscore_computed(self):
        """Z-score features should be present for each COT column."""
        cls = get_data_loader("cot_positioning")
        loader = cls()
        df = _make_df_with_cot()

        from fwbg.pipeline.context import PipelineContext
        ctx = PipelineContext(df=df, symbol="EURUSD", asset_class="forex")
        ctx = loader.execute(ctx)

        assert "cot_eurusd_zscore" in ctx.df.columns
        assert "cot_usdjpy_zscore" in ctx.df.columns

    def test_extreme_flags_are_binary(self):
        """Extreme long/short flags should be 0 or 1."""
        cls = get_data_loader("cot_positioning")
        loader = cls()
        df = _make_df_with_cot()

        from fwbg.pipeline.context import PipelineContext
        ctx = PipelineContext(df=df, symbol="EURUSD", asset_class="forex")
        ctx = loader.execute(ctx)

        for col in ["cot_eurusd_extreme_long", "cot_eurusd_extreme_short",
                     "cot_eurusd_crowded"]:
            vals = ctx.df[col].dropna().unique()
            assert set(vals).issubset({0.0, 1.0}), f"{col} has non-binary: {vals}"

    def test_weekly_momentum_columns(self):
        """Week-based momentum columns should be present."""
        cls = get_data_loader("cot_positioning")
        loader = cls()
        df = _make_df_with_cot()

        from fwbg.pipeline.context import PipelineContext
        ctx = PipelineContext(df=df, symbol="EURUSD", asset_class="forex")
        ctx = loader.execute(ctx)

        for lb in [1, 4, 12, 26]:
            assert f"cot_eurusd_chg_{lb}w" in ctx.df.columns

    def test_features_shifted(self):
        """All COT features should be shifted by 1 bar (lookahead prevention)."""
        cls = get_data_loader("cot_positioning")
        loader = cls()
        df = _make_df_with_cot()

        from fwbg.pipeline.context import PipelineContext
        ctx = PipelineContext(df=df, symbol="EURUSD", asset_class="forex")
        ctx = loader.execute(ctx)

        # First row should be NaN for all COT features (shifted)
        for col in ["cot_eurusd_zscore", "cot_eurusd_crowded"]:
            assert pd.isna(ctx.df[col].iloc[0]), f"{col} first row should be NaN"

    def test_graceful_without_cot_columns(self):
        """Should not crash if no COT columns present."""
        cls = get_data_loader("cot_positioning")
        loader = cls()

        n = 200
        close = np.full(n, 100.0)
        df = pd.DataFrame({
            "O": close, "H": close * 1.01, "L": close * 0.99, "C": close,
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        from fwbg.pipeline.context import PipelineContext
        ctx = PipelineContext(df=df, symbol="EURUSD", asset_class="forex")
        ctx = loader.execute(ctx)

        # No crash, original columns preserved
        assert "O" in ctx.df.columns
        assert "C" in ctx.df.columns

    def test_get_feature_columns(self):
        """get_feature_columns should list all expected columns."""
        cls = get_data_loader("cot_positioning")
        loader = cls()
        cols = loader.get_feature_columns()

        assert "cot_eurusd_zscore" in cols
        assert "cot_eurusd_extreme_long" in cols
        assert "cot_eurusd_chg_1w" in cols
        assert "cot_eurusd_chg_26w" in cols
