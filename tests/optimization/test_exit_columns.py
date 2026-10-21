"""Columns an exit reads by name stay in the fold frame without becoming model features."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import ta

from fwbg.core.config import StrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.data.assets import AssetConfig
from fwbg.optimization.process_fold import prepare_fold_data
from fwbg.optimization.robust_validation import WalkForwardFold
from fwbg.optimization.signal_fold import prepare_signal_fold_data
from fwbg.plugins import import_plugin_module

pytest.importorskip("fwbg_premium", reason="orb_based exit strategy is a premium plugin")

ASSET = AssetConfig(
    symbol="EURUSD", asset_class="FOREX", spread=0.0002, point=0.0001, currencies=["EUR", "USD"]
)


def _context(exit_params: dict) -> SimulationContext:
    strategy = StrategyConfig.from_dict(
        {
            "model": {"type": "xgboost", "required_features": ["feat1"]},
            "exit_strategies": [
                {"name": "orb_based", "params": {"tp_mult": 1.0, "sl_mult": 2.0, **exit_params}}
            ],
        }
    )
    return SimulationContext.create(ASSET, strategy)


def _fold() -> WalkForwardFold:
    rows = 400
    rng = np.random.default_rng(42)
    price = 1.1 + np.cumsum(rng.normal(0, 0.0005, rows))
    sparse = np.where(rng.random(rows) < 0.85, 20.0, np.nan)  # 15% NaN: dropped unless protected
    df = pd.DataFrame(
        {
            "O": price,
            "H": price + 0.001,
            "L": price - 0.001,
            "C": price,
            "rb8_orb_s00_sl_dist": sparse,
            "hl_ses_rl50_pdl_sl_dist": sparse,
            "feat1": rng.standard_normal(rows),
            "feat2": rng.standard_normal(rows),
        },
        index=pd.date_range("2020-01-01", periods=rows, freq="h"),
    )
    split = 280
    return WalkForwardFold(
        fold_id=0,
        train_start=0,
        train_end=split,
        test_start=split,
        test_end=rows,
        train_df=df.iloc[:split].copy(),
        test_df=df.iloc[split:].copy(),
    )


def _prepared_columns(ctx: SimulationContext) -> list[str]:
    train_df, _, _ = prepare_fold_data(
        _fold(),
        fold_indicators=[],
        precomputed_raw_df=None,
        preprocessing_configs=None,
        ctx=ctx,
        sym="TEST",
    )
    return list(train_df.columns)


def test_a_named_stop_column_is_not_a_required_model_feature():
    ctx = _context({"sl_dist_column": "rb8_orb_s00_sl_dist"})
    assert ctx.required_features == ["feat1"]


def test_a_named_stop_column_survives_the_fold_clean_up():
    columns = _prepared_columns(_context({"sl_dist_column": "rb8_orb_s00_sl_dist"}))
    assert "rb8_orb_s00_sl_dist" in columns
    assert "hl_ses_rl50_pdl_sl_dist" not in columns


def test_a_stop_column_pattern_survives_the_fold_clean_up():
    columns = _prepared_columns(_context({"sl_dist_column": "*_orb_s00_sl_dist"}))
    assert "rb8_orb_s00_sl_dist" in columns
    assert "hl_ses_rl50_pdl_sl_dist" not in columns


def _orb_exit():
    return import_plugin_module("fwbg-premium", "exit_strategies", "orb_based").OrbExitStrategy()


def _signal_context(exit_params: dict) -> SimulationContext:
    strategy = StrategyConfig.from_dict(
        {
            "model": {"type": "signal", "trade_directions": ["long", "short"]},
            "filters": {"allowed_hours": list(range(24))},
            "exit_strategies": [
                {"name": "orb_based", "params": {"tp_mult": 1.0, "sl_mult": 2.0, **exit_params}}
            ],
        }
    )
    return SimulationContext.create(ASSET, strategy)


def _signal_fold_frames(fold: WalkForwardFold, ctx: SimulationContext):
    train_df, test_df, _ = prepare_signal_fold_data(
        fold,
        fold_indicators=[],
        precomputed_raw_df=None,
        preprocessing_configs=None,
        ctx=ctx,
        sym="TEST",
    )
    return train_df, test_df


def _sparse_fold(column: str, values: float) -> WalkForwardFold:
    """A fold whose *column* is NaN on every other bar (5% NaN: kept by the clean-up)."""
    fold = _fold()
    for frame in (fold.train_df, fold.test_df):
        col = np.full(len(frame), values)
        col[::20] = np.nan
        frame[column] = col
    return fold


def test_signal_folds_do_not_zero_fill_a_named_level_column():
    ctx = _signal_context({"sl_level": "or_low", "sl_level_column": "rb8_orb_s00_or_low"})
    _, test_df = _signal_fold_frames(_sparse_fold("rb8_orb_s00_or_low", 1.09), ctx)
    level = test_df["rb8_orb_s00_or_low"]
    assert level.isna().any()
    assert not (level == 0).any()


def test_signal_fold_nan_stop_rows_fall_back_to_atr():
    ctx = _signal_context({"sl_dist_column": "rb8_orb_s00_sl_dist", "min_sl_pips": 1})
    fold = _sparse_fold("rb8_orb_s00_sl_dist", 0.004)
    _, test_df = _signal_fold_frames(fold, ctx)
    exit_ctx = replace(ctx, exit_strategy="orb_based", exit_params=ctx.exit_strategies[0].params)
    _, sl_dists = _orb_exit().resolve_distances(test_df, 1.0, 2.0, exit_ctx)
    nan_rows = fold.test_df["rb8_orb_s00_sl_dist"].isna().to_numpy()
    atr = ta.volatility.average_true_range(test_df["H"], test_df["L"], test_df["C"], window=14)
    expected = np.maximum(np.nan_to_num(atr.to_numpy()) * 2.0, ctx.spread)
    np.testing.assert_allclose(sl_dists[nan_rows], expected[nan_rows])
