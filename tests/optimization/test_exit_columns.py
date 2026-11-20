"""Columns an exit reads by name stay in the fold frame without becoming model features."""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import StrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.data.assets import AssetConfig
from fwbg.optimization import process_fold
from fwbg.optimization.process_fold import prepare_fold_data, process_single_fold
from fwbg.optimization.robust_validation import WalkForwardFold
from fwbg.optimization.signal_fold import prepare_signal_fold_data

ASSET = AssetConfig(
    symbol="EURUSD",
    asset_class="FOREX",
    spread=0.0002,
    point=0.0001,
    currencies=["EUR", "USD"],
)


def _context(
    exit_params: dict, required: tuple[str, ...] = ("feat1",)
) -> SimulationContext:
    strategy = StrategyConfig.from_dict(
        {
            "model": {"type": "xgboost", "required_features": list(required)},
            "exit_strategies": [
                {
                    "name": "fixed",
                    "params": {"tp_mult": 1.0, "sl_mult": 2.0, **exit_params},
                }
            ],
        }
    )
    return SimulationContext.create(ASSET, strategy)


def _fold() -> WalkForwardFold:
    rows = 400
    rng = np.random.default_rng(42)
    price = 1.1 + np.cumsum(rng.normal(0, 0.0005, rows))
    sparse = np.where(
        rng.random(rows) < 0.85, 20.0, np.nan
    )  # 15% NaN: dropped unless protected
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


def _signal_context(exit_params: dict) -> SimulationContext:
    strategy = StrategyConfig.from_dict(
        {
            "model": {"type": "signal", "trade_directions": ["long", "short"]},
            "filters": {"allowed_hours": list(range(24))},
            "exit_strategies": [
                {
                    "name": "fixed",
                    "params": {"tp_mult": 1.0, "sl_mult": 2.0, **exit_params},
                }
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
    ctx = _signal_context(
        {"sl_level": "or_low", "sl_level_column": "rb8_orb_s00_or_low"}
    )
    _, test_df = _signal_fold_frames(_sparse_fold("rb8_orb_s00_or_low", 1.09), ctx)
    level = test_df["rb8_orb_s00_or_low"]
    assert level.isna().any()
    assert not (level == 0).any()


def test_a_fold_without_selected_features_is_skipped_not_trained_on_the_stop(
    monkeypatch,
):
    """The named stop column is no forced feature: with nothing selected the
    fold is skipped instead of training a model on the stop column alone."""
    monkeypatch.setattr(process_fold, "select_features", lambda *args: (None, None))
    monkeypatch.setattr(
        process_fold, "run_grid_search", lambda *a, **k: pytest.fail("trained")
    )
    ctx = _context({"sl_dist_column": "rb8_orb_s00_sl_dist"}, required=())
    result = process_single_fold(_fold(), 0, 1, [], None, None, ctx, "TEST", 0)
    assert result == (None, [])
