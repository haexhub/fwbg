"""Columns an exit reads by name stay in the fold frame without becoming model features."""

import numpy as np
import pandas as pd

from fwbg.core.config import StrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.data.assets import AssetConfig
from fwbg.optimization.process_fold import prepare_fold_data
from fwbg.optimization.robust_validation import WalkForwardFold

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
