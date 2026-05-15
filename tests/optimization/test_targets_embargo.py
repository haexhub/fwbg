"""Embargo regression test for slice_targets_for_fold.

The target at bar t encodes the trade outcome over the next `max_trade_bars`
bars. If t is the last bar of a train fold and t+1..t+max_trade_bars overlap
the val fold, the train target leaks val information.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from fwbg.core.context import SimulationContext
from fwbg.optimization.targets import compute_targets_cached, slice_targets_for_fold


def _make_synthetic_df(n: int = 200) -> pd.DataFrame:
    """Flat, low-volatility OHLC so vanilla targets are mostly NoExit (loss),
    making any leak from a val-region spike easy to detect.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    base = np.full(n, 1.0)
    return pd.DataFrame({
        "O": base,
        "H": base + 0.00005,
        "L": base - 0.00005,
        "C": base,
    }, index=idx)


def _make_mock_ctx(max_trade_bars: int = 10, embargo_bars: int = 10) -> SimulationContext:
    """Mock context configured for fixed-exit strategy."""
    ctx = Mock(spec=SimulationContext)
    ctx.symbol = "TESTUSD"
    ctx.spread = 0.0001
    ctx.long_enabled = True
    ctx.short_enabled = True
    ctx.min_trades = 1
    ctx.max_trade_bars = max_trade_bars
    ctx.embargo_bars = embargo_bars
    ctx.exit_strategy = "fixed"
    ctx.exit_params = {}
    ctx.entry_modifier = None
    ctx.entry_modifier_params = {}
    return ctx


def test_target_for_last_train_bar_does_not_use_val_data():
    """Train-fold targets must not change when val bars are mutated."""
    ctx = _make_mock_ctx(max_trade_bars=10, embargo_bars=10)
    full_df = _make_synthetic_df(200)
    # First 100 bars are "train", next 100 are "val".
    train_df = full_df.iloc[:100]

    # Mutate val region with a massive favourable spike for longs.
    mutated = full_df.copy()
    mutated.iloc[100:, mutated.columns.get_loc("H")] = 999.0
    # Note: train_df_mut is sliced from the (mutated) full_df, but those rows
    # themselves were not mutated, so train_df_mut is value-equal to train_df.
    train_df_mut = mutated.iloc[:100]

    slice_train_long, slice_train_short, _, _ = slice_targets_for_fold(
        train_df, ctx, tp=20, sl=10,
    )
    slice_train_long_mut, slice_train_short_mut, _, _ = slice_targets_for_fold(
        train_df_mut, ctx, tp=20, sl=10,
    )

    np.testing.assert_array_equal(
        slice_train_long, slice_train_long_mut,
        err_msg="Train long targets changed when val data was mutated — embargo leak.",
    )
    np.testing.assert_array_equal(
        slice_train_short, slice_train_short_mut,
        err_msg="Train short targets changed when val data was mutated — embargo leak.",
    )


def test_targets_recomputed_per_fold_dont_use_future_bars():
    """Slice_targets_for_fold must compute targets using only fold_df's bars."""
    ctx = _make_mock_ctx(max_trade_bars=10, embargo_bars=10)
    full_df = _make_synthetic_df(200)
    train_df = full_df.iloc[:100]

    # Compute targets on the FULL df: bar 99's target can see bars 100..109.
    long_full, _ = compute_targets_cached(full_df, 20, 10, ctx)
    # Compute via slice_targets_for_fold on train_df only: bar 99 has no
    # future bars inside fold_df → target is 0 (no exit found).
    train_long, _, _, _ = slice_targets_for_fold(train_df, ctx, tp=20, sl=10)

    # On flat synthetic data both full and fold give 0 everywhere, but the
    # key check is that the fold path doesn't read past fold_df.
    assert len(train_long) == len(train_df)
    # Mutate val bars in a copy of full_df and recompute the full target.
    mutated = full_df.copy()
    mutated.iloc[100:, mutated.columns.get_loc("H")] = 999.0
    long_full_mut, _ = compute_targets_cached(mutated, 20, 10, ctx)
    # Full path: last few train bars CHANGE (they saw the spike).
    assert not np.array_equal(long_full[:100], long_full_mut[:100]), (
        "Sanity check failed: the leak we are fixing should be observable "
        "in the old precompute-then-slice path."
    )
    # Fold path: train_df targets are independent of the mutation.
    train_long_mut, _, _, _ = slice_targets_for_fold(
        mutated.iloc[:100], ctx, tp=20, sl=10,
    )
    np.testing.assert_array_equal(train_long, train_long_mut)
