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
    """Train-slice targets must not change when val bars are mutated."""
    ctx = _make_mock_ctx(max_trade_bars=10, embargo_bars=10)
    full_df = _make_synthetic_df(200)
    # First 100 bars are "train", next 100 are "val".
    train_df = full_df.iloc[:100]

    # Pre-compute targets on the full set (current cached-precompute path).
    long_full, short_full = compute_targets_cached(full_df, 20, 10, ctx)

    # Mutate val region with a massive favourable spike for longs.
    mutated = full_df.copy()
    mutated.iloc[100:, mutated.columns.get_loc("H")] = 999.0
    long_mut, short_mut = compute_targets_cached(mutated, 20, 10, ctx)

    slice_train_long, slice_train_short, _, _ = slice_targets_for_fold(
        long_full, short_full, full_df, train_df, ctx,
    )
    slice_train_long_mut, slice_train_short_mut, _, _ = slice_targets_for_fold(
        long_mut, short_mut, mutated, train_df, ctx,
    )

    # After the fix, train-slice targets at every bar must be identical
    # (or jointly NaN in the embargo zone), regardless of val-region mutations.
    np.testing.assert_array_equal(
        np.isnan(slice_train_long), np.isnan(slice_train_long_mut),
        err_msg="Train long target NaN-mask changed when val data was mutated.",
    )
    np.testing.assert_array_equal(
        np.isnan(slice_train_short), np.isnan(slice_train_short_mut),
        err_msg="Train short target NaN-mask changed when val data was mutated.",
    )
    finite_mask = ~np.isnan(slice_train_long)
    np.testing.assert_array_equal(
        slice_train_long[finite_mask], slice_train_long_mut[finite_mask],
        err_msg="Train long targets changed when val data was mutated — embargo leak.",
    )
    finite_mask = ~np.isnan(slice_train_short)
    np.testing.assert_array_equal(
        slice_train_short[finite_mask], slice_train_short_mut[finite_mask],
        err_msg="Train short targets changed when val data was mutated — embargo leak.",
    )
