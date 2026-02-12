"""
Tests für Purged CV und Sample Weights (López de Prado, AFML Ch. 4+7).

Testet:
- ValidationConfig embargo_bars und sample_weights Felder
- Embargo-Gap in Fold-Erstellung (outer + inner folds)
- Concurrent Label Counting und Sample Uniqueness Weights
"""
import pytest
import numpy as np
import pandas as pd

from fwbg.core.config import ValidationConfig
from fwbg.optimization.robust_validation import create_walk_forward_folds
from fwbg.optimization.nested_cv import nested_cv_split


class TestValidationConfigEmbargo:
    def test_embargo_bars_from_dict(self):
        config = ValidationConfig.from_dict({"embargo_bars": 100})
        assert config.embargo_bars == 100

    def test_embargo_bars_default(self):
        config = ValidationConfig.from_dict({})
        assert config.embargo_bars == 0

    def test_sample_weights_from_dict(self):
        config = ValidationConfig.from_dict({"sample_weights": True})
        assert config.sample_weights is True

    def test_sample_weights_default(self):
        config = ValidationConfig.from_dict({})
        assert config.sample_weights is False


def _make_df(n_rows: int) -> pd.DataFrame:
    """Helper: DataFrame with OHLC columns."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {"O": rng.standard_normal(n_rows), "H": rng.standard_normal(n_rows),
         "L": rng.standard_normal(n_rows), "C": rng.standard_normal(n_rows)},
        index=pd.date_range("2020-01-01", periods=n_rows, freq="h"),
    )


class TestOuterFoldEmbargo:
    def test_embargo_gap(self):
        df = _make_df(50000)
        folds = create_walk_forward_folds(df, n_folds=3, test_size=4000, embargo_bars=100)
        for fold in folds:
            assert fold.train_end + 100 <= fold.test_start

    def test_embargo_zero_no_gap(self):
        df = _make_df(50000)
        folds = create_walk_forward_folds(df, n_folds=3, test_size=4000, embargo_bars=0)
        for fold in folds:
            assert fold.train_end == fold.test_start

    def test_embargo_reduces_training_data(self):
        df = _make_df(50000)
        folds_0 = create_walk_forward_folds(df, n_folds=3, test_size=4000, embargo_bars=0)
        folds_100 = create_walk_forward_folds(df, n_folds=3, test_size=4000, embargo_bars=100)
        for f0, f100 in zip(folds_0, folds_100):
            assert len(f100.train_df) == len(f0.train_df) - 100


class TestInnerFoldEmbargo:
    def test_inner_fold_embargo_gap(self):
        df = _make_df(30000)
        result = nested_cv_split(df, holdout_ratio=0.0, n_inner_folds=3, embargo_bars=50)
        for train_df, val_df in result["inner_folds"]:
            train_end_pos = df.index.get_loc(train_df.index[-1])
            val_start_pos = df.index.get_loc(val_df.index[0])
            assert val_start_pos - train_end_pos > 50

    def test_inner_fold_no_embargo_default(self):
        df = _make_df(30000)
        result = nested_cv_split(df, holdout_ratio=0.0, n_inner_folds=3)
        for train_df, val_df in result["inner_folds"]:
            train_end_pos = df.index.get_loc(train_df.index[-1])
            val_start_pos = df.index.get_loc(val_df.index[0])
            assert val_start_pos - train_end_pos == 1


class TestComputeTargetsWithDurations:
    def test_returns_four_arrays(self):
        from fwbg.simulation.numba_core import compute_targets_with_durations_numba
        n = 200
        rng = np.random.default_rng(42)
        prices = rng.standard_normal(n).cumsum() + 100
        opens = prices.astype(np.float64)
        closes = (prices + rng.standard_normal(n) * 0.5).astype(np.float64)
        highs = np.maximum(opens, closes) + np.abs(rng.standard_normal(n) * 0.3)
        lows = np.minimum(opens, closes) - np.abs(rng.standard_normal(n) * 0.3)

        tgt_l, tgt_s, dur_l, dur_s = compute_targets_with_durations_numba(
            opens, closes, highs, lows,
            tp_distance=2.0, sl_distance=1.0, spread=0.5, slippage=0.25,
            max_bars=50, timeout_bars=20,
        )
        assert tgt_l.shape == (n,)
        assert tgt_s.shape == (n,)
        assert dur_l.shape == (n,)
        assert dur_s.shape == (n,)

    def test_durations_positive(self):
        from fwbg.simulation.numba_core import compute_targets_with_durations_numba
        n = 200
        rng = np.random.default_rng(42)
        prices = rng.standard_normal(n).cumsum() + 100
        opens = prices.astype(np.float64)
        closes = (prices + rng.standard_normal(n) * 0.5).astype(np.float64)
        highs = np.maximum(opens, closes) + np.abs(rng.standard_normal(n) * 0.3)
        lows = np.minimum(opens, closes) - np.abs(rng.standard_normal(n) * 0.3)

        _, _, dur_l, dur_s = compute_targets_with_durations_numba(
            opens, closes, highs, lows,
            tp_distance=2.0, sl_distance=1.0, spread=0.5, slippage=0.25,
            max_bars=50, timeout_bars=20,
        )
        # All durations should be > 0 except last bar
        assert np.all(dur_l[:-1] > 0)
        assert np.all(dur_s[:-1] > 0)
        # Durations bounded by max_bars
        assert np.all(dur_l <= 50)
        assert np.all(dur_s <= 50)

    def test_consistent_with_compute_targets_numba(self):
        """Targets must match between duration and non-duration variants."""
        from fwbg.simulation.numba_core import (
            compute_targets_numba,
            compute_targets_with_durations_numba,
        )
        n = 200
        rng = np.random.default_rng(42)
        prices = rng.standard_normal(n).cumsum() + 100
        opens = prices.astype(np.float64)
        closes = (prices + rng.standard_normal(n) * 0.5).astype(np.float64)
        highs = np.maximum(opens, closes) + np.abs(rng.standard_normal(n) * 0.3)
        lows = np.minimum(opens, closes) - np.abs(rng.standard_normal(n) * 0.3)

        args = (opens, closes, highs, lows, 2.0, 1.0, 0.5, 0.25, 50, 20)
        tgt_l, tgt_s = compute_targets_numba(*args)
        tgt_l2, tgt_s2, _, _ = compute_targets_with_durations_numba(*args)

        np.testing.assert_array_equal(tgt_l, tgt_l2)
        np.testing.assert_array_equal(tgt_s, tgt_s2)
