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


class TestConcurrentLabels:
    def test_no_overlap(self):
        """Duration=1 → kein Overlap → alle concurrent=1."""
        from fwbg.optimization.purging import compute_concurrent_labels
        n = 10
        durations = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.int64)
        concurrent = compute_concurrent_labels(n, durations)
        assert np.all(concurrent[:9] == 1.0)
        assert concurrent[9] == 0.0

    def test_full_overlap(self):
        """Lange Durations → viel Overlap."""
        from fwbg.optimization.purging import compute_concurrent_labels
        n = 10
        durations = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 0], dtype=np.int64)
        concurrent = compute_concurrent_labels(n, durations)
        # Zeitpunkt 0: nur Label 0 aktiv
        assert concurrent[0] == 1.0
        # Zeitpunkt 4: Labels 0,1,2,3,4 alle aktiv
        assert concurrent[4] == 5.0


class TestSampleWeights:
    def test_sum_to_n(self):
        """Gewichte müssen auf n_samples normalisiert sein."""
        from fwbg.optimization.purging import compute_sample_weights
        n = 100
        dur_l = np.full(n, 10, dtype=np.int64)
        dur_s = np.full(n, 8, dtype=np.int64)
        weights = compute_sample_weights(dur_l, dur_s, n)
        assert abs(weights.sum() - n) < 0.01

    def test_unique_higher_weight(self):
        """Samples mit wenig Overlap bekommen höheres Gewicht."""
        from fwbg.optimization.purging import compute_sample_weights
        n = 20
        dur_l = np.zeros(n, dtype=np.int64)
        dur_l[:10] = 2    # kurze Trades → wenig Overlap
        dur_l[10:] = 10   # lange Trades → viel Overlap
        dur_s = np.zeros(n, dtype=np.int64)
        weights = compute_sample_weights(dur_l, dur_s, n)
        assert weights[:10].mean() > weights[10:].mean()

    def test_zero_durations(self):
        """Bei Duration=0 überall → uniform Gewichte."""
        from fwbg.optimization.purging import compute_sample_weights
        n = 50
        dur_l = np.zeros(n, dtype=np.int64)
        dur_s = np.zeros(n, dtype=np.int64)
        weights = compute_sample_weights(dur_l, dur_s, n)
        # Alle Gewichte sollten gleich 1.0 sein
        np.testing.assert_allclose(weights, 1.0)


class TestComputeTargetsCachedDurations:
    """Integration: compute_targets_cached with return_durations=True."""

    def test_fixed_strategy_returns_4_arrays(self):
        from fwbg.optimization.targets import compute_targets_cached
        from unittest.mock import Mock

        ctx = Mock()
        ctx.spread = 0.0003
        ctx.max_trade_bars = None
        ctx.exit_strategy = "fixed"
        ctx.exit_params = {}

        n = 500
        rng = np.random.default_rng(42)
        prices = rng.standard_normal(n).cumsum() + 1.1
        df = pd.DataFrame({
            "O": prices,
            "C": prices + rng.standard_normal(n) * 0.001,
            "H": prices + np.abs(rng.standard_normal(n) * 0.002),
            "L": prices - np.abs(rng.standard_normal(n) * 0.002),
        }, index=pd.date_range("2020-01-01", periods=n, freq="h"))

        result = compute_targets_cached(
            df, tp=20, sl=10, ctx=ctx, timeout_bars=48,
            exit_strategy_mode="fixed", return_durations=True,
        )
        assert len(result) == 4
        tgt_l, tgt_s, dur_l, dur_s = result
        assert tgt_l.shape == (n,)
        assert dur_l.shape == (n,)
        assert np.all(dur_l >= 0)

    def test_fixed_without_durations_returns_2_arrays(self):
        from fwbg.optimization.targets import compute_targets_cached
        from unittest.mock import Mock

        ctx = Mock()
        ctx.spread = 0.0003
        ctx.max_trade_bars = None

        n = 200
        rng = np.random.default_rng(42)
        prices = rng.standard_normal(n).cumsum() + 1.1
        df = pd.DataFrame({
            "O": prices,
            "C": prices + rng.standard_normal(n) * 0.001,
            "H": prices + np.abs(rng.standard_normal(n) * 0.002),
            "L": prices - np.abs(rng.standard_normal(n) * 0.002),
        }, index=pd.date_range("2020-01-01", periods=n, freq="h"))

        result = compute_targets_cached(
            df, tp=20, sl=10, ctx=ctx, timeout_bars=48,
            exit_strategy_mode="fixed", return_durations=False,
        )
        assert len(result) == 2


class TestTrainModelWithWeights:
    """Integration: train_model accepts sample_weight parameter."""

    def test_train_model_with_weights(self):
        from fwbg.optimization.nested_cv import train_model
        from unittest.mock import Mock

        ctx = Mock()
        ctx.model_type = "xgboost"
        ctx.model_hyperparameters = {"n_estimators": 10, "max_depth": 3}
        ctx.min_trades = 10
        ctx.probability_calibration = False
        ctx.calibration_method = "isotonic"

        n = 200
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
        })
        targets = (rng.random(n) > 0.5).astype(float)
        weights = rng.random(n) + 0.5  # positive weights

        model = train_model(
            df, targets, ["f1", "f2"], min_trades=10, ctx=ctx,
            sample_weight=weights,
        )
        assert model is not None

    def test_train_model_without_weights(self):
        from fwbg.optimization.nested_cv import train_model
        from unittest.mock import Mock

        ctx = Mock()
        ctx.model_type = "xgboost"
        ctx.model_hyperparameters = {"n_estimators": 10, "max_depth": 3}
        ctx.min_trades = 10
        ctx.probability_calibration = False
        ctx.calibration_method = "isotonic"

        n = 200
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
        })
        targets = (rng.random(n) > 0.5).astype(float)

        model = train_model(df, targets, ["f1", "f2"], min_trades=10, ctx=ctx)
        assert model is not None
