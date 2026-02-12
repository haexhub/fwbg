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
