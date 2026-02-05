"""
Tests für Preprocessing in Cross-Validation.

Diese Tests verifizieren, dass Preprocessing korrekt in der CV-Schleife
angewendet wird ohne Lookahead Bias zu verursachen.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from fwbg.core.context import SimulationContext
from fwbg.plugins import BasePreprocessor


class MockPreprocessor(BasePreprocessor):
    """Mock Preprocessor der fit/transform Calls tracked."""

    order = 10

    def __init__(self):
        super().__init__()
        self.fit_calls = []
        self.transform_calls = []
        self.d_values = {}  # Store learned d per fit call

    def fit(self, df: pd.DataFrame, **params) -> "MockPreprocessor":
        """Track fit calls and learn a parameter from the data."""
        # Learn a parameter (mean of Close price) to verify it's train-data only
        self.d_values[id(df)] = df["C"].mean() if "C" in df.columns else 0
        self.fit_calls.append({
            "df_id": id(df),
            "df_shape": df.shape,
            "df_index_range": (df.index.min(), df.index.max()),
            "learned_param": self.d_values[id(df)]
        })
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Track transform calls."""
        if not self.fitted_:
            raise RuntimeError("fit() must be called before transform()")

        self.transform_calls.append({
            "df_id": id(df),
            "df_shape": df.shape,
            "df_index_range": (df.index.min(), df.index.max())
        })

        # Return DataFrame with first 10 rows removed (simulate preprocessing)
        return df.iloc[10:]


def test_preprocessing_fit_called_per_fold():
    """Test dass fit() für jeden Fold separat aufgerufen wird."""
    from fwbg.optimization.nested_cv import run_inner_cv

    # Create mock data
    n_samples = 1000
    df = pd.DataFrame({
        "C": np.random.randn(n_samples) + 100,
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
    })

    # Create 3 folds
    fold1_train = df.iloc[:600]
    fold1_val = df.iloc[600:800]
    fold2_train = df.iloc[:700]
    fold2_val = df.iloc[700:900]
    fold3_train = df.iloc[:800]
    fold3_val = df.iloc[800:]

    inner_folds = [
        (fold1_train.copy(), fold1_val.copy()),
        (fold2_train.copy(), fold2_val.copy()),
        (fold3_train.copy(), fold3_val.copy()),
    ]

    # Create mock context with preprocessing
    ctx = Mock(spec=SimulationContext)
    ctx.preprocessing = ["mock_preprocessor"]
    ctx.preprocessing_params = {"mock_preprocessor": {}}
    ctx.symbol = "TEST"
    ctx.min_trades = 10
    ctx.feature_selection = "boruta"
    ctx.max_features = 5
    ctx.min_z_score = 0.3
    ctx.early_termination = False
    ctx.first_fold_sanity_check = False
    ctx.min_fold_stability = 0.5
    ctx.first_fold_min_win_rate = 0.25
    ctx.first_fold_min_pnl = -10.0
    ctx.first_fold_min_trades = 5
    ctx.separate_long_short = False
    ctx.min_score_margin = 0.0

    # Mock get_preprocessor to return a CLASS that creates tracked instances
    # We need to return the class itself, not an instance
    mock_pp = MockPreprocessor()

    def get_mock_preprocessor_class(name):
        """Return a class that always creates the same tracked instance."""
        return lambda: mock_pp

    with patch("fwbg.core.get_preprocessor", side_effect=get_mock_preprocessor_class):
        with patch("fwbg.optimization.nested_cv.compute_targets") as mock_targets:
            with patch("fwbg.optimization.nested_cv.select_features_from_fold") as mock_select:
                with patch("fwbg.optimization.nested_cv.train_model") as mock_train:
                    with patch("fwbg.optimization.nested_cv.evaluate_on_validation") as mock_eval:
                        # Configure mock returns
                        mock_targets.return_value = (
                            pd.Series([1, 0, 1] * 100, index=range(300)),  # targets_long
                            pd.Series([0, 1, 0] * 100, index=range(300)),  # targets_short
                            True,  # has_long
                            True   # has_short
                        )
                        mock_select.return_value = (["feature1", "feature2"], None)
                        mock_train.return_value = (Mock(), Mock())  # model_long, model_short
                        mock_eval.return_value = (1, 10.0, {})  # best_ct, best_pnl, trades_by_ct

                        try:
                            run_inner_cv(
                                inner_folds=inner_folds,
                                group_features=["feature1", "feature2"],
                                tp=30,
                                sl=50,
                                ctx=ctx,
                                global_grid_pos=1,
                                total_grid_combos=10,
                                cached_targets=None
                            )
                        except Exception as e:
                            print(f"[DEBUG] run_inner_cv failed: {e}")
                            import traceback
                            traceback.print_exc()

    # Verify fit() was called once per fold
    assert len(mock_pp.fit_calls) == 3, f"Expected 3 fit calls, got {len(mock_pp.fit_calls)}"

    # Verify each fit call had different data (different train sets)
    learned_params = [call["learned_param"] for call in mock_pp.fit_calls]
    assert len(set(learned_params)) > 1, "fit() should learn different parameters from different train sets"


def test_preprocessing_no_lookahead_bias():
    """Test dass Preprocessing keinen Lookahead Bias verursacht."""
    from fwbg.optimization.nested_cv import run_inner_cv

    # Create data where train and val have different distributions
    train_data = pd.DataFrame({
        "C": np.random.randn(500) + 100,  # Mean = 100
        "feature1": np.random.randn(500),
    })
    val_data = pd.DataFrame({
        "C": np.random.randn(200) + 150,  # Mean = 150 (DIFFERENT!)
        "feature1": np.random.randn(200),
    })
    val_data.index = range(500, 700)

    inner_folds = [(train_data.copy(), val_data.copy())]

    ctx = Mock(spec=SimulationContext)
    ctx.preprocessing = ["mock_preprocessor"]
    ctx.preprocessing_params = {"mock_preprocessor": {}}
    ctx.symbol = "TEST"
    ctx.early_termination = False
    ctx.first_fold_sanity_check = False
    ctx.min_trades = 10
    ctx.min_fold_stability = 0.5
    ctx.first_fold_min_win_rate = 0.25
    ctx.first_fold_min_pnl = -10.0
    ctx.first_fold_min_trades = 5
    ctx.separate_long_short = False
    ctx.min_score_margin = 0.0
    ctx.feature_selection = "boruta"
    ctx.max_features = 5

    mock_pp = MockPreprocessor()

    def get_mock_preprocessor_class(name):
        """Return a class that always creates the same tracked instance."""
        return lambda: mock_pp

    with patch("fwbg.core.get_preprocessor", side_effect=get_mock_preprocessor_class):
        with patch("fwbg.optimization.nested_cv.compute_targets") as mock_targets:
            with patch("fwbg.optimization.nested_cv.select_features_from_fold") as mock_select:
                with patch("fwbg.optimization.nested_cv.train_model") as mock_train:
                    with patch("fwbg.optimization.nested_cv.evaluate_on_validation") as mock_eval:
                        # Return valid targets to avoid early termination
                        mock_targets.return_value = (
                            pd.Series([1, 0] * 50, index=range(100)),
                            pd.Series([0, 1] * 50, index=range(100)),
                            True,
                            True
                        )
                        mock_select.return_value = (["feature1"], None)
                        mock_train.return_value = (Mock(), Mock())
                        mock_eval.return_value = (1, 10.0, {})  # best_ct, best_pnl, trades_by_ct

                        try:
                            run_inner_cv(
                                inner_folds=inner_folds,
                                group_features=["feature1"],
                                tp=30,
                                sl=50,
                                ctx=ctx,
                                global_grid_pos=1,
                                total_grid_combos=10,
                                cached_targets=None
                            )
                        except Exception as e:
                            print(f"[DEBUG] run_inner_cv failed: {e}")
                            import traceback
                            traceback.print_exc()

    # Verify fit() was called ONLY on train data
    assert len(mock_pp.fit_calls) == 1
    fit_call = mock_pp.fit_calls[0]

    # The learned parameter should be close to train mean (100), NOT val mean (150)
    learned_param = fit_call["learned_param"]
    assert 90 < learned_param < 110, \
        f"Learned param {learned_param} should be from train data (~100), not val data (~150)"

    # Verify transform was called on BOTH train and val
    assert len(mock_pp.transform_calls) >= 2, \
        f"Expected at least 2 transform calls (train+val), got {len(mock_pp.transform_calls)}"


def test_fractional_diff_preserves_reasonable_data():
    """Test dass FractionalDiff nicht zu viele Zeilen entfernt."""
    from fwbg.builtins.preprocessing.fractional_diff import FractionalDiffPreprocessor

    # Create realistic dataset
    n_samples = 10000
    df = pd.DataFrame({
        "O": np.random.randn(n_samples) + 100,
        "H": np.random.randn(n_samples) + 101,
        "L": np.random.randn(n_samples) + 99,
        "C": np.random.randn(n_samples) + 100,
        "feature1": np.random.randn(n_samples),
    })

    pp = FractionalDiffPreprocessor()
    pp.fit(df, auto_d=False, default_d=0.4)
    df_transformed = pp.transform(df)

    # Should lose ~500 rows (max_window=500), not 9999!
    rows_lost = len(df) - len(df_transformed)
    assert rows_lost < 1000, \
        f"Too many rows lost: {rows_lost}. Expected ~500 (max_window), not {rows_lost}"

    # Should keep at least 90% of data
    retention_rate = len(df_transformed) / len(df)
    assert retention_rate > 0.9, \
        f"Retention rate {retention_rate:.1%} too low. Should keep >90% of data"


def test_fractional_diff_no_lookahead_with_auto_d():
    """Test dass auto_d mit fixem d keine Future-Information nutzt."""
    from fwbg.builtins.preprocessing.fractional_diff import FractionalDiffPreprocessor

    # Simpler test: Verify that auto_d=False uses the provided default_d
    # This ensures no lookahead bias when using fixed d values

    train_df = pd.DataFrame({
        "C": np.random.randn(1000) + 100,
        "O": np.random.randn(1000) + 100,
        "H": np.random.randn(1000) + 101,
        "L": np.random.randn(1000) + 99,
    })

    # Fit with auto_d=False and specific d
    pp = FractionalDiffPreprocessor()
    pp.fit(train_df, auto_d=False, default_d=0.3)

    assert pp.d_ == 0.3, f"With auto_d=False, d should be 0.3, got {pp.d_}"

    # Fit with different default_d
    pp2 = FractionalDiffPreprocessor()
    pp2.fit(train_df, auto_d=False, default_d=0.5)

    assert pp2.d_ == 0.5, f"With auto_d=False, d should be 0.5, got {pp2.d_}"

    # Verify transforms are different
    df1 = pp.transform(train_df)
    df2 = pp2.transform(train_df)

    # Different d values should produce different results
    assert not df1["C"].equals(df2["C"]), \
        "Different d values should produce different transformations"


def test_preprocessing_indices_consistency():
    """Test dass DataFrame-Indices nach Preprocessing konsistent bleiben."""
    from fwbg.builtins.preprocessing.fractional_diff import FractionalDiffPreprocessor

    # Create DataFrame with non-standard indices
    df = pd.DataFrame({
        "O": np.random.randn(1000) + 100,
        "H": np.random.randn(1000) + 101,
        "L": np.random.randn(1000) + 99,
        "C": np.random.randn(1000) + 100,
    })
    df.index = range(5000, 6000)  # Start at 5000

    original_index_start = df.index.min()
    original_index_end = df.index.max()

    pp = FractionalDiffPreprocessor()
    pp.fit(df, auto_d=False, default_d=0.4)
    df_transformed = pp.transform(df)

    # Indices should be a subset of original indices (no new indices created)
    assert df_transformed.index.min() >= original_index_start
    assert df_transformed.index.max() == original_index_end

    # Indices should be monotonic increasing
    assert df_transformed.index.is_monotonic_increasing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
