"""
Tests für Meta-Labeling (AFML Ch. 3).

Meta-Labeling: Primary model predicts direction, meta-model filters
trades by predicting whether the primary signal will be profitable.
The meta-model uses the primary model's probability as an additional feature.
"""
import numpy as np
import pandas as pd
import pytest
from fwbg_sdk.models import BaseModel, TrainingContext

from fwbg.core.context import SimulationContext
from fwbg.core import get_model


def _make_ctx(meta_labeling=False, probability_calibration=False):
    """Create a real SimulationContext for meta-labeling tests."""
    return SimulationContext(
        symbol="TESTUSD",
        asset_class="FOREX",
        spread=0.0002,
        point=0.0001,
        min_trades=10,
        grid_ct=[0.50, 0.55, 0.60],
        grid_tp=[10, 20],
        grid_sl=[20, 30],
        long_enabled=True,
        short_enabled=True,
        model_hyperparameters={"n_estimators": 20, "max_depth": 3, "random_state": 42},
        meta_labeling=meta_labeling,
        probability_calibration=probability_calibration,
    )


def _make_data(n=500, seed=42):
    """Create OHLC data with predictive features and known targets."""
    rng = np.random.default_rng(seed)
    price = 1.1000 + np.cumsum(rng.normal(0, 0.0005, n))
    atr = np.abs(rng.normal(0.001, 0.0003, n))

    targets = (rng.random(n) > 0.6).astype(float)

    df = pd.DataFrame({
        "O": price,
        "H": price + np.abs(rng.normal(0.0005, 0.0002, n)),
        "L": price - np.abs(rng.normal(0.0005, 0.0002, n)),
        "C": price + rng.normal(0, 0.0003, n),
        "_atr": atr,
        "_regime": np.full(n, 7, dtype=np.int8),
        # Features correlated with targets
        "feat1": targets + rng.normal(0, 0.3, n),
        "feat2": targets * 0.8 + rng.normal(0, 0.4, n),
        "feat3": rng.standard_normal(n),  # noise
    }, index=pd.date_range("2020-01-01", periods=n, freq="h"))

    return df, targets


class TestMetaLabelingConfig:
    """Config and context wiring tests."""

    def test_config_parsing(self):
        from fwbg.core.config import ValidationConfig

        cfg = ValidationConfig.from_dict({"meta_labeling": True})
        assert cfg.meta_labeling is True

    def test_config_defaults(self):
        from fwbg.core.config import ValidationConfig

        cfg = ValidationConfig.from_dict({})
        assert cfg.meta_labeling is False

    def test_context_wiring(self):
        ctx = _make_ctx(meta_labeling=True)
        assert ctx.meta_labeling is True


class TestOOFPredictions:
    """Tests for out-of-fold prediction generation."""

    def test_generates_oof_probs(self):
        """OOF predictions should be generated for all training samples."""
        from fwbg.optimization.nested_cv import _generate_oof_predictions

        ctx = _make_ctx()
        df, targets = _make_data(300)
        features = ["feat1", "feat2", "feat3"]

        oof_probs = _generate_oof_predictions(df, targets, features, ctx)

        assert oof_probs.shape == (300,)
        # OOF probs should be probabilities (0 to 1)
        assert np.all(oof_probs >= 0)
        assert np.all(oof_probs <= 1)
        # Not all zeros (some predictions were made)
        assert np.any(oof_probs > 0)

    def test_oof_probs_no_data_leakage(self):
        """OOF predictions on noise-only features should NOT be overly accurate."""
        from fwbg.optimization.nested_cv import _generate_oof_predictions

        ctx = _make_ctx()
        rng = np.random.default_rng(99)
        n = 300
        targets = (rng.random(n) > 0.5).astype(float)

        # Pure noise features — no signal, so OOF accuracy should be near chance
        df = pd.DataFrame({
            "noise1": rng.standard_normal(n),
            "noise2": rng.standard_normal(n),
            "noise3": rng.standard_normal(n),
        }, index=pd.date_range("2020-01-01", periods=n, freq="h"))

        oof_probs = _generate_oof_predictions(df, targets, ["noise1", "noise2", "noise3"], ctx)

        predicted = (oof_probs > 0.5).astype(float)
        accuracy = np.mean(predicted == targets)
        # Noise-only features: accuracy should be near 50% (chance level)
        assert accuracy < 0.65, f"Suspiciously high OOF accuracy on noise: {accuracy}"


class TestMetaModelTraining:
    """Tests for meta-model training."""

    def test_trains_meta_model(self):
        """Meta-model should be a fitted BaseModel."""
        from fwbg.optimization.nested_cv import _train_meta_model

        ctx = _make_ctx()
        df, targets = _make_data(300)
        features = ["feat1", "feat2", "feat3"]
        oof_probs = np.random.default_rng(42).random(300)

        meta_model = _train_meta_model(df, targets, features, oof_probs, ctx)

        assert meta_model is not None
        assert hasattr(meta_model, "predict_probability")
        # Meta-model should take features + 1 (the OOF prob column)
        X_test = pd.DataFrame(
            np.column_stack([df[features].values[:5], oof_probs[:5]]),
            columns=features + ["oof_prob"],
        )
        probs = meta_model.predict_probability(X_test)
        assert probs.shape == (5, 2)

    def test_returns_none_when_insufficient_targets(self):
        """Meta-model should return None when too few positive targets."""
        from fwbg.optimization.nested_cv import _train_meta_model

        ctx = _make_ctx()
        df, _ = _make_data(300)
        targets = np.zeros(300)  # No positive targets
        features = ["feat1", "feat2", "feat3"]
        oof_probs = np.zeros(300)

        meta_model = _train_meta_model(df, targets, features, oof_probs, ctx)
        assert meta_model is None


class TestMetaFilter:
    """Tests for the meta-labeling probability filter."""

    def test_filters_low_confidence_bars(self):
        """Bars where meta-model says 'skip' should have zeroed primary probs."""
        from fwbg.optimization.targets import _apply_meta_filter

        rng = np.random.default_rng(42)
        n = 100

        # Primary probs: all bars have high primary confidence
        probs = np.column_stack([1 - np.full(n, 0.7), np.full(n, 0.7)])
        win_idx = 1

        # Train a meta-model via plugin system
        X_train = pd.DataFrame(
            np.column_stack([rng.standard_normal((200, 3)), rng.random(200)]),
            columns=["f1", "f2", "f3", "oof_prob"],
        )
        y_train = (rng.random(200) > 0.5).astype(float)

        model_class = get_model("xgboost")
        meta_model = model_class()
        meta_model.train(
            X_train, y_train, TrainingContext(),
            n_estimators=10, max_depth=2, random_state=42,
        )

        df = pd.DataFrame({
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "f3": rng.standard_normal(n),
        })

        filtered = _apply_meta_filter(df, probs, win_idx, ["f1", "f2", "f3"], meta_model)

        # Some bars should be zeroed out (meta-model rejects them)
        assert filtered.shape == probs.shape
        # At least some bars should be filtered (zeroed)
        n_filtered = np.sum(filtered[:, win_idx] == 0.0)
        assert n_filtered > 0, "Meta-filter should reject some bars"
        # Not all bars filtered
        n_kept = np.sum(filtered[:, win_idx] > 0.0)
        assert n_kept > 0, "Meta-filter should keep some bars"

    def test_no_filter_without_meta_model(self):
        """Without meta-model, probs should pass through unchanged."""
        from fwbg.optimization.targets import _apply_meta_filter

        probs = np.array([[0.3, 0.7], [0.4, 0.6]])
        result = _apply_meta_filter(None, probs, 1, [], None)
        np.testing.assert_array_equal(result, probs)


class TestMetaLabelingEndToEnd:
    """End-to-end tests for meta-labeling in the evaluation pipeline."""

    def test_evaluate_with_meta_labeling(self):
        """Meta-labeling should filter trades in evaluate_on_validation."""
        from fwbg.optimization.nested_cv import (
            train_model, _generate_oof_predictions, _train_meta_model,
        )
        from fwbg.optimization.targets import evaluate_on_validation

        ctx = _make_ctx(meta_labeling=True)
        df, targets_long = _make_data(500)
        rng = np.random.default_rng(99)
        targets_short = (rng.random(500) > 0.6).astype(float)

        train_df = df.iloc[:350]
        val_df = df.iloc[350:]
        features = ["feat1", "feat2", "feat3"]

        # Train primary models
        mod_long = train_model(train_df, targets_long[:350], features, 10, ctx)
        mod_short = train_model(train_df, targets_short[:350], features, 10, ctx)

        # Train meta-models
        oof_long = _generate_oof_predictions(train_df, targets_long[:350], features, ctx)
        meta_long = _train_meta_model(train_df, targets_long[:350], features, oof_long, ctx)

        oof_short = _generate_oof_predictions(train_df, targets_short[:350], features, ctx)
        meta_short = _train_meta_model(train_df, targets_short[:350], features, oof_short, ctx)

        # Evaluate WITH meta-labeling
        ct_meta, pnl_meta, trades_meta = evaluate_on_validation(
            val_df, mod_long, mod_short, features, features, 20, 30, ctx,
            meta_mod_long=meta_long, meta_mod_short=meta_short,
        )

        # Evaluate WITHOUT meta-labeling (same primary models)
        ctx_no_meta = _make_ctx(meta_labeling=False)
        ct_no_meta, pnl_no_meta, trades_no_meta = evaluate_on_validation(
            val_df, mod_long, mod_short, features, features, 20, 30, ctx_no_meta,
        )

        # Meta-labeling should produce fewer or equal trades (it's a filter)
        n_trades_meta = sum(len(t) for t in trades_meta.values())
        n_trades_no_meta = sum(len(t) for t in trades_no_meta.values())
        assert n_trades_meta <= n_trades_no_meta

    def test_evaluate_single_fold_with_meta_labeling(self):
        """_evaluate_single_fold should train and use meta-models when enabled."""
        from fwbg.optimization.nested_cv import _evaluate_single_fold

        ctx = _make_ctx(meta_labeling=True)
        df, targets_long = _make_data(500)

        train_df = df.iloc[:350]
        val_df = df.iloc[350:]
        features = ["feat1", "feat2", "feat3"]

        result = _evaluate_single_fold(
            fold_idx=0,
            train_df=train_df,
            val_df=val_df,
            group_features=features,
            tp=20, sl=30, ctx=ctx,
        )

        # Should succeed (meta-labeling is a filter, doesn't break evaluation)
        # Result can be success or failure depending on trade count
        assert isinstance(result, dict)
        assert "success" in result
