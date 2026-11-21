"""Tests for the xgboost_rrr model plugin."""

import numpy as np
import pandas as pd
import pytest

from fwbg_sdk.models import TrainingContext
from fwbg.plugins import import_plugin_module

_rrr_mod = import_plugin_module("fwbg-core", "models", "xgboost_rrr")
_compute_binary_targets = _rrr_mod._compute_binary_targets


def _make_ohlc_df(n=200, seed=42):
    """Create a synthetic OHLC DataFrame with ATR."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3
    opn = close + np.random.randn(n) * 0.1
    atr = np.full(n, 1.0)
    return pd.DataFrame(
        {
            "O": opn,
            "H": high,
            "L": low,
            "C": close,
            "_atr": atr,
            "feat_1": np.random.randn(n),
            "feat_2": np.random.randn(n),
            "feat_3": np.random.randn(n),
        }
    )


class TestComputeBinaryTargets:
    def test_basic_long_targets(self):
        df = _make_ohlc_df(100)
        targets = _compute_binary_targets(
            df, rrr=2.0, base_sl_atr=1.0, direction="long"
        )
        assert targets.shape == (100,)
        assert set(np.unique(targets[~np.isnan(targets)])).issubset({0, 1})
        assert np.isnan(targets[-1])  # the last bar has no entry: unknown

    def test_basic_short_targets(self):
        df = _make_ohlc_df(100)
        targets = _compute_binary_targets(
            df, rrr=2.0, base_sl_atr=1.0, direction="short"
        )
        assert targets.shape == (100,)
        assert set(np.unique(targets[~np.isnan(targets)])).issubset({0, 1})

    def test_higher_rrr_fewer_wins(self):
        """Higher RRR (larger TP) should produce fewer or equal wins."""
        df = _make_ohlc_df(500, seed=123)
        wins_low = np.nansum(
            _compute_binary_targets(df, rrr=1.0, base_sl_atr=1.0, direction="long")
        )
        wins_high = np.nansum(
            _compute_binary_targets(df, rrr=5.0, base_sl_atr=1.0, direction="long")
        )
        assert wins_high <= wins_low

    def test_different_rrr_produce_different_targets(self):
        """Core bug fix: different RRR variants must produce different target arrays."""
        df = _make_ohlc_df(300, seed=99)
        t1 = _compute_binary_targets(df, rrr=1.5, base_sl_atr=2.0, direction="long")
        t2 = _compute_binary_targets(df, rrr=4.0, base_sl_atr=2.0, direction="long")
        assert not np.array_equal(t1, t2, equal_nan=True), (
            "Different RRR must produce different targets"
        )


class TestXGBoostRRRModel:
    @pytest.fixture
    def model(self):
        from fwbg.core.registry import get_model

        model_class = get_model("xgboost_rrr")
        return model_class()

    @pytest.fixture
    def training_data(self):
        np.random.seed(42)
        n = 200
        features = pd.DataFrame(
            {
                "feat_1": np.random.randn(n),
                "feat_2": np.random.randn(n),
                "feat_3": np.random.randn(n),
            }
        )
        targets = (np.random.rand(n) > 0.4).astype(np.float64)
        return features, targets

    @pytest.fixture
    def atr_values(self):
        return np.full(200, 1.5)

    def test_registration(self):
        from fwbg.core.registry import get_model

        model_class = get_model("xgboost_rrr")
        assert model_class.name == "xgboost_rrr"

    def test_train_and_predict(self, model, training_data):
        features, targets = training_data
        model.train(
            features,
            targets,
            TrainingContext(direction="long"),
            rrr_variants=[1.5, 2.0, 3.0],
            base_sl_atr=2.0,
        )
        assert model.is_trained
        probs = model.predict_probability(features)
        assert probs.shape == (len(features), 2)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_train_with_ohlc_data(self, model):
        """Train with full OHLC data — per-variant targets should be computed."""
        train_df = _make_ohlc_df(300)
        features = train_df[["feat_1", "feat_2", "feat_3"]]
        targets = (np.random.rand(300) > 0.5).astype(np.float64)

        ctx = TrainingContext(
            direction="long",
            fold_information={"train_df": train_df},
        )
        model.train(
            features,
            targets,
            ctx,
            rrr_variants=[1.5, 3.0],
            base_sl_atr=1.5,
        )
        assert model.is_trained
        probs = model.predict_probability(features)
        assert probs.shape == (300, 2)

    def test_trained_classes(self, model, training_data):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), rrr_variants=[2.0, 3.0])
        assert 0 in model.trained_classes
        assert 1 in model.trained_classes

    def test_selected_rrr(self, model, training_data):
        features, targets = training_data
        model.train(
            features,
            targets,
            TrainingContext(),
            rrr_variants=[1.5, 2.0, 3.0],
        )
        model.predict_probability(features)
        assert model.selected_rrr is not None
        assert model.selected_rrr.shape == (len(features),)
        assert all(r in [1.5, 2.0, 3.0] for r in model.selected_rrr)

    def test_get_per_trade_params(self, model, training_data, atr_values):
        features, targets = training_data
        model.train(
            features,
            targets,
            TrainingContext(),
            rrr_variants=[1.5, 2.0, 3.0],
            base_sl_atr=2.0,
        )
        model.predict_probability(features)
        ptp = model.get_per_trade_params(features, atr=atr_values)
        assert ptp is not None
        assert ptp.shape == (len(features), 2)
        # SL distance = base_sl_atr * atr = 2.0 * 1.5 = 3.0
        assert np.allclose(ptp[:, 1], 3.0)

    def test_get_per_trade_params_none_without_predict(
        self, model, training_data, atr_values
    ):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), rrr_variants=[2.0])
        ptp = model.get_per_trade_params(features, atr=atr_values)
        assert ptp is None

    def test_feature_importance(self, model, training_data):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), rrr_variants=[2.0, 3.0])
        importance = model.get_feature_importance()
        assert importance is not None
        assert "rrr" in importance

    def test_reduced_hyperparameters(self):
        from fwbg.core.registry import get_model

        model_class = get_model("xgboost_rrr")
        hp = {"n_estimators": 200, "rrr_variants": [2.0, 3.0]}
        reduced = model_class.get_reduced_hyperparameters(hp)
        assert reduced["n_estimators"] == 100
        assert reduced["rrr_variants"] == [2.0, 3.0]


class TestSlFirstLabels:
    """A bar that touches both TP and SL is a stop, as in the simulation kernels."""

    @staticmethod
    def _df(rows):
        a = np.array(rows, dtype=float)
        return pd.DataFrame(
            {"O": a[:, 0], "H": a[:, 1], "L": a[:, 2], "C": a[:, 3], "_atr": 1.0}
        )

    def test_long_bar_touching_both_levels_is_a_loss(self):
        # entry 100, SL 99, TP 102; bar 1 spans 98.5..102.5
        df = self._df(
            [(100, 100, 100, 100), (100, 102.5, 98.5, 100), (100, 100, 100, 100)]
        )
        t = _compute_binary_targets(df, rrr=2.0, base_sl_atr=1.0, direction="long")
        assert t[0] == 0

    def test_short_bar_touching_both_levels_is_a_loss(self):
        df = self._df(
            [(100, 100, 100, 100), (100, 101.5, 97.5, 100), (100, 100, 100, 100)]
        )
        t = _compute_binary_targets(df, rrr=2.0, base_sl_atr=1.0, direction="short")
        assert t[0] == 0

    def test_clean_tp_bar_still_wins(self):
        df = self._df(
            [(100, 100, 100, 100), (100, 102.5, 99.5, 102), (100, 100, 100, 100)]
        )
        assert _compute_binary_targets(df, 2.0, 1.0, "long")[0] == 1
        down = self._df(
            [(100, 100, 100, 100), (100, 100.5, 97.5, 98), (100, 100, 100, 100)]
        )
        assert _compute_binary_targets(down, 2.0, 1.0, "short")[0] == 1


class TestVariantSelectionByExpectedValue:
    class _StubModel:
        classes_ = np.array([0, 1])

        def __init__(self, p_by_rrr):
            self.p_by_rrr = p_by_rrr

        def predict_proba(self, df):
            p = np.array([self.p_by_rrr[r] for r in df["rrr"].values])
            return np.column_stack([1 - p, p])

    def _model(self, variants, p_by_rrr):
        model = _rrr_mod.XGBoostRRRModel()
        model._variants = variants
        model._model = self._StubModel(p_by_rrr)
        return model

    def test_higher_payoff_beats_higher_win_probability(self):
        # EV: rrr 1.0 at p=.60 -> .20 ; rrr 3.0 at p=.45 -> 0.80
        model = self._model([1.0, 3.0], {1.0: 0.60, 3.0: 0.45})
        probs = model._predict_probability_impl(pd.DataFrame({"f": [0.0, 1.0]}))
        assert list(model.selected_rrr) == [3.0, 3.0]
        assert probs[:, 1] == pytest.approx([0.45, 0.45])

    def test_low_payoff_wins_when_it_has_the_higher_expected_value(self):
        # EV: rrr 1.0 at p=.95 -> .90 ; rrr 3.0 at p=.45 -> .80
        model = self._model([1.0, 3.0], {1.0: 0.95, 3.0: 0.45})
        model._predict_probability_impl(pd.DataFrame({"f": [0.0]}))
        assert list(model.selected_rrr) == [1.0]
