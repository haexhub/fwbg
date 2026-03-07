"""Tests for the xgboost_rrr model plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg_sdk.models import TrainingContext


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

    def test_trained_classes(self, model, training_data):
        features, targets = training_data
        model.train(
            features, targets, TrainingContext(), rrr_variants=[2.0, 3.0]
        )
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
        model.train(
            features, targets, TrainingContext(), rrr_variants=[2.0]
        )
        ptp = model.get_per_trade_params(features, atr=atr_values)
        assert ptp is None

    def test_feature_importance(self, model, training_data):
        features, targets = training_data
        model.train(
            features, targets, TrainingContext(), rrr_variants=[2.0, 3.0]
        )
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
