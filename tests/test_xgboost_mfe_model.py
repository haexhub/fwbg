"""Tests for the xgboost_mfe model plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg_sdk.models import TrainingContext


class TestXGBoostMFEModel:
    @pytest.fixture
    def model(self):
        from fwbg.core.registry import get_model

        model_class = get_model("xgboost_mfe")
        return model_class()

    @pytest.fixture
    def training_data(self):
        np.random.seed(42)
        n = 200
        features = pd.DataFrame({
            "feat_1": np.random.randn(n),
            "feat_2": np.random.randn(n),
            "feat_3": np.random.randn(n),
        })
        # Continuous MFE targets (in ATR multiples)
        targets = np.abs(np.random.randn(n)) * 2.0
        return features, targets

    @pytest.fixture
    def atr_values(self):
        return np.full(200, 1.5)

    def test_registration(self):
        from fwbg.core.registry import get_model

        model_class = get_model("xgboost_mfe")
        assert model_class.name == "xgboost_mfe"

    def test_train_and_predict(self, model, training_data):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), sl_variants=[1.5, 2.0, 3.0])
        assert model.is_trained
        probs = model.predict_probability(features)
        assert probs.shape == (len(features), 2)
        assert np.all(probs[:, 1] >= 0.0)

    def test_trained_classes(self, model, training_data):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), sl_variants=[2.0])
        assert 0 in model.trained_classes
        assert 1 in model.trained_classes

    def test_predicted_mfe_reasonable(self, model, training_data):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), sl_variants=[2.0])
        probs = model.predict_probability(features)
        predicted_mfe = probs[:, 1]
        assert np.all(predicted_mfe >= 0.0)
        assert np.all(predicted_mfe < 50.0)

    def test_get_per_trade_params(self, model, training_data, atr_values):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), sl_variants=[1.5, 2.0, 3.0])
        model.predict_probability(features)
        ptp = model.get_per_trade_params(features, atr=atr_values)
        assert ptp is not None
        assert ptp.shape == (len(features), 2)
        assert np.all(ptp[:, 0] >= 0.0)
        assert np.all(ptp[:, 1] > 0.0)

    def test_selects_best_mfe_sl_ratio(self, model, training_data):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), sl_variants=[1.0, 2.0, 5.0])
        model.predict_probability(features)
        assert model.selected_sl_atr is not None
        assert model.selected_sl_atr.shape == (len(features),)
        assert all(s in [1.0, 2.0, 5.0] for s in model.selected_sl_atr)

    def test_feature_importance(self, model, training_data):
        features, targets = training_data
        model.train(features, targets, TrainingContext(), sl_variants=[2.0])
        importance = model.get_feature_importance()
        assert importance is not None
        assert "sl_atr" in importance

    def test_reduced_hyperparameters(self):
        from fwbg.core.registry import get_model

        model_class = get_model("xgboost_mfe")
        hp = {"n_estimators": 200, "sl_variants": [2.0, 3.0]}
        reduced = model_class.get_reduced_hyperparameters(hp)
        assert reduced["n_estimators"] == 100
        assert reduced["sl_variants"] == [2.0, 3.0]
