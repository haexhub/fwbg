"""Test that BaseModel exposes get_per_trade_params with default None."""
import numpy as np
import pandas as pd

from fwbg_sdk.models import BaseModel, TrainingContext


class DummyModel(BaseModel):
    name = "dummy"
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self._classes = np.array([0, 1])

    def train(self, features, targets, training_context, **hp):
        self._fitted = True

    def _predict_probability_impl(self, features):
        n = len(features)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])

    @property
    def _trained_classes_impl(self):
        return self._classes

    def _as_sklearn_estimator_impl(self):
        return None


class TestGetPerTradeParams:
    def test_default_returns_none(self):
        model = DummyModel()
        model.train(pd.DataFrame({"a": [1, 2]}), np.array([0, 1]), TrainingContext())
        result = model.get_per_trade_params(pd.DataFrame({"a": [1, 2]}))
        assert result is None

    def test_method_exists_on_base(self):
        assert hasattr(BaseModel, "get_per_trade_params")

    def test_with_atr_still_returns_none(self):
        model = DummyModel()
        model.train(pd.DataFrame({"a": [1, 2]}), np.array([0, 1]), TrainingContext())
        result = model.get_per_trade_params(
            pd.DataFrame({"a": [1, 2]}),
            atr=np.array([1.0, 1.0])
        )
        assert result is None
