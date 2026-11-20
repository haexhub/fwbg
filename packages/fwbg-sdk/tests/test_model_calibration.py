"""calibrate() must change what the pipeline reads as the model's probabilities."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fwbg_sdk.models import BaseModel


class _OverconfidentModel(BaseModel):
    """Logistic regression whose raw probabilities are pushed towards 0/1."""

    name = "overconfident_test_model"
    version = "1.0.0"

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def train(self, features, targets, training_context, **hyperparameters):
        self._model = LogisticRegression(max_iter=500).fit(features, targets)
        self._fitted = True

    def _predict_probability_impl(self, features):
        p = self._model.predict_proba(features)[:, 1]
        sharp = np.clip(0.5 + 3.0 * (p - 0.5), 0.0, 1.0)  # overconfident on purpose
        return np.column_stack([1.0 - sharp, sharp])

    @property
    def _trained_classes_impl(self):
        return self._model.classes_

    def _as_sklearn_estimator_impl(self):
        return self._model


def _data(n=600):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(n, 2))
    # noisy labels: the true win probability is far from 0/1
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-0.8 * x[:, 0]))).astype(int)
    return pd.DataFrame(x, columns=["a", "b"]), y


def test_predict_probability_is_calibrated_after_calibrate():
    X, y = _data()
    model = _OverconfidentModel()
    model.train(X, y, None)
    raw = model.predict_probability(X)

    model.calibrate(X, y, method="sigmoid")

    out = model.predict_probability(X)
    assert not np.allclose(out, raw)
    np.testing.assert_allclose(out, model._calibrated_model.predict_proba(X))
    np.testing.assert_allclose(model.predict_probability_raw(X), raw)
    np.testing.assert_allclose(model.predict_probability_calibrated(X), out)
    # calibration pulls the overconfident tails back in: less spread than the raw probabilities
    assert out[:, 1].std() < raw[:, 1].std()


def test_without_calibration_predict_probability_is_raw():
    X, y = _data()
    model = _OverconfidentModel()
    model.train(X, y, None)
    np.testing.assert_allclose(
        model.predict_probability(X), model.predict_probability_raw(X)
    )


def test_reset_drops_the_calibration():
    X, y = _data()
    model = _OverconfidentModel()
    model.train(X, y, None)
    model.calibrate(X, y, method="sigmoid")
    model.reset()
    assert model._calibrated_model is None
