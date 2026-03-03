"""Tests for the signal model plugin (v3.0.0).

SignalModel reads _composed_signal_{direction} columns directly.
No hyperparameters, no hour filter.
"""
import numpy as np
import pandas as pd
import pytest

from fwbg_sdk.models import TrainingContext
from fwbg.plugins import import_plugin_module

_mod = import_plugin_module("fwbg-core", "models", "signal")
if _mod is None:
    pytest.skip("fwbg-core signal model plugin not available", allow_module_level=True)
SignalModel = _mod.SignalModel


def _make_df(n=10, long_signal=None, short_signal=None):
    """Create a test DataFrame with composed signal columns."""
    df = pd.DataFrame({
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n),
    })
    if long_signal is not None:
        df["_composed_signal_long"] = long_signal
    if short_signal is not None:
        df["_composed_signal_short"] = short_signal
    return df


def test_train_sets_composed_signal_col_long():
    model = SignalModel()
    df = _make_df(5, long_signal=[0, 1, 0, 0, 1])
    targets = np.array([0, 1, 0, 0, 1])
    model.train(df, targets, TrainingContext(direction="long"))
    assert model._signal_col == "_composed_signal_long"
    assert model.is_trained


def test_train_sets_composed_signal_col_short():
    model = SignalModel()
    df = _make_df(5, short_signal=[0, 0, 0, 1, 0])
    targets = np.array([0, 0, 0, 1, 0])
    model.train(df, targets, TrainingContext(direction="short"))
    assert model._signal_col == "_composed_signal_short"
    assert model.is_trained


def test_predict_probability_long():
    model = SignalModel()
    long_sig = [0, 1, 0, 0, 1]
    df = _make_df(5, long_signal=long_sig, short_signal=[0, 0, 0, 1, 0])
    targets = np.array([0, 1, 0, 0, 1])
    model.train(df, targets, TrainingContext(direction="long"))
    probs = model.predict_probability(df)
    assert probs.shape == (5, 2)
    np.testing.assert_array_equal(probs[:, 1], [0, 1, 0, 0, 1])
    np.testing.assert_array_equal(probs[:, 0], [1, 0, 1, 1, 0])


def test_predict_probability_short():
    model = SignalModel()
    short_sig = [0, 0, 0, 1, 0]
    df = _make_df(5, long_signal=[0, 1, 0, 0, 1], short_signal=short_sig)
    targets = np.array([0, 0, 0, 1, 0])
    model.train(df, targets, TrainingContext(direction="short"))
    probs = model.predict_probability(df)
    np.testing.assert_array_equal(probs[:, 1], [0, 0, 0, 1, 0])


def test_trained_classes():
    model = SignalModel()
    df = _make_df(3, long_signal=[0, 1, 0])
    targets = np.array([0, 1, 0])
    model.train(df, targets, TrainingContext(direction="long"))
    np.testing.assert_array_equal(model.trained_classes, [0, 1])


def test_nan_in_signal_treated_as_zero():
    model = SignalModel()
    df = _make_df(4, long_signal=[0, np.nan, 1, np.nan])
    targets = np.array([0, 0, 1, 0])
    model.train(df, targets, TrainingContext(direction="long"))
    probs = model.predict_probability(df)
    np.testing.assert_array_equal(probs[:, 1], [0, 0, 1, 0])


def test_missing_signal_column_returns_zeros():
    model = SignalModel()
    df = _make_df(3)  # No signal columns
    targets = np.array([0, 1, 0])
    model.train(df, targets, TrainingContext(direction="long"))
    probs = model.predict_probability(df)
    np.testing.assert_array_equal(probs[:, 1], [0, 0, 0])
    np.testing.assert_array_equal(probs[:, 0], [1, 1, 1])


def test_ct_threshold_mechanism():
    """CT=0.5 correctly filters signal vs non-signal bars."""
    model = SignalModel()
    df = _make_df(5, long_signal=[0, 1, 0, 1, 0])
    targets = np.array([0, 1, 0, 1, 0])
    model.train(df, targets, TrainingContext(direction="long"))
    probs = model.predict_probability(df)
    ct = 0.5
    win_idx = np.where(model.trained_classes == 1)[0][0]
    passes = probs[:, win_idx] >= ct
    np.testing.assert_array_equal(passes, [False, True, False, True, False])


def test_reduced_hyperparameters_returns_empty():
    """SignalModel has no hyperparameters — always returns empty dict."""
    params = {"some_key": "value"}
    reduced = SignalModel.get_reduced_hyperparameters(params)
    assert reduced == {}


def test_extra_hyperparameters_are_ignored():
    """Any kwargs passed to train() are silently ignored."""
    model = SignalModel()
    df = _make_df(3, long_signal=[0, 1, 0])
    targets = np.array([0, 1, 0])
    model.train(
        df, targets, TrainingContext(direction="long"),
        some_unknown_param="value",
    )
    assert model.is_trained
    assert model._signal_col == "_composed_signal_long"


def test_signal_values_clipped_to_0_1():
    """Signal values > 1 are clipped to 1, < 0 to 0."""
    model = SignalModel()
    df = _make_df(4, long_signal=[-0.5, 0.5, 1.5, 2.0])
    targets = np.array([0, 1, 1, 1])
    model.train(df, targets, TrainingContext(direction="long"))
    probs = model.predict_probability(df)
    np.testing.assert_array_equal(probs[:, 1], [0, 0.5, 1.0, 1.0])
