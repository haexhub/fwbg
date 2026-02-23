"""Tests for the signal model plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg_sdk.models import TrainingContext
from fwbg.plugins import import_plugin_module

_mod = import_plugin_module("fwbg-core", "models", "signal")
if _mod is None:
    pytest.skip("fwbg-core signal model plugin not available", allow_module_level=True)
SignalModel = _mod.SignalModel


def _make_df(n=10, signal_bull=None, signal_bear=None):
    """Create a test DataFrame with signal columns."""
    df = pd.DataFrame({
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n),
    })
    if signal_bull is not None:
        df["orb_s0_retest_bull"] = signal_bull
    if signal_bear is not None:
        df["orb_s0_retest_bear"] = signal_bear
    return df


def test_train_sets_direction_long():
    model = SignalModel()
    df = _make_df(5, signal_bull=[0, 1, 0, 0, 1], signal_bear=[0, 0, 0, 1, 0])
    targets = np.array([0, 1, 0, -1, 1])
    ctx = TrainingContext(direction="long")
    model.train(
        df, targets, ctx,
        signal_column_long="orb_s0_retest_bull",
        signal_column_short="orb_s0_retest_bear",
    )
    assert model._signal_col == "orb_s0_retest_bull"
    assert model.is_trained


def test_train_sets_direction_short():
    model = SignalModel()
    df = _make_df(5, signal_bull=[0, 1, 0, 0, 1], signal_bear=[0, 0, 0, 1, 0])
    targets = np.array([0, 1, 0, -1, 1])
    ctx = TrainingContext(direction="short")
    model.train(
        df, targets, ctx,
        signal_column_long="orb_s0_retest_bull",
        signal_column_short="orb_s0_retest_bear",
    )
    assert model._signal_col == "orb_s0_retest_bear"
    assert model.is_trained


def test_predict_probability_long_signal():
    model = SignalModel()
    bull = [0, 1, 0, 0, 1]
    bear = [0, 0, 0, 1, 0]
    df = _make_df(5, signal_bull=bull, signal_bear=bear)
    targets = np.array([0, 1, 0, -1, 1])
    model.train(
        df, targets, TrainingContext(direction="long"),
        signal_column_long="orb_s0_retest_bull",
        signal_column_short="orb_s0_retest_bear",
    )
    probs = model.predict_probability(df)
    assert probs.shape == (5, 2)
    # Win class (idx 1) should match bull signal
    np.testing.assert_array_equal(probs[:, 1], [0, 1, 0, 0, 1])
    np.testing.assert_array_equal(probs[:, 0], [1, 0, 1, 1, 0])


def test_predict_probability_short_signal():
    model = SignalModel()
    bull = [0, 1, 0, 0, 1]
    bear = [0, 0, 0, 1, 0]
    df = _make_df(5, signal_bull=bull, signal_bear=bear)
    targets = np.array([0, 1, 0, -1, 1])
    model.train(
        df, targets, TrainingContext(direction="short"),
        signal_column_long="orb_s0_retest_bull",
        signal_column_short="orb_s0_retest_bear",
    )
    probs = model.predict_probability(df)
    # Win class (idx 1) should match bear signal
    np.testing.assert_array_equal(probs[:, 1], [0, 0, 0, 1, 0])


def test_trained_classes():
    model = SignalModel()
    df = _make_df(3, signal_bull=[0, 1, 0])
    targets = np.array([0, 1, 0])
    model.train(
        df, targets, TrainingContext(direction="long"),
        signal_column_long="orb_s0_retest_bull",
    )
    np.testing.assert_array_equal(model.trained_classes, [0, 1])


def test_nan_in_signal_treated_as_zero():
    model = SignalModel()
    df = _make_df(4, signal_bull=[0, np.nan, 1, np.nan])
    targets = np.array([0, 0, 1, 0])
    model.train(
        df, targets, TrainingContext(direction="long"),
        signal_column_long="orb_s0_retest_bull",
    )
    probs = model.predict_probability(df)
    np.testing.assert_array_equal(probs[:, 1], [0, 0, 1, 0])


def test_missing_signal_column_returns_zeros():
    model = SignalModel()
    df = _make_df(3)  # No signal columns
    targets = np.array([0, 1, 0])
    model.train(
        df, targets, TrainingContext(direction="long"),
        signal_column_long="orb_s0_retest_bull",
    )
    probs = model.predict_probability(df)
    np.testing.assert_array_equal(probs[:, 1], [0, 0, 0])
    np.testing.assert_array_equal(probs[:, 0], [1, 1, 1])


def test_ct_threshold_mechanism():
    """Verify that CT=0.5 correctly filters signal vs non-signal bars."""
    model = SignalModel()
    bull = [0, 1, 0, 1, 0]
    df = _make_df(5, signal_bull=bull)
    targets = np.array([0, 1, 0, 1, 0])
    model.train(
        df, targets, TrainingContext(direction="long"),
        signal_column_long="orb_s0_retest_bull",
    )
    probs = model.predict_probability(df)
    ct = 0.5
    win_idx = np.where(model.trained_classes == 1)[0][0]
    # Only signal bars should pass the threshold
    passes = probs[:, win_idx] >= ct
    np.testing.assert_array_equal(passes, [False, True, False, True, False])


def test_reduced_hyperparameters_passthrough():
    params = {"signal_column_long": "x", "signal_column_short": "y"}
    reduced = SignalModel.get_reduced_hyperparameters(params)
    assert reduced == params
    assert reduced is not params  # Should be a copy


def test_direction_none_defaults_to_short():
    """When direction is None (legacy), default to short column."""
    model = SignalModel()
    df = _make_df(3, signal_bull=[0, 1, 0], signal_bear=[1, 0, 0])
    targets = np.array([1, 0, 0])
    model.train(
        df, targets, TrainingContext(direction=None),
        signal_column_long="orb_s0_retest_bull",
        signal_column_short="orb_s0_retest_bear",
    )
    # direction=None -> else branch -> short column
    assert model._signal_col == "orb_s0_retest_bear"


def _make_df_with_datetime_index(n=24, signal_col="sig"):
    """DataFrame with DatetimeIndex and a signal column (1 signal per bar)."""
    idx = pd.date_range("2024-06-01 00:00", periods=n, freq="h")
    df = pd.DataFrame({
        "feature_a": np.random.randn(n),
        signal_col: np.ones(n),  # all bars have signal=1
    }, index=idx)
    return df


def test_hour_filter_zeros_outside_window():
    """Signals outside [7, 17) should be zeroed out."""
    df = _make_df_with_datetime_index(24, signal_col="sig")
    model = SignalModel()
    targets = np.ones(24)
    model.train(
        df, targets, TrainingContext(direction="long"),
        signal_column_long="sig",
        signal_start_hour=7,
        signal_end_hour=17,
    )
    probs = model.predict_probability(df)

    # Hours 0-6 and 17-23 should have prob=0 for win class
    hours = df.index.hour
    for i, h in enumerate(hours):
        if 7 <= h < 17:
            assert probs[i, 1] == 1.0, f"Hour {h} should pass"
        else:
            assert probs[i, 1] == 0.0, f"Hour {h} should be filtered"


def test_no_hour_filter_by_default():
    """Without start/end hour, all signals pass through."""
    df = _make_df_with_datetime_index(24, signal_col="sig")
    model = SignalModel()
    targets = np.ones(24)
    model.train(
        df, targets, TrainingContext(direction="long"),
        signal_column_long="sig",
    )
    probs = model.predict_probability(df)
    np.testing.assert_array_equal(probs[:, 1], np.ones(24))


def test_hour_filter_with_non_datetime_index():
    """Hour filter is silently skipped for non-DatetimeIndex."""
    df = pd.DataFrame({
        "feature_a": np.random.randn(5),
        "sig": [0, 1, 1, 0, 1],
    })
    model = SignalModel()
    targets = np.array([0, 1, 1, 0, 1])
    model.train(
        df, targets, TrainingContext(direction="long"),
        signal_column_long="sig",
        signal_start_hour=7,
        signal_end_hour=17,
    )
    probs = model.predict_probability(df)
    np.testing.assert_array_equal(probs[:, 1], [0, 1, 1, 0, 1])
