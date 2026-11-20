"""Regression tests for train-only stateful indicator fitting."""

import numpy as np
import pandas as pd
import pytest

from fwbg.pipeline.features import compute_indicator_pool
from fwbg.optimization.nested_cv import refit_stateful_features
from fwbg.plugins import import_plugin_module


import_plugin_module("fwbg-core", "indicators", "autoencoder_features")


def _frame(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.2, n))
    frame = pd.DataFrame(
        {
            "O": close,
            "H": close + 0.1,
            "L": close - 0.1,
            "C": close,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    for i in range(10):
        frame[f"feature_{i}"] = rng.normal(size=n)
    return frame


def test_stateful_pool_requires_an_explicit_training_frame():
    with pytest.raises(ValueError, match="require fit_df"):
        compute_indicator_pool(
            _frame(40), indicators=["fwbg-core:autoencoder_features"]
        )


def test_autoencoder_transform_isolated_from_future_tail():
    frame = _frame()
    train = frame.iloc[:120]
    evaluation = frame.iloc[120:].copy()

    baseline = compute_indicator_pool(
        evaluation,
        indicators=["fwbg-core:autoencoder_features"],
        fit_df=train,
    )

    perturbed = evaluation.copy()
    perturbed.iloc[30:, perturbed.columns.get_loc("feature_0")] += 10_000.0
    changed = compute_indicator_pool(
        perturbed,
        indicators=["fwbg-core:autoencoder_features"],
        fit_df=train,
    )

    output_columns = [
        "ae_latent_0",
        "ae_latent_1",
        "ae_reconstruction_error",
        "ae_explained_variance",
    ]
    np.testing.assert_allclose(
        baseline[output_columns].iloc[:30],
        changed[output_columns].iloc[:30],
        equal_nan=True,
    )
    assert not np.allclose(
        baseline["ae_reconstruction_error"].iloc[31:].fillna(0),
        changed["ae_reconstruction_error"].iloc[31:].fillna(0),
    )


def test_inner_refit_uses_only_inner_training_rows():
    frame = _frame()
    outer_train = frame.iloc[:180]
    outer_features = compute_indicator_pool(
        outer_train,
        indicators=["fwbg-core:autoencoder_features"],
        fit_df=outer_train,
    )
    base_columns = set(outer_train.columns)
    output_columns = [c for c in outer_features if c not in base_columns]
    configs = [{"name": "fwbg-core:autoencoder_features", "params": {}}]

    inner_train = outer_features.iloc[:100].copy()
    inner_val = outer_features.iloc[100:140].copy()
    for part in (inner_train, inner_val):
        part.attrs["fwbg_stateful_indicator_configs"] = configs
        part.attrs["fwbg_stateful_output_columns"] = output_columns

    baseline_train, baseline_val = refit_stateful_features([(inner_train, inner_val)])[
        0
    ]
    mutated_val = inner_val.copy()
    mutated_val.iloc[20:, mutated_val.columns.get_loc("feature_0")] += 10_000.0
    changed_train, changed_val = refit_stateful_features([(inner_train, mutated_val)])[
        0
    ]

    np.testing.assert_allclose(
        baseline_train[["ae_latent_0", "ae_reconstruction_error"]],
        changed_train[["ae_latent_0", "ae_reconstruction_error"]],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        baseline_val[["ae_latent_0", "ae_reconstruction_error"]].iloc[:20],
        changed_val[["ae_latent_0", "ae_reconstruction_error"]].iloc[:20],
        equal_nan=True,
    )


def test_refit_trims_only_the_leading_warm_up_and_keeps_the_index_contiguous(
    monkeypatch,
):
    """dropna() on the whole frame removed interior rows too: holes in the bar sequence."""
    import fwbg.pipeline.features as features
    from fwbg.optimization.nested_cv import _trim_leading_nan

    def fake_pool(frame, indicators, fit_df):
        out = frame.copy()
        out["ae_latent_0"] = np.arange(len(out), dtype=float)
        out.iloc[0, out.columns.get_loc("ae_latent_0")] = np.nan  # shifted by design
        out.iloc[5, out.columns.get_loc("ae_latent_0")] = np.nan  # interior gap
        out.iloc[7, out.columns.get_loc("feature_0")] = np.nan  # not a stateful column
        return out

    monkeypatch.setattr(features, "compute_indicator_pool", fake_pool)
    frame = _frame(40)
    train, val = frame.iloc[:20].copy(), frame.iloc[20:].copy()
    for part in (train, val):
        part.attrs["fwbg_stateful_indicator_configs"] = [{"name": "x", "params": {}}]
        part.attrs["fwbg_stateful_output_columns"] = ["ae_latent_0"]

    out_train, out_val = refit_stateful_features([(train, val)])[0]

    # warm-up row 0 is gone, every later row stays (interior NaN is kept as NaN)
    assert list(out_train.index) == list(train.index[1:])
    assert list(out_val.index) == list(val.index[1:])
    assert np.isnan(out_train["ae_latent_0"].loc[train.index[5]])
    assert out_train.attrs["fwbg_stateful_output_columns"] == ["ae_latent_0"]

    allnan = pd.DataFrame({"a": [np.nan, np.nan]})
    assert _trim_leading_nan(allnan, ["a"]).empty
    assert _trim_leading_nan(allnan, []).shape == (2, 1)
