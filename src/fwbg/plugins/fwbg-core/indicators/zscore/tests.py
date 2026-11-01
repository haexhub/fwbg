"""Tests for the zscore indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

ZscoreIndicator = import_plugin_module("fwbg-core", "indicators", "zscore").ZscoreIndicator


def frame(closes):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"O": c, "H": c, "L": c, "C": c}, index=pd.date_range("2024-01-01", periods=len(c), freq="B"))


def test_zscore_of_the_close_against_its_window():
    out = ZscoreIndicator().compute(frame([1, 2, 3, 4]), lookbacks=[3], use_log=False)
    assert out["zscore_3"].iloc[:2].isna().all()
    assert out["zscore_3"].iloc[2] == pytest.approx(1.0)  # (3 - 2) / 1
    assert out["zscore_3"].iloc[3] == pytest.approx(1.0)


def test_log_zscore_is_scale_free():
    a = ZscoreIndicator().compute(frame([100, 101, 99, 103, 98]), lookbacks=[3])["zscore_3"]
    b = ZscoreIndicator().compute(frame([1, 1.01, 0.99, 1.03, 0.98]), lookbacks=[3])["zscore_3"]
    pd.testing.assert_series_equal(a, b)


def test_exit_flags_mark_the_side_of_the_mean():
    out = ZscoreIndicator().compute(frame([1, 2, 3, 2, 1]), lookbacks=[3], use_log=False)
    assert out["zscore_3_above_mean"].iloc[2:].tolist() == [1.0, 0.0, 0.0]
    assert out["zscore_3_below_mean"].iloc[2:].tolist() == [0.0, 1.0, 1.0]
    assert out[["zscore_3_above_mean", "zscore_3_below_mean"]].iloc[:2].isna().all().all()


def test_several_lookbacks_and_flat_window():
    out = ZscoreIndicator().compute(frame([5, 5, 5, 6]), lookbacks=[2, 3], use_log=False)
    assert {"zscore_2", "zscore_3"} <= set(out.columns)
    assert np.isnan(out["zscore_3"].iloc[2])  # no dispersion -> no z-score


@pytest.mark.parametrize("bad", [[1], [2.5], ["20"], [True]])
def test_invalid_lookbacks_raise(bad):
    with pytest.raises(ValueError, match="lookbacks"):
        ZscoreIndicator().compute(frame([1, 2, 3]), lookbacks=bad)


def test_prefix_stable():
    df = frame(np.cumsum(np.random.default_rng(1).normal(0, 1, 60)) + 100)
    full = ZscoreIndicator().compute(df, lookbacks=[10])
    prefix = ZscoreIndicator().compute(df.iloc[:40], lookbacks=[10])
    pd.testing.assert_frame_equal(full.iloc[:40], prefix)


def test_entry_threshold_flags_with_lookback_free_names():
    out = ZscoreIndicator().compute(frame([1, 2, 3, 2, 1, 1]), lookbacks=[3], use_log=False, entry_z=0.9)
    z = out["zscore_3"]
    pd.testing.assert_series_equal(out["zscore_entry_long"], (z < -0.9).astype(float).where(z.notna()), check_names=False)
    pd.testing.assert_series_equal(out["zscore_entry_short"], (z > 0.9).astype(float).where(z.notna()), check_names=False)
    pd.testing.assert_series_equal(out["zscore_exit_long"], out["zscore_3_above_mean"], check_names=False)
    pd.testing.assert_series_equal(out["zscore_exit_short"], out["zscore_3_below_mean"], check_names=False)
    assert out["zscore_entry_short"].iloc[2] == 1.0 and out["zscore_entry_long"].iloc[4] == 1.0


def test_entry_threshold_columns_only_when_set():
    ind = ZscoreIndicator()
    assert "zscore_entry_long" not in ind.compute(frame([1, 2, 3]), lookbacks=[2]).columns
    cols = ind.get_feature_columns({"lookbacks": [2], "entry_z": 2.0})
    assert {"zscore_entry_long", "zscore_entry_short", "zscore_exit_long", "zscore_exit_short"} <= set(cols)


@pytest.mark.parametrize("params", [
    {"lookbacks": [2, 3], "entry_z": 2.0},
    {"lookbacks": [3], "entry_z": -1.0},
    {"lookbacks": [3], "entry_z": float("nan")},
    {"lookbacks": [3], "entry_z": "2"},
])
def test_invalid_entry_threshold_raises(params):
    with pytest.raises(ValueError, match="entry_z"):
        ZscoreIndicator().compute(frame([1, 2, 3, 4]), use_log=False, **params)
