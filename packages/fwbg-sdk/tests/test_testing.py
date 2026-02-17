import pandas as pd
import numpy as np
from fwbg_sdk.testing import (
    create_sample_ohlcv, assert_features_shifted,
    assert_no_inf, create_sample_asset,
)


def test_create_sample_ohlcv_returns_dataframe():
    df = create_sample_ohlcv(bars=100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert set(df.columns) >= {"O", "H", "L", "C", "V"}


def test_create_sample_ohlcv_is_deterministic_with_seed():
    df1 = create_sample_ohlcv(bars=50, seed=42)
    df2 = create_sample_ohlcv(bars=50, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_create_sample_ohlcv_hlc_valid():
    df = create_sample_ohlcv(bars=200)
    assert (df["H"] >= df["L"]).all()
    assert (df["H"] >= df["O"]).all()
    assert (df["H"] >= df["C"]).all()
    assert (df["L"] <= df["O"]).all()
    assert (df["L"] <= df["C"]).all()


def test_assert_features_shifted_passes_on_shifted():
    from fwbg_sdk import shift_features
    features = {"feat_a": pd.Series([1.0, 2.0, 3.0, 4.0])}
    idx = pd.RangeIndex(4)
    shifted = shift_features(features, idx)
    df = pd.DataFrame({"O": [1, 2, 3, 4]}, index=idx)
    result = pd.concat([df, shifted], axis=1)
    assert_features_shifted(result, ["feat_a"])  # should not raise


def test_assert_features_shifted_fails_on_unshifted():
    df = pd.DataFrame({"O": [1, 2, 3, 4], "feat_a": [1.0, 2.0, 3.0, 4.0]})
    try:
        assert_features_shifted(df, ["feat_a"])
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_assert_no_inf_passes_clean():
    df = pd.DataFrame({"feat": [1.0, 2.0, float("nan"), 4.0]})
    assert_no_inf(df, ["feat"])


def test_assert_no_inf_fails_on_inf():
    df = pd.DataFrame({"feat": [1.0, float("inf"), 3.0]})
    try:
        assert_no_inf(df, ["feat"])
        assert False, "Should have raised"
    except AssertionError:
        pass


def test_create_sample_asset():
    asset = create_sample_asset("EURUSD")
    assert asset.symbol == "EURUSD"
    assert asset.asset_class == "FOREX"
    assert asset.spread > 0
    assert asset.point > 0
