"""Tests for displacement indicator plugin."""

import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_disp = import_plugin_module("fwbg-core", "indicators", "displacement")
if _disp is None:
    pytest.skip("displacement plugin not available", allow_module_level=True)


def _make_ohlc(n=500, seed=42):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "O": close * (1 + rng.normal(0, 0.001, n)),
            "H": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "L": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "C": close,
        },
        index=idx,
    )
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)
    return df


def _make_impulse_df(n=200):
    """Create data with clear impulse moves that produce FVGs."""
    rng = np.random.default_rng(99)
    prices = 100 + np.cumsum(rng.normal(0, 0.3, n))
    for i in range(30, n - 2, 30):
        prices[i] += 5.0
        prices[i + 1] += 6.0
    df = pd.DataFrame(
        {
            "O": prices + rng.normal(0, 0.1, n),
            "H": prices + np.abs(rng.normal(0, 0.5, n)) + 0.5,
            "L": prices - np.abs(rng.normal(0, 0.5, n)) - 0.5,
            "C": prices,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
    return df


def _get_indicator():
    return _disp.DisplacementIndicator()


class TestDisplacementFeatures:
    def test_all_features_present(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        for col in ind.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_features_have_values(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc(n=500))
        late = result.iloc[50:]
        for col in ind.get_feature_columns():
            non_null = late[col].dropna()
            assert len(non_null) > 0, f"{col} all NaN after warmup"

    def test_ratios_between_0_and_1(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc(n=1000))
        for col in [
            "disp_body_ratio",
            "disp_upper_wick_ratio",
            "disp_lower_wick_ratio",
            "disp_close_position",
        ]:
            vals = result[col].dropna()
            assert (vals >= -0.01).all() and (vals <= 1.01).all(), (
                f"{col} out of [0, 1] range"
            )

    def test_fvg_binary(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        vals = result["disp_fvg_formed"].dropna()
        assert set(vals.unique()).issubset({0.0, 1.0})

    def test_fvg_detected_in_impulse_data(self):
        ind = _get_indicator()
        result = ind.compute(_make_impulse_df())
        fvg_count = result["disp_fvg_formed"].dropna().sum()
        assert fvg_count > 0, "Impulse data should produce FVGs"

    def test_range_expansion_positive(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        vals = result["disp_range_expansion"].dropna()
        assert (vals >= 0).all()

    def test_body_atr_positive(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        vals = result["disp_body_atr"].dropna()
        assert (vals >= 0).all()


class TestDisplacementShiftAndInf:
    def test_shift_applied(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc())
        for col in ind.get_feature_columns():
            assert pd.isna(result[col].iloc[0]), f"{col} not shifted"

    def test_no_inf_values(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc(n=1000))
        for col in ind.get_feature_columns():
            vals = result[col].dropna()
            assert not np.isinf(vals).any(), f"{col} has inf"

    def test_no_undeclared_features(self):
        ind = _get_indicator()
        df = _make_ohlc()
        original = set(df.columns)
        result = ind.compute(df)
        undeclared = set(result.columns) - original - set(ind.get_feature_columns())
        assert not undeclared, f"Undeclared: {undeclared}"

    def test_feature_count(self):
        ind = _get_indicator()
        assert len(ind.get_feature_columns()) == 9


class TestDisplacementParameters:
    def test_get_default_params(self):
        params = _disp.DisplacementIndicator.get_default_params()
        assert params["atr_period"] == 14
        assert params["range_avg_period"] == 20

    def test_get_param_schema(self):
        schema = _disp.DisplacementIndicator.get_param_schema()
        assert "atr_period" in schema
        assert "range_avg_period" in schema

    def test_custom_params(self):
        ind = _get_indicator()
        result = ind.compute(_make_ohlc(), atr_period=7, range_avg_period=10)
        assert "disp_body_atr" in result.columns


class TestDisplacementDiscovery:
    def test_plugin_discoverable(self):
        from fwbg.core import discover_plugins, get_indicator

        discover_plugins()
        cls = get_indicator("displacement")
        assert cls is not None


class TestConsecutiveDirectionStart:
    @staticmethod
    def _df(opens, closes):
        n = len(opens)
        o = np.array(opens, dtype=float)
        c = np.array(closes, dtype=float)
        return pd.DataFrame(
            {"O": o, "H": np.maximum(o, c) + 1, "L": np.minimum(o, c) - 1, "C": c},
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )

    def test_run_counts_from_first_bar(self):
        # bullish, bullish, bullish, bearish, bearish, doji, bullish
        df = self._df(
            [10, 11, 12, 13, 12, 11, 11],
            [11, 12, 13, 12, 11, 11, 12],
        )
        result = _get_indicator().compute(df)
        # row r holds bar r-1 (shift); bar 0 starts the run at +1
        got = result["disp_consecutive_dir"].iloc[1:].to_numpy()
        np.testing.assert_array_equal(got, [1.0, 2.0, 3.0, -1.0, -2.0, 0.0])

    def test_leading_bearish_run(self):
        df = self._df([10, 9, 8, 9], [9, 8, 7, 10])
        result = _get_indicator().compute(df)
        got = result["disp_consecutive_dir"].iloc[1:].to_numpy()
        np.testing.assert_array_equal(got, [-1.0, -2.0, -3.0])


class TestFvgWarmup:
    def test_first_bars_are_nan_not_zero(self):
        result = _get_indicator().compute(_make_ohlc(n=50))
        # bars 0 and 1 cannot form a 3-candle pattern -> rows 0..2 after shift
        assert result["disp_fvg_formed"].iloc[:3].isna().all()
        assert result["disp_fvg_formed"].iloc[3:].notna().all()


def _make_body_df(n=300, seed=5):
    """Candles with real bodies so that body / ATR spans a useful range."""
    rng = np.random.default_rng(seed)
    o = 100 + np.cumsum(rng.normal(0, 0.3, n))
    c = o + rng.normal(0, 1.0, n)
    h = np.maximum(o, c) + rng.uniform(0, 0.5, n)
    low = np.minimum(o, c) - rng.uniform(0, 0.5, n)
    return pd.DataFrame(
        {"O": o, "H": h, "L": low, "C": c},
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


class TestImpulseThreshold:
    @staticmethod
    def _ref(df, atr_period, threshold):
        o, h, low, c = (df[k].to_numpy() for k in ("O", "H", "L", "C"))
        prev_c = np.r_[c[0], c[:-1]]
        tr = np.maximum(h - low, np.maximum(abs(h - prev_c), abs(low - prev_c)))
        atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().to_numpy()
        body = abs(c - o)
        return np.where(body / atr >= threshold, np.sign(c - o), 0.0)

    def test_matches_reference(self):
        df = _make_body_df()
        for thr in (0.5, 0.8, 1.2):
            result = _get_indicator().compute(df, impulse_threshold=thr)
            expected = pd.Series(self._ref(df, 14, thr), index=df.index).shift(1)
            assert expected.abs().sum() > 0  # the reference flags something
            pd.testing.assert_series_equal(
                result["disp_impulse"], expected, check_names=False
            )

    def test_threshold_is_monotone_and_default_in_schema(self):
        df = _make_body_df()
        loose = _get_indicator().compute(df, impulse_threshold=0.5)["disp_impulse"]
        strict = _get_indicator().compute(df, impulse_threshold=1.2)["disp_impulse"]
        assert loose.abs().sum() > strict.abs().sum()
        assert (
            _disp.DisplacementIndicator.get_default_params()["impulse_threshold"] == 1.5
        )
        assert "impulse_threshold" in _disp.DisplacementIndicator.get_param_schema()
