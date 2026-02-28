import numpy as np
import pandas as pd
import pytest
from fwbg.plugins import import_plugin_module

_wor = import_plugin_module("fwbg-core", "indicators", "weekly_opening_range")
if _wor is None:
    pytest.skip("weekly_opening_range plugin not available", allow_module_level=True)


def _find_indicator_class(module):
    """Find the concrete indicator class in the module (skip abstract base classes)."""
    import inspect as _inspect
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and not _inspect.isabstract(obj)
            and hasattr(obj, "compute")
            and hasattr(obj, "get_feature_columns")
        ):
            return obj
    raise RuntimeError(f"Could not find indicator class in {module}")


def _make_m15_df(n_weeks=8, known_wor_high=101.0, known_wor_low=99.0, seed=42):
    np.random.seed(seed)
    bars_per_week = 5 * 24 * 4  # 5 days * 24 h * 4 bars/h = 480
    n = n_weeks * bars_per_week
    start = pd.Timestamp("2022-01-03 00:00")  # Monday
    idx = pd.date_range(start, periods=n, freq="15min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.05)
    df = pd.DataFrame({"O": close * 0.9995, "H": close * 1.002, "L": close * 0.998, "C": close}, index=idx)
    df.iloc[0, df.columns.get_loc("H")] = known_wor_high
    df.iloc[0, df.columns.get_loc("L")] = known_wor_low
    df.iloc[1, df.columns.get_loc("H")] = known_wor_high
    df.iloc[1, df.columns.get_loc("L")] = known_wor_low
    return df


def _get_ind():
    cls = _find_indicator_class(_wor)
    return cls()


class TestWORFeaturePresence:
    def test_features_computed_after_warmup(self):
        """WOR features should have non-NaN values after the first week range bars."""
        df = _make_m15_df(n_weeks=10)
        ind = _get_ind()
        result = ind.compute(df)
        wor_cols = [c for c in result.columns if c.startswith("wor_")]
        assert len(wor_cols) > 0, "No wor_* features computed"
        for col in wor_cols:
            valid = result[col].dropna()
            assert len(valid) > 0, f"{col} is all NaN"

    def test_first_bar_is_nan(self):
        """First bar should be NaN (shift applied, no lookahead)."""
        df = _make_m15_df(n_weeks=4)
        ind = _get_ind()
        result = ind.compute(df)
        for col in ind.get_feature_columns():
            if col in result.columns:
                assert pd.isna(result[col].iloc[0]), f"{col} row 0 should be NaN"

    def test_feature_count_positive(self):
        """Indicator must declare at least one feature column."""
        ind = _get_ind()
        assert len(ind.get_feature_columns()) > 0

    def test_returns_dataframe(self):
        """compute() must return a DataFrame."""
        df = _make_m15_df(n_weeks=2)
        ind = _get_ind()
        result = ind.compute(df)
        assert isinstance(result, pd.DataFrame)

    def test_original_columns_preserved(self):
        """Original OHLC columns must be present in the result."""
        df = _make_m15_df(n_weeks=2)
        ind = _get_ind()
        result = ind.compute(df)
        for col in ["O", "H", "L", "C"]:
            assert col in result.columns, f"Column {col} missing from result"


class TestWORPositionAndBreakout:
    def test_position_between_0_and_1_inside_range(self):
        """When close is inside WOR range, position should be in [0, 1]."""
        df = _make_m15_df(known_wor_high=102.0, known_wor_low=98.0)
        ind = _get_ind()
        result = ind.compute(df)
        if "wor_position" not in result.columns:
            pytest.skip("wor_position not in result")
        # Only check the first week where WOR is 98-102 (known boundaries)
        first_week_end = pd.Timestamp("2022-01-07 23:59")
        first_week = result[result.index <= first_week_end]
        inside = first_week[
            (first_week["C"] >= 98.0) & (first_week["C"] <= 102.0) & first_week["wor_position"].notna()
        ]
        if len(inside) > 0:
            assert (inside["wor_position"].between(0, 1)).all(), (
                f"Position inside range should be [0, 1], got "
                f"[{inside['wor_position'].min():.3f}, {inside['wor_position'].max():.3f}]"
            )

    def test_breakout_up_when_close_above_wor_high(self):
        """Bars with close above WOR high should have wor_breakout_up=1."""
        df = _make_m15_df(known_wor_high=101.0, known_wor_low=99.0)
        ind = _get_ind()
        result = ind.compute(df)
        if "wor_breakout_up" not in result.columns:
            pytest.skip("wor_breakout_up not in result")
        breakout_bars = result[result["C"] > 101.0].dropna(subset=["wor_breakout_up"])
        if len(breakout_bars) > 0:
            assert (breakout_bars["wor_breakout_up"] == 1).any(), (
                "Close above WOR high should set wor_breakout_up=1"
            )

    def test_breakout_down_when_close_below_wor_low(self):
        """Bars with close below WOR low should have wor_breakout_down=1."""
        df = _make_m15_df(known_wor_high=101.0, known_wor_low=99.0)
        ind = _get_ind()
        result = ind.compute(df)
        if "wor_breakout_down" not in result.columns:
            pytest.skip("wor_breakout_down not in result")
        breakout_bars = result[result["C"] < 99.0].dropna(subset=["wor_breakout_down"])
        if len(breakout_bars) > 0:
            assert (breakout_bars["wor_breakout_down"] == 1).any(), (
                "Close below WOR low should set wor_breakout_down=1"
            )

    def test_breakout_up_and_down_mutually_exclusive(self):
        """Cannot have wor_breakout_up=1 and wor_breakout_down=1 simultaneously."""
        df = _make_m15_df()
        ind = _get_ind()
        result = ind.compute(df)
        if "wor_breakout_up" not in result.columns or "wor_breakout_down" not in result.columns:
            pytest.skip("breakout columns not in result")
        both = (result["wor_breakout_up"] == 1) & (result["wor_breakout_down"] == 1)
        assert not both.any(), "Cannot have both wor_breakout_up and wor_breakout_down at same bar"

    def test_no_breakout_inside_range(self):
        """Bars strictly inside WOR should have both breakout flags = 0."""
        df = _make_m15_df(known_wor_high=101.0, known_wor_low=99.0)
        ind = _get_ind()
        result = ind.compute(df)
        if "wor_breakout_up" not in result.columns or "wor_breakout_down" not in result.columns:
            pytest.skip("breakout columns not in result")
        # Only check the first week where WOR is known to be 99-101
        first_week_end = pd.Timestamp("2022-01-07 23:59")
        first_week = result[result.index <= first_week_end]
        inside = first_week[
            (first_week["C"] > 99.0) & (first_week["C"] < 101.0)
            & first_week["wor_breakout_up"].notna() & first_week["wor_breakout_down"].notna()
        ]
        if len(inside) > 0:
            assert (inside["wor_breakout_up"] == 0).all(), "Close inside range should have wor_breakout_up=0"
            assert (inside["wor_breakout_down"] == 0).all(), "Close inside range should have wor_breakout_down=0"

    def test_dist_to_high_positive_when_below_wor(self):
        """wor_dist_to_high should be positive when close is below WOR high."""
        df = _make_m15_df(known_wor_high=101.0, known_wor_low=99.0)
        ind = _get_ind()
        result = ind.compute(df)
        if "wor_dist_to_high" not in result.columns:
            pytest.skip("wor_dist_to_high not in result")
        # Restrict to the first week where WOR high is known to be 101.0
        first_week_end = pd.Timestamp("2022-01-07 23:59")
        first_week = result[result.index <= first_week_end]
        below_high = first_week[(first_week["C"] < 101.0) & first_week["wor_dist_to_high"].notna()]
        if len(below_high) > 0:
            assert (below_high["wor_dist_to_high"] > 0).all(), (
                "wor_dist_to_high should be positive when close is below WOR high"
            )

    def test_dist_to_low_positive_when_above_wor(self):
        """wor_dist_to_low should be positive when close is above WOR low."""
        df = _make_m15_df(known_wor_high=101.0, known_wor_low=99.0)
        ind = _get_ind()
        result = ind.compute(df)
        if "wor_dist_to_low" not in result.columns:
            pytest.skip("wor_dist_to_low not in result")
        # Restrict to the first week where WOR low is known to be 99.0
        first_week_end = pd.Timestamp("2022-01-07 23:59")
        first_week = result[result.index <= first_week_end]
        above_low = first_week[(first_week["C"] > 99.0) & first_week["wor_dist_to_low"].notna()]
        if len(above_low) > 0:
            assert (above_low["wor_dist_to_low"] > 0).all(), (
                "wor_dist_to_low should be positive when close is above WOR low"
            )


class TestWORRangeBars:
    def test_range_bars_2_default(self):
        """Default range_bars=2: features are computed and non-empty."""
        df = _make_m15_df(n_weeks=2)
        ind = _get_ind()
        result = ind.compute(df, range_bars=2)
        wor_cols = [c for c in result.columns if c.startswith("wor_")]
        assert len(wor_cols) > 0, "No wor_* features with range_bars=2"

    def test_range_bars_list_adds_prefixed_columns(self):
        """range_bars=[2, 4] should produce columns prefixed wor_rb2_ and wor_rb4_."""
        df = _make_m15_df(n_weeks=4)
        ind = _get_ind()
        result = ind.compute(df, range_bars=[2, 4])
        rb2_cols = [c for c in result.columns if c.startswith("wor_rb2_")]
        rb4_cols = [c for c in result.columns if c.startswith("wor_rb4_")]
        assert len(rb2_cols) > 0, "No wor_rb2_* columns for range_bars=[2, 4]"
        assert len(rb4_cols) > 0, "No wor_rb4_* columns for range_bars=[2, 4]"

    def test_wor_range_valid_after_range_bars(self):
        """First valid wor_range value must appear at or after bar index range_bars."""
        df = _make_m15_df(n_weeks=2)
        ind = _get_ind()
        result = ind.compute(df, range_bars=2)
        if "wor_range" not in result.columns:
            pytest.skip("wor_range not in result")
        first_valid_idx = result["wor_range"].first_valid_index()
        assert first_valid_idx is not None, "wor_range is entirely NaN"
        pos = result.index.get_loc(first_valid_idx)
        assert pos >= 2, f"wor_range valid too early at position {pos}"


class TestWORStatistics:
    def test_avg_range_positive_after_multiple_weeks(self):
        """After several weeks, wor_stat_avg_range should be positive."""
        df = _make_m15_df(n_weeks=10)
        ind = _get_ind()
        result = ind.compute(df)
        if "wor_stat_avg_range" not in result.columns:
            pytest.skip("wor_stat_avg_range not in result")
        valid = result["wor_stat_avg_range"].dropna()
        assert len(valid) > 0, "wor_stat_avg_range should have values after multiple weeks"
        assert (valid > 0).all(), "Average WOR range must be positive"

    def test_breakout_rate_between_0_and_1(self):
        """wor_stat_breakout_rate must be in [0, 1]."""
        df = _make_m15_df(n_weeks=10)
        ind = _get_ind()
        result = ind.compute(df)
        if "wor_stat_breakout_rate" not in result.columns:
            pytest.skip("wor_stat_breakout_rate not in result")
        valid = result["wor_stat_breakout_rate"].dropna()
        if len(valid) > 0:
            assert (valid >= 0).all() and (valid <= 1).all(), (
                f"wor_stat_breakout_rate out of [0, 1]: min={valid.min():.3f}, max={valid.max():.3f}"
            )

    def test_stats_disabled(self):
        """With enable_stats=False, no wor_stat_* columns should appear."""
        df = _make_m15_df(n_weeks=4)
        ind = _get_ind()
        result = ind.compute(df, enable_stats=False)
        stat_cols = [c for c in result.columns if c.startswith("wor_stat_")]
        assert len(stat_cols) == 0, f"wor_stat_* columns present despite enable_stats=False: {stat_cols}"

    def test_daily_data_not_processed(self):
        """Daily timeframe data should be returned unchanged (no wor_* columns added)."""
        n = 100
        idx = pd.date_range("2022-01-03", periods=n, freq="D")
        df = pd.DataFrame({"O": 100.0, "H": 101.0, "L": 99.0, "C": 100.0}, index=idx)
        ind = _get_ind()
        result = ind.compute(df)
        wor_cols = [c for c in result.columns if c.startswith("wor_")]
        assert len(wor_cols) == 0, (
            f"Daily data should not produce any WOR feature columns, got: {wor_cols}"
        )

    def test_weekly_data_not_processed(self):
        """Weekly timeframe data should also be returned unchanged."""
        n = 50
        idx = pd.date_range("2022-01-03", periods=n, freq="W")
        df = pd.DataFrame({"O": 100.0, "H": 101.0, "L": 99.0, "C": 100.0}, index=idx)
        ind = _get_ind()
        result = ind.compute(df)
        wor_cols = [c for c in result.columns if c.startswith("wor_")]
        assert len(wor_cols) == 0, (
            f"Weekly data should not produce any WOR feature columns, got: {wor_cols}"
        )


class TestPluginAttributes:
    def test_no_inf_values(self):
        """No feature column may contain inf values."""
        df = _make_m15_df(n_weeks=6)
        ind = _get_ind()
        result = ind.compute(df)
        for col in ind.get_feature_columns():
            if col in result.columns:
                bad = result[col].isin([float("inf"), float("-inf")])
                assert not bad.any(), f"{col} contains inf values"

    def test_only_declared_features_added(self):
        """compute() must not add columns not declared in get_feature_columns()."""
        df = _make_m15_df(n_weeks=4)
        ind = _get_ind()
        result = ind.compute(df)
        original = set(df.columns)
        new_cols = set(result.columns) - original
        declared = set(ind.get_feature_columns())
        undeclared = new_cols - declared
        assert not undeclared, f"Undeclared columns added: {undeclared}"

    def test_idempotent(self):
        """Calling compute twice on the same df should produce identical results."""
        df = _make_m15_df(n_weeks=4)
        ind = _get_ind()
        result1 = ind.compute(df)
        result2 = ind.compute(df)
        wor_cols = [c for c in result1.columns if c.startswith("wor_")]
        for col in wor_cols:
            pd.testing.assert_series_equal(result1[col], result2[col], check_names=True)

    def test_indicator_name(self):
        """Indicator must have the correct name attribute."""
        ind = _get_ind()
        assert ind.name == "weekly_opening_range"

    def test_indicator_version(self):
        """Indicator must have a non-empty version string."""
        ind = _get_ind()
        assert hasattr(ind, "version") and isinstance(ind.version, str)
        assert len(ind.version) > 0

    def test_get_default_params(self):
        """get_default_params must return a dict with expected keys."""
        cls = _find_indicator_class(_wor)
        params = cls.get_default_params()
        assert isinstance(params, dict)
        for key in ("range_bars", "atr_period", "stat_window", "enable_stats"):
            assert key in params, f"Missing key {key!r} in get_default_params()"

    def test_wor_range_non_negative(self):
        """wor_range (normalized OR range) must be non-negative where valid."""
        df = _make_m15_df(n_weeks=4)
        ind = _get_ind()
        result = ind.compute(df)
        if "wor_range" not in result.columns:
            pytest.skip("wor_range not in result")
        valid = result["wor_range"].dropna()
        assert (valid >= 0).all(), "wor_range should be non-negative"
