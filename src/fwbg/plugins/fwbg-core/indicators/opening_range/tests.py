"""Tests for Opening Range Breakout indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_orb = import_plugin_module("fwbg-core", "indicators", "opening_range")
if _orb is None:
    pytest.skip("fwbg-core opening_range plugin not available", allow_module_level=True)


def _make_ohlc_15min(n=2000, seed=42):
    """Create OHLCV DataFrame with 15-minute frequency."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")

    df = pd.DataFrame({
        "O": close * (1 + rng.normal(0, 0.001, n)),
        "H": close * (1 + np.abs(rng.normal(0, 0.003, n))),
        "L": close * (1 - np.abs(rng.normal(0, 0.003, n))),
        "C": close,
    }, index=idx)
    # Ensure H >= max(O,C) and L <= min(O,C)
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)
    return df


def _make_ohlc_hourly(n=2000, seed=42):
    """Create OHLCV DataFrame with hourly frequency."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")

    df = pd.DataFrame({
        "O": close * (1 + rng.normal(0, 0.001, n)),
        "H": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "L": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "C": close,
    }, index=idx)
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)
    return df


def _make_ohlc_daily(n=500, seed=42):
    """Create OHLCV DataFrame with daily frequency."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")

    df = pd.DataFrame({
        "O": close * (1 + rng.normal(0, 0.005, n)),
        "H": close * (1 + np.abs(rng.normal(0, 0.01, n))),
        "L": close * (1 - np.abs(rng.normal(0, 0.01, n))),
        "C": close,
    }, index=idx)
    df["H"] = df[["O", "H", "C"]].max(axis=1)
    df["L"] = df[["O", "L", "C"]].min(axis=1)
    return df


def _get_indicator():
    return _orb.OpeningRangeIndicator()


class TestRollingORB:
    """Tests for rolling (hourly) ORB features."""

    def test_rolling_features_computed(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df)

        for col in ["orb_range", "orb_position", "orb_breakout_up",
                     "orb_breakout_down", "orb_range_vs_atr"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_rolling_features_have_values(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=2000)
        result = ind.compute(df)

        for col in ["orb_range", "orb_position", "orb_range_vs_atr"]:
            non_null = result[col].dropna()
            assert len(non_null) > 0, f"{col} is all NaN"

    def test_range_positive(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df)

        range_vals = result["orb_range"].dropna()
        assert (range_vals >= 0).all(), "orb_range should be non-negative"

    def test_breakout_binary(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df)

        for col in ["orb_breakout_up", "orb_breakout_down"]:
            vals = result[col].dropna()
            assert set(vals.unique()).issubset({0, 1}), f"{col} should be binary"

    def test_position_ranges(self):
        """Position should be around 0-1 mostly, can exceed for breakouts."""
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df)

        pos = result["orb_position"].dropna()
        # Most values should be in reasonable range
        within_range = ((pos >= -1) & (pos <= 2)).mean()
        assert within_range > 0.8, "Most positions should be in [-1, 2] range"

    def test_no_lookahead(self):
        """First row of every feature should be NaN (shifted by 1)."""
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df)

        for col in ["orb_range", "orb_position", "orb_breakout_up"]:
            assert pd.isna(result[col].iloc[0]), f"{col} first row should be NaN"

    def test_hourly_data_rolling_is_nan(self):
        """On hourly data with range_bars=1, rolling ORB is all NaN
        (only 1 bar per hour → no bar after range is established)."""
        ind = _get_indicator()
        df = _make_ohlc_hourly()
        result = ind.compute(df)

        assert "orb_range" in result.columns
        # Rolling features are NaN because each hour has only 1 bar
        assert result["orb_range"].dropna().empty

    def test_hourly_data_session_features_work(self):
        """Session ORB should work on hourly data (range persists across hours)."""
        ind = _get_indicator()
        df = _make_ohlc_hourly(n=5000)
        result = ind.compute(df)

        non_null = result["orb_s08_range"].dropna()
        assert len(non_null) > 0

    def test_no_inf_values(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df)

        feature_cols = [c for c in result.columns if c.startswith("orb_")]
        for col in feature_cols:
            vals = result[col].dropna()
            assert not np.isinf(vals).any(), f"{col} contains inf values"


class TestSessionORB:
    """Tests for session-specific ORB features."""

    def test_session_features_computed(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df)

        for h in [0, 8, 13, 14]:
            prefix = f"orb_s{h:02d}"
            for suffix in ["_range", "_position", "_breakout_up",
                           "_breakout_down", "_range_vs_atr"]:
                col = f"{prefix}{suffix}"
                assert col in result.columns, f"Missing column: {col}"

    def test_session_features_have_values(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)  # ~52 days, enough for all sessions
        result = ind.compute(df)

        for h in [0, 8, 13, 14]:
            col = f"orb_s{h:02d}_range"
            non_null = result[col].dropna()
            assert len(non_null) > 0, f"{col} is all NaN"

    def test_custom_sessions(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df, sessions=[9, 17])

        assert "orb_s09_range" in result.columns
        assert "orb_s17_range" in result.columns
        # Default sessions should NOT be present
        assert "orb_s00_range" not in result.columns

    def test_session_breakout_binary(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df)

        for h in [0, 8, 13, 14]:
            for direction in ["up", "down"]:
                col = f"orb_s{h:02d}_breakout_{direction}"
                vals = result[col].dropna()
                if len(vals) > 0:
                    assert set(vals.unique()).issubset({0, 1}), f"{col} not binary"


class TestStatFeatures:
    """Tests for rolling statistical features."""

    def test_stat_features_computed(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df)

        for col in ["orb_stat_avg_range", "orb_stat_breakout_rate",
                     "orb_stat_continuation_rate"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_stat_features_have_values(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df)

        for col in ["orb_stat_avg_range", "orb_stat_breakout_rate",
                     "orb_stat_continuation_rate"]:
            non_null = result[col].dropna()
            assert len(non_null) > 0, f"{col} is all NaN"

    def test_rates_between_0_and_1(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=5000)
        result = ind.compute(df)

        for col in ["orb_stat_breakout_rate", "orb_stat_continuation_rate"]:
            vals = result[col].dropna()
            if len(vals) > 0:
                assert (vals >= 0).all() and (vals <= 1).all(), \
                    f"{col} should be between 0 and 1"


class TestDailySkip:
    """Daily data should not produce ORB features."""

    def test_daily_returns_unchanged(self):
        ind = _get_indicator()
        df = _make_ohlc_daily()
        result = ind.compute(df)

        orb_cols = [c for c in result.columns if c.startswith("orb_")]
        assert len(orb_cols) == 0, "Daily data should not produce ORB features"


class TestParameters:
    """Test parameter variations."""

    def test_range_bars_2(self):
        ind = _get_indicator()
        df = _make_ohlc_15min(n=3000)
        result = ind.compute(df, range_bars=2)

        assert "orb_range" in result.columns
        non_null = result["orb_range"].dropna()
        assert len(non_null) > 0

    def test_disable_rolling(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df, enable_rolling=False)

        assert "orb_range" not in result.columns
        assert "orb_s08_range" in result.columns

    def test_disable_session(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df, enable_session=False)

        assert "orb_range" in result.columns
        assert "orb_s08_range" not in result.columns

    def test_disable_stats(self):
        ind = _get_indicator()
        df = _make_ohlc_15min()
        result = ind.compute(df, enable_stats=False)

        assert "orb_stat_avg_range" not in result.columns

    def test_get_default_params(self):
        params = _orb.OpeningRangeIndicator.get_default_params()
        assert params["range_bars"] == 1
        assert params["sessions"] == [0, 8, 13, 14]

    def test_get_param_schema(self):
        schema = _orb.OpeningRangeIndicator.get_param_schema()
        assert "range_bars" in schema
        assert "sessions" in schema
        assert schema["range_bars"]["type"] == "int"
        assert schema["sessions"]["type"] == "list[int]"
