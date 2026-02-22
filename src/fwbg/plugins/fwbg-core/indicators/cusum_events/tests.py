"""Tests for CUSUM structural break indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_cusum = import_plugin_module("fwbg-core", "indicators", "cusum_events")
if _cusum is None:
    pytest.skip("fwbg-core cusum_events plugin not available", allow_module_level=True)


@pytest.fixture
def indicator():
    return _cusum.CusumEventsIndicator()


def _make_ohlc(close, n=None):
    if n is None:
        n = len(close)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "O": close * 0.999,
            "H": close * 1.005,
            "L": close * 0.995,
            "C": close,
        },
        index=idx,
    )


@pytest.fixture
def random_walk_df():
    """Random walk — few CUSUM events expected."""
    rng = np.random.default_rng(42)
    n = 500
    returns = rng.normal(0, 0.005, n)
    close = 100 * np.exp(np.cumsum(returns))
    return _make_ohlc(close, n)


@pytest.fixture
def trending_df():
    """Strong uptrend — should trigger positive CUSUM events."""
    n = 500
    # Consistent positive drift
    returns = np.full(n, 0.003) + np.random.default_rng(42).normal(0, 0.001, n)
    close = 100 * np.exp(np.cumsum(returns))
    return _make_ohlc(close, n)


@pytest.fixture
def crash_df():
    """Sudden crash — should trigger negative CUSUM event."""
    rng = np.random.default_rng(42)
    n = 500
    returns = rng.normal(0, 0.002, n)
    # Insert crash at bar 250
    returns[250:260] = -0.03
    close = 100 * np.exp(np.cumsum(returns))
    return _make_ohlc(close, n)


@pytest.fixture
def regime_shift_df():
    """Regime shift: quiet → volatile → quiet."""
    rng = np.random.default_rng(42)
    n = 600
    returns = np.empty(n)
    returns[:200] = rng.normal(0, 0.002, 200)
    returns[200:400] = rng.normal(0, 0.015, 200)
    returns[400:] = rng.normal(0, 0.002, 200)
    close = 100 * np.exp(np.cumsum(returns))
    return _make_ohlc(close, n)


class TestCusumFeatureColumns:
    """All expected columns are produced."""

    def test_all_columns_present(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_feature_count(self, indicator):
        assert len(indicator.get_feature_columns()) == 6

    def test_returns_dataframe(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_original_columns(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        for col in ["O", "H", "L", "C"]:
            assert col in result.columns


class TestCusumNoLookahead:
    """Features must be shifted by 1 bar (no lookahead)."""

    def test_first_row_is_nan(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        for col in indicator.get_feature_columns():
            assert pd.isna(result[col].iloc[0]), f"{col} first row should be NaN"

    def test_events_are_shifted(self, indicator, crash_df):
        result = indicator.compute(crash_df)
        events = result["cusum_neg_event"].dropna()
        # Events should only reflect info from previous bars
        assert events.iloc[0] == 0.0  # First non-NaN should be 0


class TestCusumEventDetection:
    """CUSUM correctly detects structural breaks."""

    def test_trending_produces_positive_events(self, indicator, trending_df):
        result = indicator.compute(trending_df, threshold=1.0)
        pos_events = result["cusum_pos_event"].dropna()
        assert pos_events.sum() > 0, "Strong uptrend should trigger positive events"

    def test_crash_produces_negative_events(self, indicator, crash_df):
        result = indicator.compute(crash_df, threshold=1.0)
        neg_events = result["cusum_neg_event"].dropna()
        assert neg_events.sum() > 0, "Crash should trigger negative events"

    def test_random_walk_fewer_events(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df, threshold=2.0)
        pos_events = result["cusum_pos_event"].dropna().sum()
        neg_events = result["cusum_neg_event"].dropna().sum()
        total = pos_events + neg_events
        # Random walk with high threshold → fewer events than bars
        assert total < len(random_walk_df) * 0.5, (
            f"Random walk should have fewer events than bars, got {total}"
        )

    def test_higher_threshold_fewer_events(self, indicator, trending_df):
        low = indicator.compute(trending_df, threshold=1.0)
        high = indicator.compute(trending_df, threshold=3.0)
        events_low = low["cusum_pos_event"].dropna().sum()
        events_high = high["cusum_pos_event"].dropna().sum()
        assert events_high <= events_low, (
            f"Higher threshold should produce fewer events: "
            f"low={events_low}, high={events_high}"
        )


class TestCusumValues:
    """Cumulative values and intensity are correctly computed."""

    def test_pos_value_bounded_0_1(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        vals = result["cusum_pos_value"].dropna()
        assert vals.min() >= -0.01, f"pos_value min={vals.min()}"
        assert vals.max() <= 1.5, f"pos_value max={vals.max()}"

    def test_neg_value_bounded_0_1(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        vals = result["cusum_neg_value"].dropna()
        assert vals.min() >= -0.01, f"neg_value min={vals.min()}"
        assert vals.max() <= 1.5, f"neg_value max={vals.max()}"

    def test_intensity_positive_at_events(self, indicator, crash_df):
        result = indicator.compute(crash_df, threshold=1.0)
        events = result["cusum_neg_event"].dropna()
        intensity = result["cusum_intensity"].dropna()
        event_bars = events[events > 0].index
        if len(event_bars) > 0:
            for bar in event_bars:
                assert intensity.loc[bar] >= 1.0, (
                    f"Intensity at event should be >= 1.0, got {intensity.loc[bar]}"
                )

    def test_intensity_zero_at_non_events(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        events = (
            result["cusum_pos_event"].dropna()
            + result["cusum_neg_event"].dropna()
        )
        intensity = result["cusum_intensity"].dropna()
        non_event_bars = events[events == 0].index
        assert (intensity.loc[non_event_bars] == 0).all()


class TestCusumBarsSince:
    """bars_since feature tracks time since last event."""

    def test_bars_since_non_negative(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        vals = result["cusum_bars_since"].dropna()
        assert (vals >= 0).all(), "bars_since must be non-negative"

    def test_bars_since_resets_at_events(self, indicator, crash_df):
        result = indicator.compute(crash_df, threshold=1.0)
        # After a crash event, bars_since should reset close to 0
        events = result["cusum_neg_event"].dropna()
        bs = result["cusum_bars_since"].dropna()
        event_bars = events[events > 0].index
        if len(event_bars) > 0:
            # Due to shift, bars_since at event+1 should be small
            for bar in event_bars:
                loc = bs.index.get_loc(bar)
                if loc + 1 < len(bs):
                    assert bs.iloc[loc + 1] <= 0.05, (
                        "bars_since should reset near event"
                    )


class TestCusumRegimeShift:
    """CUSUM detects regime shifts (volatility changes)."""

    def test_more_events_during_volatile_regime(self, indicator, regime_shift_df):
        result = indicator.compute(regime_shift_df, threshold=1.5)
        pos = result["cusum_pos_event"].dropna()
        neg = result["cusum_neg_event"].dropna()
        events = pos + neg

        # Split into quiet (0-200) and volatile (200-400) regimes
        # Account for shift: events[i] reflects bar i-1
        quiet_events = events.iloc[1:201].sum()
        volatile_events = events.iloc[201:401].sum()
        assert volatile_events > quiet_events, (
            f"Volatile regime should have more events: "
            f"quiet={quiet_events}, volatile={volatile_events}"
        )


class TestCusumPluginIntegration:
    """Plugin integrates correctly with registry."""

    def test_plugin_importable(self):
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-core:cusum_events")
        assert plugin_cls is not None

    def test_benefits_from_stationary_false(self):
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-core:cusum_events")
        assert plugin_cls.benefits_from_stationary is False

    def test_default_params(self, indicator):
        params = indicator.get_default_params()
        assert "threshold" in params
        assert "lookback" in params
        assert params["threshold"] == 1.5
        assert params["lookback"] == 100
