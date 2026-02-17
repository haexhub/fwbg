"""
Tests for TrendIndicators plugin.

These tests verify:
- ADX computation and expected behavior
- EMA distance calculations
- Efficiency ratio bounds
- Feature column generation
"""
import numpy as np
import pandas as pd
import pytest


def create_ohlc(close, high_factor=1.005, low_factor=0.995, n_bars=None):
    """Create OHLC DataFrame from close prices."""
    if n_bars is None:
        n_bars = len(close)

    df = pd.DataFrame({
        'O': close * 0.999,
        'H': close * high_factor,
        'L': close * low_factor,
        'C': close,
    }, index=pd.date_range('2024-01-01', periods=n_bars, freq='h'))
    return df


@pytest.fixture
def indicator():
    """Create TrendIndicators instance."""
    from fwbg.plugins import import_plugin_module
    _trend = import_plugin_module("fwbg-core", "indicators", "trend")
    return _trend.TrendIndicators()


@pytest.fixture
def sample_ohlc():
    """Create sample OHLC data."""
    np.random.seed(42)
    n = 200
    returns = np.random.randn(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    return create_ohlc(close)


@pytest.fixture
def trending_ohlc():
    """Create trending OHLC data."""
    n = 200
    close = 100 * np.cumprod(1 + np.full(n, 0.01))
    return create_ohlc(close)


@pytest.fixture
def sideways_ohlc():
    """Create sideways OHLC data."""
    np.random.seed(42)
    n = 200
    close = 100 + np.random.randn(n) * 0.5
    return create_ohlc(close)


class TestADX:
    """Tests for ADX indicator."""

    def test_adx_high_in_strong_trend(self, indicator, trending_ohlc):
        """ADX should be high in strong trend."""
        result = indicator.compute(trending_ohlc)
        adx = result["trend_adx_14"].dropna()
        assert adx.mean() > 25, f"Expected ADX > 25 in trend, got {adx.mean():.1f}"

    def test_adx_lower_in_sideways(self, indicator, trending_ohlc, sideways_ohlc):
        """ADX should be lower in sideways than in trend."""
        result_trend = indicator.compute(trending_ohlc)
        result_sideways = indicator.compute(sideways_ohlc)

        adx_trend = result_trend["trend_adx_14"].dropna().mean()
        adx_sideways = result_sideways["trend_adx_14"].dropna().mean()

        assert adx_trend > adx_sideways, \
            f"ADX in trend ({adx_trend:.1f}) should be > sideways ({adx_sideways:.1f})"


class TestEMADistance:
    """Tests for EMA distance indicator."""

    def test_positive_distance_above_ema(self, indicator):
        """Price above EMA should have positive distance."""
        n = 200
        close = 100 * np.cumprod(1 + np.full(n, 0.005))
        df = create_ohlc(close)
        result = indicator.compute(df)

        ema_dist = result["trend_ema_dist_21"].dropna()
        assert ema_dist.iloc[-50:].mean() > 0, "Price above EMA should have positive distance"

    def test_negative_distance_below_ema(self, indicator):
        """Price below EMA should have negative distance."""
        n = 200
        close = 100 * np.cumprod(1 - np.full(n, 0.005))
        df = create_ohlc(close)
        result = indicator.compute(df)

        ema_dist = result["trend_ema_dist_21"].dropna()
        assert ema_dist.iloc[-50:].mean() < 0, "Price below EMA should have negative distance"


class TestEfficiencyRatio:
    """Tests for Kaufman's Efficiency Ratio."""

    def test_high_er_in_linear_trend(self, indicator):
        """ER should be close to 1 in linear trend."""
        n = 200
        close = np.linspace(100, 150, n)
        df = create_ohlc(close)
        result = indicator.compute(df)

        er = result["trend_er_20"].dropna()
        assert er.mean() > 0.8, f"Expected ER > 0.8 in linear trend, got {er.mean():.2f}"

    def test_low_er_in_noisy_sideways(self, indicator):
        """ER should be low in noisy sideways."""
        np.random.seed(42)
        n = 200
        close = 100 + np.random.randn(n) * 3
        df = create_ohlc(close)
        result = indicator.compute(df)

        er = result["trend_er_20"].dropna()
        assert er.mean() < 0.3, f"Expected ER < 0.3 in noisy sideways, got {er.mean():.2f}"


class TestFeatureColumns:
    """Tests for feature column generation."""

    def test_get_feature_columns_returns_list(self, indicator, sample_ohlc):
        """get_feature_columns should return list of column names."""
        result = indicator.compute(sample_ohlc)
        columns = indicator.get_feature_columns()

        assert isinstance(columns, list)
        assert len(columns) > 0
        assert all(c in result.columns for c in columns)

    def test_all_features_have_prefix(self, indicator, sample_ohlc):
        """All feature columns should have trend_ prefix."""
        _ = indicator.compute(sample_ohlc)
        columns = indicator.get_feature_columns()

        for col in columns:
            assert col.startswith("trend_"), f"Feature {col} missing trend_ prefix"


class TestValidate:
    """Tests for validate method."""

    def test_validate_returns_true(self, indicator):
        """validate() should return True for properly configured plugin."""
        assert indicator.validate() is True


class TestPluginAttributes:
    """Tests for plugin class attributes."""

    def test_name_attribute(self, indicator):
        """Plugin should have name attribute."""
        assert indicator.name == "trend"

    def test_version_attribute(self, indicator):
        """Plugin should have version attribute."""
        assert hasattr(indicator, "version")
        assert isinstance(indicator.version, str)

    def test_phase_attribute(self, indicator):
        """Plugin should have phase attribute."""
        from fwbg_sdk import PluginPhase
        assert indicator.phase == PluginPhase.INDICATORS
