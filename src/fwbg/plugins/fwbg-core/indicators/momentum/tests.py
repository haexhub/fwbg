"""
Tests for MomentumIndicators plugin.

These tests verify:
- RSI computation and bounds
- Stochastic oscillator behavior
- ROC calculations
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
    """Create MomentumIndicators instance."""
    from fwbg.plugins import import_plugin_module
    _momentum = import_plugin_module("fwbg-core", "indicators", "momentum")
    return _momentum.MomentumIndicators()


@pytest.fixture
def sample_ohlc():
    """Create sample OHLC data."""
    np.random.seed(42)
    n = 200
    returns = np.random.randn(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    return create_ohlc(close)


class TestRSI:
    """Tests for RSI indicator."""

    def test_rsi_high_after_gains(self, indicator):
        """RSI should be high after continuous gains."""
        n = 100
        close = 100 * np.cumprod(1 + np.full(n, 0.01))
        df = create_ohlc(close)
        result = indicator.compute(df)

        rsi = result["mom_rsi_14"].dropna()
        assert rsi.iloc[-1] > 70, f"Expected RSI > 70 after gains, got {rsi.iloc[-1]:.1f}"

    def test_rsi_low_after_losses(self, indicator):
        """RSI should be low after continuous losses."""
        n = 100
        close = 100 * np.cumprod(1 - np.full(n, 0.01))
        df = create_ohlc(close)
        result = indicator.compute(df)

        rsi = result["mom_rsi_14"].dropna()
        assert rsi.iloc[-1] < 30, f"Expected RSI < 30 after losses, got {rsi.iloc[-1]:.1f}"

    def test_rsi_around_50_in_sideways(self, indicator):
        """RSI should be around 50 in balanced market."""
        np.random.seed(42)
        n = 300
        returns = np.random.randn(n) * 0.01
        close = 100 * np.exp(np.cumsum(returns))
        df = create_ohlc(close)
        result = indicator.compute(df)

        rsi = result["mom_rsi_14"].dropna()
        assert 40 < rsi.mean() < 60, f"Expected RSI ~50 in sideways, got {rsi.mean():.1f}"


class TestStochastic:
    """Tests for Stochastic oscillator."""

    def test_stochastic_high_at_period_high(self, indicator):
        """Stochastic should be high when close is at period high."""
        n = 50
        close = np.concatenate([np.full(35, 100), np.linspace(100, 120, 15)])
        high = close * 1.001
        low = close * 0.999

        df = pd.DataFrame({
            'O': close, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        result = indicator.compute(df)

        stoch_k = result["mom_stoch_k_14"].dropna()
        assert stoch_k.iloc[-1] > 80, f"Expected Stoch > 80 at high, got {stoch_k.iloc[-1]:.1f}"

    def test_stochastic_low_at_period_low(self, indicator):
        """Stochastic should be low when close is at period low."""
        n = 50
        close = np.concatenate([np.full(35, 120), np.linspace(120, 100, 15)])
        high = close * 1.001
        low = close * 0.999

        df = pd.DataFrame({
            'O': close, 'H': high, 'L': low, 'C': close,
        }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
        result = indicator.compute(df)

        stoch_k = result["mom_stoch_k_14"].dropna()
        assert stoch_k.iloc[-1] < 20, f"Expected Stoch < 20 at low, got {stoch_k.iloc[-1]:.1f}"


class TestROC:
    """Tests for Rate of Change."""

    def test_roc_equals_percentage_change(self, indicator):
        """ROC should equal percentage change."""
        n = 100
        close = np.linspace(100, 150, n)
        df = create_ohlc(close)
        result = indicator.compute(df)

        roc_10 = result["mom_roc_10"].dropna()
        expected = (close[10:] - close[:-10]) / close[:-10] * 100

        # Compare (allowing for shift)
        np.testing.assert_array_almost_equal(
            roc_10.iloc[-50:].values,
            expected[-51:-1],
            decimal=5
        )


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
        """All feature columns should have mom_ prefix."""
        _ = indicator.compute(sample_ohlc)
        columns = indicator.get_feature_columns()

        for col in columns:
            assert col.startswith("mom_"), f"Feature {col} missing mom_ prefix"


class TestPluginAttributes:
    """Tests for plugin class attributes."""

    def test_name_attribute(self, indicator):
        """Plugin should have name attribute."""
        assert indicator.name == "momentum"

    def test_version_attribute(self, indicator):
        """Plugin should have version attribute."""
        assert hasattr(indicator, "version")
        assert isinstance(indicator.version, str)

    def test_validate_returns_true(self, indicator):
        """validate() should return True."""
        assert indicator.validate() is True
