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


# ── NEW: MACD tests ───────────────────────────────────────────────────────────

class TestMACD:
    """
    MACD = 12-EMA - 26-EMA (trend following momentum oscillator).
    Signal line = 9-EMA of MACD. Histogram = MACD - Signal.

    Note: the plugin normalises all MACD values by dividing by Close price,
    so columns are named trend_macd_line, trend_macd, trend_macd_signal,
    trend_macd_above_zero, trend_macd_hist_flip.
    """

    @staticmethod
    def _indicator():
        from fwbg.plugins import import_plugin_module
        _trend = import_plugin_module("fwbg-core", "indicators", "trend")
        return _trend.TrendIndicators()

    def test_macd_line_positive_in_sustained_uptrend(self):
        """12-EMA > 26-EMA in uptrend -> MACD line should be positive."""
        n = 300
        close = np.linspace(100, 200, n)
        df = create_ohlc(close)
        result = self._indicator().compute(df)
        col = "trend_macd_line"
        assert col in result.columns, f"Expected {col} column"
        valid = result[col].dropna()
        assert len(valid) >= 50
        assert (valid.iloc[-50:] > 0).all(), \
            f"{col}: should be positive in uptrend, got min={valid.iloc[-50:].min():.6f}"

    def test_macd_line_negative_in_sustained_downtrend(self):
        """12-EMA < 26-EMA in downtrend -> MACD line should be negative."""
        n = 300
        close = np.linspace(200, 100, n)
        df = create_ohlc(close)
        result = self._indicator().compute(df)
        col = "trend_macd_line"
        assert col in result.columns, f"Expected {col} column"
        valid = result[col].dropna()
        assert len(valid) >= 50
        assert (valid.iloc[-50:] < 0).all(), \
            f"{col}: should be negative in downtrend, got max={valid.iloc[-50:].max():.6f}"

    def test_macd_hist_flip_fires_at_trend_reversal(self):
        """MACD histogram sign change should produce a flip signal at reversal."""
        _n = 400
        close = np.concatenate([np.linspace(100, 80, 200), np.linspace(80, 120, 200)])
        df = create_ohlc(close)
        result = self._indicator().compute(df)
        col = "trend_macd_hist_flip"
        assert col in result.columns, f"Expected {col} column"
        flips = result[col].dropna()
        assert flips.sum() > 0, f"{col}: should have at least one flip at reversal"

    def test_macd_above_zero_positive_in_uptrend(self):
        """In sustained uptrend, trend_macd_above_zero (sign of MACD line) should be 1."""
        n = 300
        close = np.linspace(100, 200, n)
        df = create_ohlc(close)
        result = self._indicator().compute(df)
        col = "trend_macd_above_zero"
        assert col in result.columns, f"Expected {col} column"
        valid = result[col].dropna()
        assert len(valid) >= 50
        assert (valid.iloc[-50:] == 1).all(), \
            f"{col}: should be 1 in uptrend"


# ── NEW: Aroon tests ──────────────────────────────────────────────────────────

class TestAroon:
    """
    Aroon Up/Down measures time since the last N-period high/low.
    Range: 0-100. High Aroon Up = recent new high = uptrend.
    """

    @staticmethod
    def _indicator():
        from fwbg.plugins import import_plugin_module
        _trend = import_plugin_module("fwbg-core", "indicators", "trend")
        return _trend.TrendIndicators()

    def test_aroon_up_near_100_in_new_high_trend(self):
        """Continuous new highs -> Aroon Up approaches 100."""
        n = 300
        close = np.linspace(100, 200, n)  # continuous new highs
        df = create_ohlc(close)
        result = self._indicator().compute(df)
        col = "trend_aroon_up"
        assert col in result.columns, f"Expected {col} column"
        valid = result[col].dropna()
        assert len(valid) >= 50
        assert valid.iloc[-50:].mean() > 80, \
            f"{col}: Aroon Up should be near 100 in continuous new highs trend, got {valid.iloc[-50:].mean():.1f}"

    def test_aroon_down_near_100_in_new_low_trend(self):
        """Continuous new lows -> Aroon Down approaches 100."""
        n = 300
        close = np.linspace(200, 100, n)
        df = create_ohlc(close)
        result = self._indicator().compute(df)
        col = "trend_aroon_down"
        assert col in result.columns, f"Expected {col} column"
        valid = result[col].dropna()
        assert len(valid) >= 50
        assert valid.iloc[-50:].mean() > 80, \
            f"{col}: Aroon Down should be near 100 in downtrend, got {valid.iloc[-50:].mean():.1f}"

    def test_aroon_values_bounded_0_to_100(self):
        """Aroon Up and Down must always be in [0, 100]."""
        n = 300
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = create_ohlc(close)
        result = self._indicator().compute(df)
        aroon_cols = [c for c in result.columns if "aroon" in c]
        assert len(aroon_cols) > 0, "Expected trend_aroon_up / trend_aroon_down columns"
        for col in aroon_cols:
            valid = result[col].dropna()
            assert (valid >= 0).all() and (valid <= 100).all(), \
                f"{col}: Aroon must be in [0,100], got [{valid.min():.1f}, {valid.max():.1f}]"


# ── NEW: Supertrend tests ─────────────────────────────────────────────────────

class TestSupertrend:
    """
    Supertrend is an ATR-based trend-following indicator.
    Direction: +1 (uptrend / price above band) or -1 (downtrend / price below band).
    Flip: fires when direction changes (trend_supertrend_flip = 1.0).
    """

    @staticmethod
    def _indicator():
        from fwbg.plugins import import_plugin_module
        _trend = import_plugin_module("fwbg-core", "indicators", "trend")
        return _trend.TrendIndicators()

    def test_supertrend_is_binary_plus_minus_1(self):
        """Supertrend direction must be exactly +1 or -1."""
        n = 300
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = create_ohlc(close)
        result = self._indicator().compute(df)
        col = "trend_supertrend"
        assert col in result.columns, f"Expected {col} column"
        valid = result[col].dropna()
        assert valid.isin([1, -1]).all(), \
            f"{col}: supertrend must be +1 or -1, got unique values: {sorted(valid.unique())}"

    def test_supertrend_positive_in_sustained_uptrend(self):
        """In a strong uptrend, supertrend should be +1 most of the time."""
        n = 300
        close = np.linspace(100, 200, n)
        df = create_ohlc(close)
        result = self._indicator().compute(df)
        col = "trend_supertrend"
        assert col in result.columns, f"Expected {col} column"
        valid = result[col].dropna()
        assert len(valid) >= 100
        pct_positive = (valid.iloc[-100:] == 1).mean()
        assert pct_positive > 0.6, \
            f"{col}: should mostly be +1 in uptrend, got {pct_positive:.1%}"

    def test_supertrend_flip_fires_at_direction_change(self):
        """After a trend reversal (up -> down), trend_supertrend_flip should fire."""
        _n = 400
        close = np.concatenate([np.linspace(100, 200, 200), np.linspace(200, 100, 200)])
        df = create_ohlc(close)
        result = self._indicator().compute(df)
        col = "trend_supertrend_flip"
        assert col in result.columns, f"Expected {col} column"
        flips = result[col].dropna()
        assert flips.sum() > 0, f"{col}: should have at least one flip at reversal"


# ── NEW: CCI tests ────────────────────────────────────────────────────────────

class TestCCI:
    """
    Commodity Channel Index: measures deviation of Typical Price from its SMA.
    Overbought: CCI > 100. Oversold: CCI < -100. Neutral: near 0.
    Plugin columns: trend_cci_14, trend_cci_20.
    """

    @staticmethod
    def _indicator():
        from fwbg.plugins import import_plugin_module
        _trend = import_plugin_module("fwbg-core", "indicators", "trend")
        return _trend.TrendIndicators()

    def test_cci_high_in_overbought_uptrend(self):
        """In a sustained uptrend, CCI should frequently exceed 50."""
        n = 300
        close = np.linspace(100, 200, n)
        df = create_ohlc(close)
        result = self._indicator().compute(df)
        cci_cols = [c for c in result.columns if "cci" in c]
        assert len(cci_cols) > 0, "Expected at least one trend_cci_* column"
        for col in cci_cols:
            valid = result[col].dropna()
            assert len(valid) >= 50
            assert valid.iloc[-50:].mean() > 50, \
                f"{col}: CCI should be high (>50) in sustained uptrend, got mean={valid.iloc[-50:].mean():.1f}"

    def test_cci_low_in_oversold_downtrend(self):
        """In a sustained downtrend, CCI should be below -50."""
        n = 300
        close = np.linspace(200, 100, n)
        df = create_ohlc(close)
        result = self._indicator().compute(df)
        cci_cols = [c for c in result.columns if "cci" in c]
        assert len(cci_cols) > 0, "Expected at least one trend_cci_* column"
        for col in cci_cols:
            valid = result[col].dropna()
            assert len(valid) >= 50
            assert valid.iloc[-50:].mean() < -50, \
                f"{col}: CCI should be low (<-50) in sustained downtrend, got mean={valid.iloc[-50:].mean():.1f}"
