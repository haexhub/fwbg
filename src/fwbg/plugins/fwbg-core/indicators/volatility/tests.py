"""Tests for volatility indicator plugin - RV/IV features."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_vol = import_plugin_module("fwbg-core", "indicators", "volatility")
if _vol is None:
    pytest.skip("fwbg-core volatility plugin not available", allow_module_level=True)


def _make_ohlc_with_macro(n=2000):
    """Create OHLC DataFrame with macro columns for testing."""
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="h")

    df = pd.DataFrame({
        "O": close * 0.999,
        "H": close * 1.005,
        "L": close * 0.995,
        "C": close,
        "macro_vix": 20.0 + rng.normal(0, 0.5, n).cumsum() * 0.1,
    }, index=idx)
    return df


class TestRealizedVsImpliedVol:
    """Tests for RV/VIX features in volatility plugin."""

    def _get_indicator(self):
        return _vol.VolatilityIndicators()

    def test_rv_computed(self):
        """Realized vol should be computed from close-to-close returns."""
        ind = self._get_indicator()
        df = _make_ohlc_with_macro(n=2000)
        result = ind.compute(df)

        assert "vol_rv_20" in result.columns
        rv = result["vol_rv_20"].dropna()
        assert len(rv) > 0
        assert rv.mean() > 0

    def test_rv_iv_ratio_with_vix(self):
        """RV/VIX ratio should be computed when macro_vix present."""
        ind = self._get_indicator()
        df = _make_ohlc_with_macro(n=2000)
        result = ind.compute(df)

        assert "vol_rv_iv_ratio" in result.columns
        assert "vol_rv_iv_spread" in result.columns
        ratio = result["vol_rv_iv_ratio"].dropna()
        assert len(ratio) > 0
        assert ratio.mean() > 0

    def test_rv_iv_not_computed_without_vix(self):
        """RV/VIX should NOT be computed when macro_vix absent."""
        ind = self._get_indicator()
        n = 2000
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.005,
            "L": close * 0.995, "C": close,
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

        result = ind.compute(df)

        assert "vol_rv_20" in result.columns
        assert "vol_rv_iv_ratio" not in result.columns
        assert "vol_rv_iv_spread" not in result.columns

    def test_rv_no_lookahead(self):
        """RV features should be shifted by 1 bar."""
        ind = self._get_indicator()
        df = _make_ohlc_with_macro(n=2000)
        result = ind.compute(df)

        assert pd.isna(result["vol_rv_20"].iloc[0])
        assert pd.isna(result["vol_rv_iv_ratio"].iloc[0])


# ── NEW: ATR correctness tests ───────────────────────────────────────────────

class TestATR:
    """ATR (Average True Range) measures volatility as the smoothed true range."""

    def _get_indicator(self):
        return _vol.VolatilityIndicators()

    def test_atr_positive_for_non_zero_range_bars(self):
        """ATR must be > 0 when bars have H > L."""
        n = 300
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "O": close * 0.998,
            "H": close * 1.01,
            "L": close * 0.99,
            "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = self._get_indicator().compute(df)
        # Skip warm-up bars (ATR initializes to 0 for the first ~period bars)
        valid = result["vol_atr"].dropna()
        valid_warmed = valid[valid > 0]
        assert len(valid_warmed) > n // 2, \
            f"Most ATR values should be > 0, but only {len(valid_warmed)} of {len(valid)} are"

    def test_atr_scales_with_price_range(self):
        """Larger H-L range -> larger ATR (comparing two constant-range datasets)."""
        n = 200
        atr_values = {}
        for scale in [0.01, 0.10]:
            close = np.full(n, 100.0)
            df = pd.DataFrame({
                "O": close,
                "H": close * (1 + scale),
                "L": close * (1 - scale),
                "C": close,
            }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
            result = self._get_indicator().compute(df)
            atr_values[scale] = result["vol_atr"].dropna().mean()
        assert atr_values[0.10] > atr_values[0.01], \
            f"ATR with wider range should be larger: {atr_values[0.10]:.4f} vs {atr_values[0.01]:.4f}"

    def test_atr_pct_is_normalized(self):
        """vol_atr_pct = ATR / Close -> should be ~0.02 (2%) when range is +-1%."""
        n = 200
        close = np.full(n, 100.0)
        df = pd.DataFrame({
            "O": close,
            "H": close * 1.01,  # +1%
            "L": close * 0.99,  # -1%
            "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = self._get_indicator().compute(df)
        atr_pct_cols = [c for c in result.columns if "atr_pct" in c and "rank" not in c]
        assert len(atr_pct_cols) > 0, "Expected at least one vol_atr_pct_* column"
        for col in atr_pct_cols:
            valid = result[col].dropna()
            # Skip leading warm-up zeros (ATR needs ~period bars to stabilize)
            warmed = valid[valid > 0]
            assert len(warmed) > n // 2, \
                f"{col}: expected most bars to have ATR% > 0, got only {len(warmed)}"
            # Warmed-up ATR% for +-1% range (2% total) should be in (0.005, 0.05)
            assert (warmed > 0.005).all() and (warmed < 0.05).all(), \
                f"{col}: ATR% for +-1% range should be ~0.02, got range [{warmed.min():.4f}, {warmed.max():.4f}]"


# ── NEW: Bollinger Band tests ─────────────────────────────────────────────────

class TestBollingerBands:
    """Bollinger Bands = SMA +- k*sigma. Position band (pband) measures where price is in the band."""

    def _get_indicator(self):
        return _vol.VolatilityIndicators()

    def test_bb_pband_in_reasonable_range(self):
        """Bollinger %Band should be mostly in [-3, 4] for normal data."""
        n = 500
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.3)
        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.005,
            "L": close * 0.995, "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = self._get_indicator().compute(df)
        pband_cols = [c for c in result.columns if "bb_pband" in c]
        assert len(pband_cols) > 0, "Expected at least one vol_bb_pband_* column"
        for col in pband_cols:
            valid = result[col].dropna()
            in_range = valid.between(-3, 4)
            assert in_range.mean() > 0.95, \
                f"{col}: >5% of values outside [-3, 4] range"

    def test_bb_pband_high_in_strong_uptrend(self):
        """In persistent uptrend, price above upper band -> pband > 0.5 frequently."""
        n = 300
        close = np.linspace(100, 200, n)
        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.003,
            "L": close * 0.997, "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = self._get_indicator().compute(df)
        pband_cols = [c for c in result.columns if "bb_pband" in c]
        assert len(pband_cols) > 0, "Expected at least one vol_bb_pband_* column"
        for col in pband_cols:
            valid = result[col].dropna()
            pct_above = (valid > 0.5).mean()
            assert pct_above > 0.5, \
                f"{col}: in uptrend, pband should mostly be > 0.5, got {pct_above:.1%}"

    def test_bb_wband_positive(self):
        """BB width band must always be > 0."""
        n = 300
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.005,
            "L": close * 0.995, "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = self._get_indicator().compute(df)
        wband_cols = [c for c in result.columns if "bb_wband" in c]
        assert len(wband_cols) > 0, "Expected at least one vol_bb_wband_* column"
        for col in wband_cols:
            valid = result[col].dropna()
            assert (valid > 0).all(), f"{col}: BB width must be positive"


# ── NEW: OHLC Volatility Estimators ──────────────────────────────────────────

class TestOHLCVolatilityEstimators:
    """
    OHLC-based estimators are more efficient than close-only realized vol:
    - Garman-Klass (GK): uses O, H, L, C -> captures intrabar moves
    - Parkinson: uses H, L only -> simple but no overnight info
    - Yang-Zhang (YZ): combines overnight gap, open-to-close, and H-L -> most robust
    """

    def _get_indicator(self):
        return _vol.VolatilityIndicators()

    def test_all_estimators_non_negative(self):
        """All volatility estimators must be >= 0."""
        n = 300
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "O": close * 0.998, "H": close * 1.01,
            "L": close * 0.99, "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = self._get_indicator().compute(df)
        est_cols = [c for c in result.columns
                    if any(est in c for est in ["vol_gk_", "vol_parkinson_", "vol_yz_"])]
        assert len(est_cols) > 0, "Expected OHLC estimator columns (vol_gk_*, vol_parkinson_*, vol_yz_*)"
        for col in est_cols:
            valid = result[col].dropna()
            assert (valid >= 0).all(), f"{col}: volatility estimator must be >= 0"

    def test_estimators_higher_with_more_volatile_data(self):
        """Higher H-L range -> higher OHLC volatility estimates."""
        n = 300
        results = {}
        for vol_level, scale in [("low", 0.001), ("high", 0.02)]:
            np.random.seed(42)
            close = 100 + np.cumsum(np.random.randn(n) * scale * 10)
            df = pd.DataFrame({
                "O": close * (1 - scale), "H": close * (1 + scale * 2),
                "L": close * (1 - scale * 2), "C": close,
            }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
            results[vol_level] = self._get_indicator().compute(df)

        est_cols = [c for c in results["low"].columns
                    if any(est in c for est in ["vol_gk_", "vol_parkinson_", "vol_yz_"])
                    and "ratio" not in c]
        assert len(est_cols) > 0, "Expected OHLC estimator columns"
        for col in est_cols:
            low_val = results["low"][col].dropna().mean()
            high_val = results["high"][col].dropna().mean()
            assert high_val > low_val, \
                f"{col}: high-vol data should give higher estimate than low-vol"


# ── NEW: Compression detection ────────────────────────────────────────────────

class TestVolatilityCompression:
    """
    Volatility compression: ATR percentile < 20th AND BB width < 20th percentile.
    Compression often precedes large breakout moves.
    """

    def _get_indicator(self):
        return _vol.VolatilityIndicators()

    def test_compression_flag_is_binary(self):
        """vol_compression must be 0 or 1."""
        n = 500
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.3)
        df = pd.DataFrame({
            "O": close * 0.999, "H": close * 1.005,
            "L": close * 0.995, "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = self._get_indicator().compute(df)
        assert "vol_compression" in result.columns, "Expected vol_compression column"
        valid = result["vol_compression"].dropna()
        assert valid.isin([0, 1]).all(), "vol_compression must be binary 0/1"

    def test_compression_rate_low_in_high_volatility(self):
        """During highly volatile periods, compression should rarely trigger."""
        n = 500
        np.random.seed(42)
        # High volatility: +-5% bars
        close = 100 + np.cumsum(np.random.randn(n) * 2.0)
        df = pd.DataFrame({
            "O": close * 0.98, "H": close * 1.05,
            "L": close * 0.95, "C": close,
        }, index=pd.date_range("2022-01-03", periods=n, freq="h"))
        result = self._get_indicator().compute(df)
        assert "vol_compression" in result.columns, "Expected vol_compression column"
        valid = result["vol_compression"].dropna()
        compress_rate = valid.mean()
        assert compress_rate < 0.5, \
            f"High volatility should rarely compress, got rate={compress_rate:.1%}"
