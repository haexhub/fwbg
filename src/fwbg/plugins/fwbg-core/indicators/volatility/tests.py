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

    df = pd.DataFrame(
        {
            "O": close * 0.999,
            "H": close * 1.005,
            "L": close * 0.995,
            "C": close,
            "macro_vix": 20.0 + rng.normal(0, 0.5, n).cumsum() * 0.1,
        },
        index=idx,
    )
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
        df = pd.DataFrame(
            {
                "O": close * 0.999,
                "H": close * 1.005,
                "L": close * 0.995,
                "C": close,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )

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
        df = pd.DataFrame(
            {
                "O": close * 0.998,
                "H": close * 1.01,
                "L": close * 0.99,
                "C": close,
            },
            index=pd.date_range("2022-01-03", periods=n, freq="h"),
        )
        result = self._get_indicator().compute(df)
        # Skip warm-up bars (ATR initializes to 0 for the first ~period bars)
        valid = result["vol_atr"].dropna()
        valid_warmed = valid[valid > 0]
        assert len(valid_warmed) > n // 2, (
            f"Most ATR values should be > 0, but only {len(valid_warmed)} of {len(valid)} are"
        )

    def test_atr_scales_with_price_range(self):
        """Larger H-L range -> larger ATR (comparing two constant-range datasets)."""
        n = 200
        atr_values = {}
        for scale in [0.01, 0.10]:
            close = np.full(n, 100.0)
            df = pd.DataFrame(
                {
                    "O": close,
                    "H": close * (1 + scale),
                    "L": close * (1 - scale),
                    "C": close,
                },
                index=pd.date_range("2022-01-03", periods=n, freq="h"),
            )
            result = self._get_indicator().compute(df)
            atr_values[scale] = result["vol_atr"].dropna().mean()
        assert atr_values[0.10] > atr_values[0.01], (
            f"ATR with wider range should be larger: {atr_values[0.10]:.4f} vs {atr_values[0.01]:.4f}"
        )

    def test_atr_pct_is_normalized(self):
        """vol_atr_pct = ATR / Close -> should be ~0.02 (2%) when range is +-1%."""
        n = 200
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "O": close,
                "H": close * 1.01,  # +1%
                "L": close * 0.99,  # -1%
                "C": close,
            },
            index=pd.date_range("2022-01-03", periods=n, freq="h"),
        )
        result = self._get_indicator().compute(df)
        atr_pct_cols = [c for c in result.columns if "atr_pct" in c and "rank" not in c]
        assert len(atr_pct_cols) > 0, "Expected at least one vol_atr_pct_* column"
        for col in atr_pct_cols:
            valid = result[col].dropna()
            # Skip leading warm-up zeros (ATR needs ~period bars to stabilize)
            warmed = valid[valid > 0]
            assert len(warmed) > n // 2, (
                f"{col}: expected most bars to have ATR% > 0, got only {len(warmed)}"
            )
            # Warmed-up ATR% for +-1% range (2% total) should be in (0.005, 0.05)
            assert (warmed > 0.005).all() and (warmed < 0.05).all(), (
                f"{col}: ATR% for +-1% range should be ~0.02, got range [{warmed.min():.4f}, {warmed.max():.4f}]"
            )


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
        df = pd.DataFrame(
            {
                "O": close * 0.999,
                "H": close * 1.005,
                "L": close * 0.995,
                "C": close,
            },
            index=pd.date_range("2022-01-03", periods=n, freq="h"),
        )
        result = self._get_indicator().compute(df)
        pband_cols = [c for c in result.columns if "bb_pband" in c]
        assert len(pband_cols) > 0, "Expected at least one vol_bb_pband_* column"
        for col in pband_cols:
            valid = result[col].dropna()
            in_range = valid.between(-3, 4)
            assert in_range.mean() > 0.95, f"{col}: >5% of values outside [-3, 4] range"

    def test_bb_pband_high_in_strong_uptrend(self):
        """In persistent uptrend, price above upper band -> pband > 0.5 frequently."""
        n = 300
        close = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "O": close * 0.999,
                "H": close * 1.003,
                "L": close * 0.997,
                "C": close,
            },
            index=pd.date_range("2022-01-03", periods=n, freq="h"),
        )
        result = self._get_indicator().compute(df)
        pband_cols = [c for c in result.columns if "bb_pband" in c]
        assert len(pband_cols) > 0, "Expected at least one vol_bb_pband_* column"
        for col in pband_cols:
            valid = result[col].dropna()
            pct_above = (valid > 0.5).mean()
            assert pct_above > 0.5, (
                f"{col}: in uptrend, pband should mostly be > 0.5, got {pct_above:.1%}"
            )

    def test_bb_wband_positive(self):
        """BB width band must always be > 0."""
        n = 300
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame(
            {
                "O": close * 0.999,
                "H": close * 1.005,
                "L": close * 0.995,
                "C": close,
            },
            index=pd.date_range("2022-01-03", periods=n, freq="h"),
        )
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
        df = pd.DataFrame(
            {
                "O": close * 0.998,
                "H": close * 1.01,
                "L": close * 0.99,
                "C": close,
            },
            index=pd.date_range("2022-01-03", periods=n, freq="h"),
        )
        result = self._get_indicator().compute(df)
        est_cols = [
            c
            for c in result.columns
            if any(est in c for est in ["vol_gk_", "vol_parkinson_", "vol_yz_"])
        ]
        assert len(est_cols) > 0, (
            "Expected OHLC estimator columns (vol_gk_*, vol_parkinson_*, vol_yz_*)"
        )
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
            df = pd.DataFrame(
                {
                    "O": close * (1 - scale),
                    "H": close * (1 + scale * 2),
                    "L": close * (1 - scale * 2),
                    "C": close,
                },
                index=pd.date_range("2022-01-03", periods=n, freq="h"),
            )
            results[vol_level] = self._get_indicator().compute(df)

        est_cols = [
            c
            for c in results["low"].columns
            if any(est in c for est in ["vol_gk_", "vol_parkinson_", "vol_yz_"])
            and "ratio" not in c
        ]
        assert len(est_cols) > 0, "Expected OHLC estimator columns"
        for col in est_cols:
            low_val = results["low"][col].dropna().mean()
            high_val = results["high"][col].dropna().mean()
            assert high_val > low_val, (
                f"{col}: high-vol data should give higher estimate than low-vol"
            )


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
        df = pd.DataFrame(
            {
                "O": close * 0.999,
                "H": close * 1.005,
                "L": close * 0.995,
                "C": close,
            },
            index=pd.date_range("2022-01-03", periods=n, freq="h"),
        )
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
        df = pd.DataFrame(
            {
                "O": close * 0.98,
                "H": close * 1.05,
                "L": close * 0.95,
                "C": close,
            },
            index=pd.date_range("2022-01-03", periods=n, freq="h"),
        )
        result = self._get_indicator().compute(df)
        assert "vol_compression" in result.columns, "Expected vol_compression column"
        valid = result["vol_compression"].dropna()
        compress_rate = valid.mean()
        assert compress_rate < 0.5, (
            f"High volatility should rarely compress, got rate={compress_rate:.1%}"
        )


def test_rv_on_daily_bars_uses_days_and_annualises_with_252():
    from fwbg.plugins import import_plugin_module

    ind = import_plugin_module(
        "fwbg-core", "indicators", "volatility"
    ).VolatilityIndicators()
    rng = np.random.default_rng(0)
    daily_ret = rng.normal(0, 0.01, 300)  # 1 % daily -> ~15.9 % annualised
    close = 100 * np.exp(np.cumsum(daily_ret))
    df = pd.DataFrame(
        {"O": close, "H": close * 1.005, "L": close * 0.995, "C": close},
        index=pd.date_range("2024-01-01", periods=300, freq="B"),
    )
    rv = ind.compute(df, bars_per_day=1)["vol_rv_20"]
    assert (
        rv.iloc[:11].isna().all() and rv.iloc[12:].notna().all()
    )  # 20-day window, half filled + shift
    assert rv.iloc[50:].mean() == pytest.approx(100 * 0.01 * np.sqrt(252), rel=0.2)


@pytest.mark.parametrize("bars_per_day", [0, -1, True, float("nan"), float("inf")])
def test_rv_rejects_invalid_bars_per_day(bars_per_day):
    from fwbg.plugins import import_plugin_module

    ind = import_plugin_module(
        "fwbg-core", "indicators", "volatility"
    ).VolatilityIndicators()
    df = pd.DataFrame(
        {"O": [1.0] * 30, "H": [1.1] * 30, "L": [0.9] * 30, "C": [1.0] * 30},
        index=pd.date_range("2024-01-01", periods=30, freq="B"),
    )
    with pytest.raises(ValueError, match="bars_per_day"):
        ind.compute(df, bars_per_day=bars_per_day)


# -- regression tests: bars per day, Yang-Zhang, ATR warm-up, column list -----


def _vol_indicator():
    return _vol.VolatilityIndicators()


def _ohlc_from_closes(close, index, gap=0.0, spread=0.004):
    """OHLC bars whose open sits ``gap`` away from the previous close."""
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + gap)
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    return pd.DataFrame({"O": open_, "H": high, "L": low, "C": close}, index=index)


def _reference_rv(close, window_bars, bars_per_day):
    """Annualised close-to-close vol of the window ending at the last bar (percent)."""
    log_ret = np.diff(np.log(close))
    return log_ret[-window_bars:].std(ddof=1) * np.sqrt(252 * bars_per_day) * 100


class TestRealizedVolBarsPerDay:
    def test_daily_bars_default_uses_20_bar_window_and_sqrt_252(self):
        rng = np.random.default_rng(1)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
        df = _ohlc_from_closes(close, pd.bdate_range("2024-01-01", periods=120))
        rv = _vol_indicator().compute(df)["vol_rv_20"]
        # value of row i uses bars <= i - 1 (shifted by one bar)
        assert rv.iloc[-1] == pytest.approx(_reference_rv(close[:-1], 20, 1), rel=1e-9)
        # old default (24 bars per day) needed 240 bars before the first value
        assert rv.iloc[30:].notna().all()

    def test_hourly_bars_default_uses_480_bar_window_and_sqrt_252_24(self):
        rng = np.random.default_rng(2)
        n = 1500
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        df = _ohlc_from_closes(close, pd.date_range("2024-01-01", periods=n, freq="h"))
        rv = _vol_indicator().compute(df)["vol_rv_20"]
        assert rv.iloc[-1] == pytest.approx(
            _reference_rv(close[:-1], 480, 24), rel=1e-9
        )

    def test_explicit_bars_per_day_overrides_inference(self):
        rng = np.random.default_rng(3)
        n = 400
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        df = _ohlc_from_closes(close, pd.date_range("2024-01-01", periods=n, freq="h"))
        rv = _vol_indicator().compute(df, bars_per_day=7)["vol_rv_20"]
        assert rv.iloc[-1] == pytest.approx(_reference_rv(close[:-1], 140, 7), rel=1e-9)

    @pytest.mark.parametrize(
        "index, expected",
        [
            (pd.bdate_range("2024-01-01", periods=60), 1.0),
            (pd.date_range("2024-01-01", periods=60, freq="D"), 1.0),
            (pd.date_range("2024-01-01", periods=240, freq="h"), 24.0),
            (pd.date_range("2024-01-01", periods=960, freq="15min"), 96.0),
            (pd.date_range("2024-01-01", periods=30, freq="W-FRI"), 0.2),
        ],
    )
    def test_infer_bars_per_day(self, index, expected):
        bars_per_day, _ = _vol._infer_bars_per_day(pd.DataFrame(index=index))
        assert bars_per_day == pytest.approx(expected)

    def test_infer_bars_per_day_for_a_seven_hour_session(self):
        days = pd.bdate_range("2024-01-01", periods=10)
        index = pd.DatetimeIndex(
            [d + pd.Timedelta(hours=h) for d in days for h in range(8, 15)]
        )
        bars_per_day, _ = _vol._infer_bars_per_day(pd.DataFrame(index=index))
        assert bars_per_day == 7.0


def _reference_yang_zhang(open_, high, low, close, window):
    """Yang-Zhang volatility of the window ending at the last bar, from the formula."""
    o, h, lo, c = (
        np.asarray(x, dtype=float)[-window:] for x in (open_, high, low, close)
    )
    prev_c = np.asarray(close, dtype=float)[-window - 1 : -1]
    overnight = np.log(o / prev_c)
    open_close = np.log(c / o)
    rs = np.log(h / o) * np.log(h / c) + np.log(lo / o) * np.log(lo / c)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var = overnight.var(ddof=1) + k * open_close.var(ddof=1) + (1 - k) * rs.mean()
    return np.sqrt(var)


class TestYangZhang:
    def _frame(self, n=120, gap=0.01, seed=4):
        rng = np.random.default_rng(seed)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0, gap, n))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
        index = pd.bdate_range("2024-01-01", periods=n)
        return pd.DataFrame({"O": open_, "H": high, "L": low, "C": close}, index=index)

    @pytest.mark.parametrize("window", [20, 50])
    def test_matches_hand_computed_reference_with_gaps(self, window):
        df = self._frame()
        out = _vol_indicator().compute(df, vol_est_windows=[window, 20])
        # row of the last bar holds the estimate of the window ending one bar earlier
        expected = _reference_yang_zhang(
            df["O"][:-1], df["H"][:-1], df["L"][:-1], df["C"][:-1], window
        )
        assert out[f"vol_yz_{window}"].iloc[-1] == pytest.approx(expected, rel=1e-6)

    def test_overnight_variance_is_not_counted_twice(self):
        """Without gaps the open-to-close and close-to-close terms coincide; with gaps
        the estimate must follow the open-to-close term (old code used close-to-close)."""
        df = self._frame(gap=0.02)
        out = _vol_indicator().compute(df)
        window = 20
        d = df.iloc[:-1]
        log_cc = np.log(d["C"] / d["C"].shift(1)).iloc[-window:]
        o = np.log(d["O"] / d["C"].shift(1)).iloc[-window:]
        co = np.log(d["C"] / d["O"]).iloc[-window:]
        rs = (
            np.log(d["H"] / d["O"]) * np.log(d["H"] / d["C"])
            + np.log(d["L"] / d["O"]) * np.log(d["L"] / d["C"])
        ).iloc[-window:]
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        wrong = np.sqrt(o.var() + k * log_cc.var() + (1 - k) * rs.mean())
        right = np.sqrt(o.var() + k * co.var() + (1 - k) * rs.mean())
        assert right != pytest.approx(wrong, rel=1e-3)
        assert out["vol_yz_20"].iloc[-1] == pytest.approx(right, rel=1e-6)


def _reference_atr(high, low, close, window):
    """Wilder ATR: mean of the first ``window`` true ranges, then the recursion."""
    high, low, close = (np.asarray(x, dtype=float) for x in (high, low, close))
    tr = high - low
    tr[1:] = np.maximum.reduce(
        [
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ]
    )
    atr = np.full(len(close), np.nan)
    atr[window - 1] = tr[:window].mean()
    for i in range(window, len(close)):
        atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return atr


class TestATRWarmup:
    def _df(self, n=300):
        rng = np.random.default_rng(5)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        return _ohlc_from_closes(
            close, pd.date_range("2024-01-01", periods=n, freq="h")
        )

    def test_atr_is_nan_until_first_full_window_and_matches_wilder(self):
        df = self._df()
        out = _vol_indicator().compute(df)
        expected = _reference_atr(df["H"], df["L"], df["C"], 14)
        # shifted by one bar: first value (ATR of bar 13) sits in row 14
        for col in ("vol_atr", "_atr"):
            assert out[col].iloc[:14].isna().all()
            np.testing.assert_allclose(
                out[col].iloc[14:].values, expected[13:-1], rtol=1e-9
            )

    @pytest.mark.parametrize("period", [7, 14, 21])
    def test_atr_pct_has_nan_warmup(self, period):
        df = self._df()
        out = _vol_indicator().compute(df)
        col = out[f"vol_atr_pct_{period}"]
        assert col.iloc[:period].isna().all()
        expected = _reference_atr(df["H"], df["L"], df["C"], period) / df["C"].values
        np.testing.assert_allclose(
            col.iloc[period:].values, expected[period - 1 : -1], rtol=1e-9
        )

    def test_compression_and_ranks_are_nan_during_warmup(self):
        out = _vol_indicator().compute(self._df(400))
        # ranks need compression_lookback // 2 = 50 valid values after the ATR warm-up
        assert out["vol_atr_pct_14_rank"].iloc[:60].isna().all()
        assert out["vol_compression"].iloc[:60].isna().all()
        assert out["vol_compression"].iloc[100:].isin([0.0, 1.0]).all()

    def test_series_shorter_than_window_gives_nan_not_error(self):
        df = self._df(10)
        out = _vol_indicator().compute(df)
        assert out["vol_atr"].isna().all()


class TestFeatureColumnsFollowParams:
    def test_default_columns_are_all_produced(self):
        ind = _vol_indicator()
        df = _make_ohlc_with_macro(400)
        assert set(ind.get_feature_columns()) <= set(ind.compute(df).columns)
        assert ind.get_feature_columns() == ind.get_feature_columns({})

    def test_custom_params_columns_match_compute(self):
        ind = _vol_indicator()
        params = {
            "atr_periods": [5, 14],
            "bb_period": 30,
            "vol_est_windows": [10, 20],
            "rv_window": 10,
        }
        result = ind.compute(_make_ohlc_with_macro(400), **params)
        cols = ind.get_feature_columns(params)
        assert set(cols) <= set(result.columns)
        produced = {c for c in result.columns if c.startswith("vol_")}
        assert produced == set(cols)

    def test_no_20_window_and_no_atr14_drop_dependent_columns(self):
        ind = _vol_indicator()
        params = {"atr_periods": [7, 21], "vol_est_windows": [30]}
        cols = ind.get_feature_columns(params)
        assert "vol_yz_atr_ratio" not in cols
        assert "vol_compression" not in cols
        result = ind.compute(_make_ohlc_with_macro(400), **params)
        assert set(cols) <= set(result.columns)
        assert "vol_compression" not in result.columns
