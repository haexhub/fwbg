"""Tests for ADX Indicator plugin."""

import numpy as np
import pandas as pd
import pytest


def create_ohlc(close, high_factor=1.005, low_factor=0.995):
    n = len(close)
    return pd.DataFrame(
        {
            "O": close * 0.999,
            "H": close * high_factor,
            "L": close * low_factor,
            "C": close,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


@pytest.fixture
def indicator():
    from fwbg.plugins import import_plugin_module

    mod = import_plugin_module("fwbg-core", "indicators", "adx")
    return mod.ADXIndicator()


class TestADX:
    def test_adx_high_in_strong_trend(self, indicator):
        n = 200
        close = 100 * np.cumprod(1 + np.full(n, 0.01))
        df = create_ohlc(close)
        result = indicator.compute(df)
        adx = result["adx_14"].dropna()
        assert adx.mean() > 25, f"Expected ADX > 25 in trend, got {adx.mean():.1f}"

    def test_adx_lower_in_sideways(self, indicator):
        n = 200
        trending = 100 * np.cumprod(1 + np.full(n, 0.01))
        np.random.seed(42)
        sideways = 100 + np.random.randn(n) * 0.5

        result_trend = indicator.compute(create_ohlc(trending))
        result_sideways = indicator.compute(create_ohlc(sideways))

        adx_trend = result_trend["adx_14"].dropna().mean()
        adx_sideways = result_sideways["adx_14"].dropna().mean()
        assert adx_trend > adx_sideways

    def test_feature_columns(self, indicator):
        columns = indicator.get_feature_columns()
        assert "adx_7" in columns
        assert "adx_14" in columns
        assert "adx_21" in columns

    def test_name(self, indicator):
        assert indicator.name == "adx"


def _random_ohlc(n=300, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(size=n))
    high = close + rng.random(n)
    low = close - rng.random(n)
    return pd.DataFrame(
        {"O": close, "H": high, "L": low, "C": close},
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


def reference_adx(high, low, close, n):
    """Wilder ADX written out bar by bar, independent of the ta library."""
    h, lo, c = (np.asarray(x, dtype=float) for x in (high, low, close))
    size = len(c)
    tr = np.full(size, np.nan)
    pdm = np.zeros(size)
    ndm = np.zeros(size)
    for i in range(1, size):
        tr[i] = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
        up, down = h[i] - h[i - 1], lo[i - 1] - lo[i]
        pdm[i] = up if (up > down and up > 0) else 0.0
        ndm[i] = down if (down > up and down > 0) else 0.0

    def dx(trs, p, m):
        pdi, ndi = 100 * p / trs, 100 * m / trs
        return 100 * abs(pdi - ndi) / (pdi + ndi) if pdi + ndi else 0.0

    dxs = np.full(size, np.nan)
    adx = np.full(size, np.nan)
    trs, p, m = tr[1 : n + 1].sum(), pdm[1 : n + 1].sum(), ndm[1 : n + 1].sum()
    dxs[n] = dx(trs, p, m)
    for i in range(n + 1, size):
        trs = trs - trs / n + tr[i]
        p = p - p / n + pdm[i]
        m = m - m / n + ndm[i]
        dxs[i] = dx(trs, p, m)
    adx[2 * n - 1] = dxs[n : 2 * n].mean()
    for i in range(2 * n, size):
        adx[i] = (adx[i - 1] * (n - 1) + dxs[i]) / n
    return adx


class TestADXValues:
    @pytest.mark.parametrize("period", [7, 14, 21])
    def test_matches_wilder_reference_after_one_bar_shift(self, indicator, period):
        df = _random_ohlc()
        result = indicator.compute(df, periods=[period])
        expected = pd.Series(
            reference_adx(df["H"], df["L"], df["C"], period), index=df.index
        ).shift(1)
        pd.testing.assert_series_equal(
            result[f"adx_{period}"], expected, check_names=False, atol=1e-9, rtol=0
        )

    def test_warmup_is_nan_not_zero(self, indicator):
        # The first value needs 2 * period bars; the shift adds one more NaN row.
        result = indicator.compute(_random_ohlc(), periods=[14])
        col = result["adx_14"]
        assert col.iloc[: 2 * 14].isna().all()
        assert col.iloc[2 * 14 :].notna().all()
        assert (col.dropna() > 0).all()

    def test_input_shorter_than_two_periods_gives_nan_columns(self, indicator):
        df = _random_ohlc(n=20)
        result = indicator.compute(df, periods=[7, 14, 21])
        assert result["adx_21"].isna().all()
        assert result["adx_14"].isna().all()  # 20 rows < 28
        assert result["adx_7"].iloc[14:].notna().all()  # 20 rows >= 14

    def test_nan_gap_does_not_poison_later_values(self, indicator):
        df = _random_ohlc()
        gapped = df.copy()
        gapped.iloc[100, gapped.columns.get_indexer(["H", "L", "C"])] = np.nan
        result = indicator.compute(gapped, periods=[14])["adx_14"]
        clean = indicator.compute(df, periods=[14])["adx_14"]
        assert np.isnan(result.iloc[101])  # the gap bar itself (shifted by one)
        assert result.iloc[102:].notna().all()
        # The bar after the gap is bridged, so values differ from the clean run,
        # but they stay in the same range instead of turning NaN.
        assert abs(result.iloc[-1] - clean.iloc[-1]) < 10

    def test_no_lookahead_truncation_invariance(self, indicator):
        df = _random_ohlc(n=90)
        full = indicator.compute(df, periods=[7])["adx_7"]
        for cut in range(20, 90):
            part = indicator.compute(df.iloc[:cut], periods=[7])["adx_7"]
            pd.testing.assert_series_equal(part, full.iloc[:cut], atol=1e-9, rtol=0)
