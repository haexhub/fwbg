"""Tests for Supertrend Indicator plugin (standard + MTF)."""

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

    mod = import_plugin_module("fwbg-core", "indicators", "supertrend")
    return mod.SupertrendIndicator()


@pytest.fixture
def mtf_indicator():
    from fwbg.plugins import import_plugin_module

    mod = import_plugin_module("fwbg-core", "indicators", "supertrend")
    return mod.SupertrendMTFIndicator()


class TestSupertrend:
    def test_direction_binary(self, indicator):
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(300) * 0.5)
        result = indicator.compute(create_ohlc(close))
        valid = result["st_direction"].dropna()
        assert valid.isin([1, -1]).all()

    def test_positive_in_uptrend(self, indicator):
        close = np.linspace(100, 200, 300)
        result = indicator.compute(create_ohlc(close))
        valid = result["st_direction"].dropna()
        pct_positive = (valid.iloc[-100:] == 1).mean()
        assert pct_positive > 0.6

    def test_flip_at_reversal(self, indicator):
        close = np.concatenate([np.linspace(100, 200, 200), np.linspace(200, 100, 200)])
        result = indicator.compute(create_ohlc(close))
        flips = result["st_flip"].dropna()
        assert flips.sum() > 0

    def test_name(self, indicator):
        assert indicator.name == "supertrend"


class TestSupertrendMTF:
    def test_direction_exists(self, mtf_indicator):
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(300) * 0.5)
        result = mtf_indicator.compute(create_ohlc(close))
        assert "st_d1_direction" in result.columns

    def test_name(self, mtf_indicator):
        assert mtf_indicator.name == "supertrend_mtf"


def _random_ohlc(n, seed, freq="h"):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(size=n))
    high = close + rng.random(n) * 1.5
    low = close - rng.random(n) * 1.5
    return pd.DataFrame(
        {"O": close, "H": high, "L": low, "C": close},
        index=pd.date_range("2024-01-01", periods=n, freq=freq),
    )


def reference_supertrend(high, low, close, period, mult):
    """Supertrend with Wilder ATR (seeded by the mean of the first `period` true
    ranges), recursion starting on the first valid ATR bar."""
    h, lo, c = (list(map(float, x)) for x in (high, low, close))
    n = len(c)
    tr = [h[0] - lo[0]] + [
        max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
        for i in range(1, n)
    ]
    direction = [np.nan] * n
    line = [np.nan] * n
    if n < period:
        return direction, line
    atr = sum(tr[:period]) / period
    start = period - 1
    upper = (h[start] + lo[start]) / 2 + mult * atr
    lower = (h[start] + lo[start]) / 2 - mult * atr
    direction[start], line[start] = 1.0, lower
    for i in range(period, n):
        atr = (atr * (period - 1) + tr[i]) / period
        mid = (h[i] + lo[i]) / 2
        new_upper, new_lower = mid + mult * atr, mid - mult * atr
        # bands only tighten while the previous close stayed inside them
        lower = lower if (new_lower < lower and c[i - 1] > lower) else new_lower
        upper = upper if (new_upper > upper and c[i - 1] < upper) else new_upper
        if direction[i - 1] == 1:
            direction[i] = -1.0 if c[i] < lower else 1.0
        else:
            direction[i] = 1.0 if c[i] > upper else -1.0
        line[i] = lower if direction[i] == 1 else upper
    return direction, line


class TestSupertrendValues:
    @pytest.mark.parametrize("period, mult", [(14, 3.0), (7, 2.0), (3, 1.0)])
    def test_matches_reference_after_one_bar_shift(self, indicator, period, mult):
        df = _random_ohlc(400, seed=21)
        result = indicator.compute(df, period=period, multiplier=mult)
        direction, line = reference_supertrend(df["H"], df["L"], df["C"], period, mult)
        pd.testing.assert_series_equal(
            result["st_direction"],
            pd.Series(direction, index=df.index).shift(1),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result["_st_line"],
            pd.Series(line, index=df.index).shift(1),
            check_names=False,
            atol=1e-9,
            rtol=0,
        )

    def test_warmup_is_nan_and_has_no_flip_events(self, indicator):
        df = _random_ohlc(200, seed=22)
        result = indicator.compute(df, period=14)
        # ATR is valid from row 13; first direction row 13 -> feature row 14;
        # the first flip decision needs a predecessor -> feature row 15.
        assert result["st_direction"].iloc[:14].isna().all()
        assert result["_st_line"].iloc[:14].isna().all()
        assert result["st_direction"].iloc[14:].notna().all()
        assert result["st_flip"].iloc[:15].isna().all()
        assert result["st_flip"].iloc[15:].notna().all()
        assert (result["_st_line"].dropna() > 0).all()

    def test_flip_marks_direction_changes(self, indicator):
        df = _random_ohlc(300, seed=23)
        result = indicator.compute(df, period=5, multiplier=1.0)
        direction = result["st_direction"]
        changed = (direction != direction.shift(1)).astype(float)
        valid = direction.notna() & direction.shift(1).notna()
        pd.testing.assert_series_equal(
            result["st_flip"][valid], changed[valid], check_names=False
        )
        assert result["st_flip"].sum() > 0

    def test_input_shorter_than_period_is_all_nan(self, indicator):
        result = indicator.compute(_random_ohlc(10, seed=24), period=14)
        assert result[["st_direction", "st_flip", "_st_line"]].isna().all().all()


def _daily_reference(df, period, mult):
    daily = df.resample("D").agg({"H": "max", "L": "min", "C": "last"})
    return reference_supertrend(daily["H"], daily["L"], daily["C"], period, mult)


class TestSupertrendMTFValues:
    def _m15(self, days=45, seed=31):
        return _random_ohlc(days * 96, seed=seed, freq="15min")

    def test_equals_supertrend_of_completed_daily_candles(self, mtf_indicator):
        df = self._m15()
        period, mult = 5, 2.0
        result = mtf_indicator.compute(df, period=period, multiplier=mult, d1_bars=96)
        d_dir, d_line = _daily_reference(df, period, mult)

        days = df.index.normalize()
        day_pos = pd.Series(range(days.nunique()), index=days.unique())
        for row_day, label in ((10, "mid"), (20, "later"), (44, "last")):
            rows = result[days == day_pos.index[row_day]]
            # every bar of day D reports the candle of day D-1
            assert (rows["st_d1_direction"] == d_dir[row_day - 1]).all(), label
            np.testing.assert_allclose(
                rows["_st_d1_line"].to_numpy(), d_line[row_day - 1], rtol=1e-12
            )

    def test_no_value_before_first_valid_completed_candle(self, mtf_indicator):
        df = self._m15()
        result = mtf_indicator.compute(df, period=5, multiplier=2.0, d1_bars=96)
        # first valid daily candle is day index 4 (period 5), complete at its last
        # bar; it becomes visible on the first bar of day 5 (row 5 * 96).
        assert result["st_d1_direction"].iloc[: 5 * 96].isna().all()
        assert result["st_d1_direction"].iloc[5 * 96 :].notna().all()

    def test_dist_atr_uses_daily_atr(self, mtf_indicator):
        df = self._m15()
        period = 5
        result = mtf_indicator.compute(df, period=period, multiplier=2.0, d1_bars=96)
        daily = df.resample("D").agg({"H": "max", "L": "min", "C": "last"})
        tr = pd.concat(
            [
                daily["H"] - daily["L"],
                (daily["H"] - daily["C"].shift(1)).abs(),
                (daily["L"] - daily["C"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = np.full(len(daily), np.nan)
        atr[period - 1] = tr.iloc[:period].mean()
        for i in range(period, len(daily)):
            atr[i] = (atr[i - 1] * (period - 1) + tr.iloc[i]) / period
        _, d_line = _daily_reference(df, period, 2.0)

        row = 30 * 96 + 17  # inside day 30; features come from the bar before
        expected = (df["C"].iloc[row - 1] - d_line[29]) / atr[29]
        assert result["st_d1_dist_atr"].iloc[row] == pytest.approx(expected, rel=1e-9)

    def test_truncation_invariance_dense_cut_points(self, mtf_indicator):
        # A partial last day must never be treated as a finished candle: cut at
        # every row around two day boundaries and compare with the full run.
        df = self._m15(days=14)
        kwargs = {"period": 3, "multiplier": 2.0, "d1_bars": 96}
        full = mtf_indicator.compute(df, **kwargs)
        cols = ["st_d1_direction", "st_d1_dist_atr", "_st_d1_line"]
        for boundary in (7 * 96, 10 * 96):
            for cut in range(boundary - 8, boundary + 8):
                part = mtf_indicator.compute(df.iloc[:cut], **kwargs)
                pd.testing.assert_frame_equal(
                    part[cols], full[cols].iloc[:cut], check_exact=False
                )

    def test_h4_buckets_use_four_hour_candles(self, mtf_indicator):
        df = self._m15(days=10)
        result = mtf_indicator.compute(df, period=3, multiplier=2.0, d1_bars=16)
        h4 = df.resample("4h").agg({"H": "max", "L": "min", "C": "last"})
        h_dir, _ = reference_supertrend(h4["H"], h4["L"], h4["C"], 3, 2.0)
        # bar 16 * 7 is the first base bar of H4 candle 7 -> shows candle 6
        assert result["st_d1_direction"].iloc[16 * 7] == h_dir[6]

    def test_invalid_d1_bars_raises(self, mtf_indicator):
        with pytest.raises(ValueError, match="d1_bars"):
            mtf_indicator.compute(self._m15(days=5), d1_bars=1)
