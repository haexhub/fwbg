"""Test MFE target computation."""

import numpy as np
import pandas as pd
import pytest

from fwbg.optimization.targets import compute_mfe_targets


def _make_trending_df(n=100, direction="up"):
    np.random.seed(42)
    if direction == "up":
        base = 100.0 + np.arange(n) * 0.5 + np.random.randn(n) * 0.2
    else:
        base = 200.0 - np.arange(n) * 0.5 + np.random.randn(n) * 0.2
    df = pd.DataFrame(
        {
            "O": base,
            "H": base + abs(np.random.randn(n)) * 0.5,
            "L": base - abs(np.random.randn(n)) * 0.5,
            "C": base + np.random.randn(n) * 0.1,
            "_atr": np.full(n, 1.0),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="15min"),
    )
    return df


class TestComputeMfeTargets:
    def test_returns_correct_shape(self):
        df = _make_trending_df(100)
        mfe_long, mfe_short = compute_mfe_targets(
            df, sl_atr=2.0, max_bars=20, spread=0.5
        )
        assert mfe_long.shape == (100,)
        assert mfe_short.shape == (100,)

    def test_mfe_non_negative(self):
        df = _make_trending_df(100)
        mfe_long, mfe_short = compute_mfe_targets(
            df, sl_atr=2.0, max_bars=20, spread=0.5
        )
        assert np.all(mfe_long >= 0.0)
        assert np.all(mfe_short >= 0.0)

    def test_uptrend_favors_long_mfe(self):
        df = _make_trending_df(100, direction="up")
        mfe_long, mfe_short = compute_mfe_targets(
            df, sl_atr=3.0, max_bars=50, spread=0.1
        )
        assert np.nanmean(mfe_long) > np.nanmean(mfe_short)

    def test_downtrend_favors_short_mfe(self):
        df = _make_trending_df(100, direction="down")
        mfe_long, mfe_short = compute_mfe_targets(
            df, sl_atr=3.0, max_bars=50, spread=0.1
        )
        assert np.nanmean(mfe_short) > np.nanmean(mfe_long)

    def test_last_bar_is_zero(self):
        df = _make_trending_df(50)
        mfe_long, mfe_short = compute_mfe_targets(
            df, sl_atr=2.0, max_bars=20, spread=0.0
        )
        assert mfe_long[-1] == 0.0
        assert mfe_short[-1] == 0.0

    def test_nan_atr_skipped(self):
        df = _make_trending_df(50)
        df.loc[df.index[5], "_atr"] = np.nan
        mfe_long, _ = compute_mfe_targets(df, sl_atr=2.0, max_bars=20, spread=0.0)
        assert mfe_long[5] == 0.0

    def test_vol_atr_fallback(self):
        df = _make_trending_df(50)
        df.rename(columns={"_atr": "vol_atr"}, inplace=True)
        mfe_long, mfe_short = compute_mfe_targets(
            df, sl_atr=2.0, max_bars=20, spread=0.0
        )
        assert mfe_long.shape == (50,)

    def test_no_atr_raises(self):
        df = _make_trending_df(50)
        df.drop(columns=["_atr"], inplace=True)
        with pytest.raises(ValueError, match="ATR column"):
            compute_mfe_targets(df, sl_atr=2.0, max_bars=20, spread=0.0)


class TestMfeStopsAtTheStopBar:
    """The stop bar's favourable extreme is excluded (SL first)."""

    @staticmethod
    def _df(rows):
        a = np.array(rows, dtype=float)
        return pd.DataFrame(
            {"O": a[:, 0], "H": a[:, 1], "L": a[:, 2], "C": a[:, 3], "_atr": 1.0}
        )

    def test_long_mfe_excludes_the_high_of_the_stop_bar(self):
        # entry 100, SL 1 ATR below.  Bar 1 reaches +0.5; bar 2 hits 99 and also
        # spikes to 103 -- the spike may have come after the stop-out.
        df = self._df(
            [
                (100, 100, 100, 100),
                (100, 100.5, 99.5, 100),
                (100, 103, 99, 100),
                (100, 100, 100, 100),
            ]
        )
        mfe_long, _ = compute_mfe_targets(df, sl_atr=1.0, max_bars=10)
        assert mfe_long[0] == pytest.approx(0.5)

    def test_short_mfe_excludes_the_low_of_the_stop_bar(self):
        df = self._df(
            [
                (100, 100, 100, 100),
                (100, 100.5, 99.5, 100),
                (100, 101, 97, 100),
                (100, 100, 100, 100),
            ]
        )
        _, mfe_short = compute_mfe_targets(df, sl_atr=1.0, max_bars=10)
        assert mfe_short[0] == pytest.approx(0.5)

    def test_bar_without_stop_still_counts(self):
        df = self._df(
            [(100, 100, 100, 100), (100, 103, 99.5, 102), (100, 100, 100, 100)]
        )
        mfe_long, _ = compute_mfe_targets(df, sl_atr=1.0, max_bars=10)
        assert mfe_long[0] == pytest.approx(3.0)
