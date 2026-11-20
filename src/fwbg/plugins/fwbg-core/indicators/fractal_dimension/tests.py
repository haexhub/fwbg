"""Tests for Higuchi Fractal Dimension indicator plugin."""

import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_fd = import_plugin_module("fwbg-core", "indicators", "fractal_dimension")
if _fd is None:
    pytest.skip(
        "fwbg-core fractal_dimension plugin not available", allow_module_level=True
    )


@pytest.fixture
def indicator():
    return _fd.FractalDimensionIndicator()


def _make_ohlc(close):
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trending_df():
    """Strong linear trend with minimal noise — should have low FD (~1.0)."""
    n = 500
    close = 100 + np.linspace(0, 50, n) + np.random.default_rng(42).normal(0, 0.1, n)
    return _make_ohlc(close)


@pytest.fixture
def random_walk_df():
    """Random walk — FD should be near 1.5."""
    n = 500
    close = 100 + np.cumsum(np.random.default_rng(42).normal(0, 1, n))
    return _make_ohlc(close)


@pytest.fixture
def noisy_df():
    """IID noise (not a walk) — FD should be near 2.0."""
    n = 500
    close = 100 + np.random.default_rng(42).normal(0, 5, n)
    return _make_ohlc(close)


@pytest.fixture
def sine_df():
    """Sine wave — smooth/periodic, should have low FD (~1.0-1.3)."""
    n = 500
    t = np.linspace(0, 8 * np.pi, n)
    close = 100 + 10 * np.sin(t)
    return _make_ohlc(close)


# ---------------------------------------------------------------------------
# Feature column tests
# ---------------------------------------------------------------------------


class TestFeatureColumns:
    """All expected columns are produced."""

    def test_all_columns_present(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df, windows=[50])
        for suffix in [
            "higuchi_50",
            "higuchi_change_50",
            "complexity_ratio_50",
            "regime_50",
        ]:
            col = f"fd_{suffix}"
            assert col in result.columns, f"Missing: {col}"

    def test_default_windows_columns(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_feature_count(self, indicator):
        # 3 windows * 4 features = 12
        assert len(indicator.get_feature_columns()) == 12

    def test_returns_dataframe(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df, windows=[50])
        assert isinstance(result, pd.DataFrame)

    def test_preserves_original_columns(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df, windows=[50])
        for col in ["O", "H", "L", "C"]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# No-lookahead tests
# ---------------------------------------------------------------------------


class TestNoLookahead:
    """Features must be shifted by 1 bar (no lookahead)."""

    def test_first_row_is_nan(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df, windows=[50])
        for col in [
            "fd_higuchi_50",
            "fd_higuchi_change_50",
            "fd_complexity_ratio_50",
            "fd_regime_50",
        ]:
            assert pd.isna(result[col].iloc[0]), f"{col} first row should be NaN"

    def test_shift_by_one(self, indicator, random_walk_df):
        """The value at index i should reflect data up to i-1."""
        result = indicator.compute(random_walk_df, windows=[50])
        fd = result["fd_higuchi_50"]
        # First non-NaN should be at row index 50 (warmup=50, then shift+1)
        first_valid = fd.first_valid_index()
        first_valid_pos = result.index.get_loc(first_valid)
        # window=50 means first computed at idx 49, shifted to idx 50
        assert first_valid_pos == 50


# ---------------------------------------------------------------------------
# Fractal dimension value tests
# ---------------------------------------------------------------------------


class TestFractalDimensionValues:
    """HFD produces correct values for known signals."""

    def test_linear_trend_low_fd(self, indicator, trending_df):
        result = indicator.compute(trending_df, windows=[50], k_max=10)
        fd = result["fd_higuchi_50"].dropna()
        median_fd = fd.median()
        assert 1.0 <= median_fd <= 1.35, (
            f"Linear trend should have low FD, got median={median_fd:.3f}"
        )

    def test_sine_wave_low_fd(self, indicator, sine_df):
        result = indicator.compute(sine_df, windows=[50], k_max=10)
        fd = result["fd_higuchi_50"].dropna()
        median_fd = fd.median()
        assert 1.0 <= median_fd <= 1.35, (
            f"Sine wave should have low FD, got median={median_fd:.3f}"
        )

    def test_random_walk_mid_fd(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df, windows=[50], k_max=10)
        fd = result["fd_higuchi_50"].dropna()
        median_fd = fd.median()
        assert 1.3 <= median_fd <= 1.7, (
            f"Random walk should have FD near 1.5, got median={median_fd:.3f}"
        )

    def test_noisy_data_high_fd(self, indicator, noisy_df):
        result = indicator.compute(noisy_df, windows=[50], k_max=10)
        fd = result["fd_higuchi_50"].dropna()
        median_fd = fd.median()
        assert 1.7 <= median_fd <= 2.1, (
            f"IID noise should have high FD near 2.0, got median={median_fd:.3f}"
        )

    def test_fd_bounded(self, indicator, random_walk_df):
        """FD values should generally stay within [1.0, 2.0]."""
        result = indicator.compute(random_walk_df, windows=[50], k_max=10)
        fd = result["fd_higuchi_50"].dropna()
        assert fd.min() >= 0.8, f"FD too low: {fd.min():.3f}"
        assert fd.max() <= 2.2, f"FD too high: {fd.max():.3f}"


# ---------------------------------------------------------------------------
# Derived feature tests
# ---------------------------------------------------------------------------


class TestDerivedFeatures:
    """Change, complexity ratio, and regime features."""

    def test_complexity_ratio_near_zero_for_random_walk(
        self, indicator, random_walk_df
    ):
        result = indicator.compute(random_walk_df, windows=[50], k_max=10)
        cr = result["fd_complexity_ratio_50"].dropna()
        median_cr = cr.median()
        assert median_cr < 0.5, (
            f"Random walk complexity ratio should be near 0, got {median_cr:.3f}"
        )

    def test_complexity_ratio_high_for_trend(self, indicator, trending_df):
        result = indicator.compute(trending_df, windows=[50], k_max=10)
        cr = result["fd_complexity_ratio_50"].dropna()
        median_cr = cr.median()
        assert median_cr > 0.3, (
            f"Trending complexity ratio should be high, got {median_cr:.3f}"
        )

    def test_regime_trending(self, indicator, trending_df):
        result = indicator.compute(trending_df, windows=[50], k_max=10)
        regime = result["fd_regime_50"].dropna()
        # Most values should be -1 (trending)
        trending_frac = (regime == -1.0).mean()
        assert trending_frac > 0.5, (
            f"Trending data should have mostly regime=-1, got {trending_frac:.1%}"
        )

    def test_regime_mean_reverting(self, indicator, noisy_df):
        result = indicator.compute(noisy_df, windows=[50], k_max=10)
        regime = result["fd_regime_50"].dropna()
        # Most values should be 1 (mean-reverting / complex)
        mr_frac = (regime == 1.0).mean()
        assert mr_frac > 0.5, (
            f"Noisy data should have mostly regime=1, got {mr_frac:.1%}"
        )

    def test_regime_values_discrete(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df, windows=[50], k_max=10)
        regime = result["fd_regime_50"].dropna()
        unique = set(regime.unique())
        assert unique.issubset({-1.0, 0.0, 1.0}), (
            f"Regime should only contain -1, 0, 1; got {unique}"
        )

    def test_higuchi_change_exists(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df, windows=[50], k_max=10)
        change = result["fd_higuchi_change_50"].dropna()
        assert len(change) > 0, "Change feature should have non-NaN values"


# ---------------------------------------------------------------------------
# Parameter variation tests
# ---------------------------------------------------------------------------


class TestParameterVariation:
    """Different windows and k_max values work correctly."""

    def test_different_windows(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df, windows=[30, 80])
        assert "fd_higuchi_30" in result.columns
        assert "fd_higuchi_80" in result.columns
        # Smaller window should have earlier first valid value
        fd30 = result["fd_higuchi_30"]
        fd80 = result["fd_higuchi_80"]
        first30 = result.index.get_loc(fd30.first_valid_index())
        first80 = result.index.get_loc(fd80.first_valid_index())
        assert first30 < first80

    def test_different_k_max(self, indicator, random_walk_df):
        r1 = indicator.compute(random_walk_df, windows=[50], k_max=5)
        r2 = indicator.compute(random_walk_df, windows=[50], k_max=15)
        fd1 = r1["fd_higuchi_50"].dropna()
        fd2 = r2["fd_higuchi_50"].dropna()
        # Both should produce valid values; results may differ slightly
        assert len(fd1) > 0
        assert len(fd2) > 0
        # But both should be in reasonable range for random walk
        assert 1.2 <= fd1.median() <= 1.8
        assert 1.2 <= fd2.median() <= 1.8


# ---------------------------------------------------------------------------
# Plugin integration tests
# ---------------------------------------------------------------------------


class TestPluginIntegration:
    """Plugin integrates correctly with the registry."""

    def test_plugin_importable(self):
        from fwbg.pipeline import get_registry

        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-core:fractal_dimension")
        assert plugin_cls is not None

    def test_benefits_from_stationary_false(self):
        from fwbg.pipeline import get_registry

        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-core:fractal_dimension")
        assert plugin_cls.benefits_from_stationary is False

    def test_default_params(self, indicator):
        params = indicator.get_default_params()
        assert "windows" in params
        assert "k_max" in params
        assert params["windows"] == [50, 100, 200]
        assert params["k_max"] == 10

    def test_name_and_version(self, indicator):
        assert indicator.name == "fractal_dimension"
        assert indicator.version == "1.1.0"


# ---------------------------------------------------------------------------
# Regime dead band calibration and windows-aware columns
# ---------------------------------------------------------------------------


class TestRegimeCalibration:
    @staticmethod
    def _rw_regime_fraction(window, k_max, n_paths=300, seed=11):
        rng = np.random.default_rng(seed)
        ind = _fd.FractalDimensionIndicator()
        nonzero = 0
        for _ in range(n_paths):
            # window + 2 closes: first valid fd row is at bar window-1, we read
            # the last (shifted) row which holds the fd of the full window
            close = 100 + np.cumsum(rng.normal(0, 1, window + 1))
            out = ind.compute(_make_ohlc(close), windows=[window], k_max=k_max)
            r = out[f"fd_regime_{window}"].iloc[-1]
            assert not np.isnan(r)
            nonzero += r != 0.0
        return nonzero / n_paths

    @pytest.mark.parametrize("window", [50, 100])
    def test_random_walk_rarely_leaves_dead_band(self, window):
        frac = self._rw_regime_fraction(window, 10)
        # target ~5 %; allow sampling noise and calibration error. A fixed
        # 1.4..1.6 band gives ~43 % at window=50 and ~25 % at window=100.
        assert frac < 0.12, f"RW labelled trend/MR in {frac:.1%} of windows"

    def test_rw_fd_std_matches_simulation(self):
        rng = np.random.default_rng(5)
        for window in (50, 200):
            vals = [
                _fd._higuchi_fd(np.cumsum(rng.normal(size=window)), 10)
                for _ in range(400)
            ]
            assert _fd.rw_fd_std(window, 10) == pytest.approx(np.std(vals), rel=0.2)
            assert np.mean(vals) == pytest.approx(1.5, abs=0.03)

    def test_band_is_exact_function_of_estimator_noise(self, indicator, random_walk_df):
        out = indicator.compute(random_walk_df, windows=[50], k_max=10, regime_z=1.96)
        fd = out["fd_higuchi_50"]
        band = 1.96 * _fd.rw_fd_std(50, 10)
        expected = pd.Series(
            np.where(
                fd.isna(),
                np.nan,
                np.where(fd < 1.5 - band, -1.0, np.where(fd > 1.5 + band, 1.0, 0.0)),
            ),
            index=fd.index,
        )
        pd.testing.assert_series_equal(out["fd_regime_50"], expected, check_names=False)


class TestWindowsParamColumns:
    def test_get_feature_columns_honours_windows(self, indicator):
        cols = indicator.get_feature_columns({"windows": [30]})
        assert cols == [
            "fd_higuchi_30",
            "fd_higuchi_change_30",
            "fd_complexity_ratio_30",
            "fd_regime_30",
        ]
        assert indicator.get_signal_columns({"windows": [30, 60]}) == [
            "fd_regime_30",
            "fd_regime_60",
        ]

    def test_columns_match_compute_for_custom_windows(self, indicator, random_walk_df):
        out = indicator.compute(random_walk_df, windows=[30, 60])
        for col in indicator.get_feature_columns({"windows": [30, 60]}):
            assert col in out.columns
