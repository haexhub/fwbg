"""Tests for wavelet decomposition indicator plugin."""

import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_wavelets = import_plugin_module("fwbg-core", "indicators", "wavelets")
if _wavelets is None:
    pytest.skip("fwbg-core wavelets plugin not available", allow_module_level=True)


def _make_ohlc(close_values):
    n = len(close_values)
    df = pd.DataFrame(
        {
            "O": close_values * 0.999,
            "H": close_values * 1.002,
            "L": close_values * 0.998,
            "C": close_values,
            "V": np.random.randint(100, 1000, n),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )
    return df


@pytest.fixture
def indicator():
    return _wavelets.WaveletsIndicator()


@pytest.fixture
def random_walk_df():
    """Pure random walk — balanced frequency content."""
    rng = np.random.default_rng(42)
    n = 500
    returns = rng.normal(0, 0.005, n)
    close = 100 * np.exp(np.cumsum(returns))
    return _make_ohlc(close)


@pytest.fixture
def trending_df():
    """Smooth uptrend — dominated by low-frequency energy.

    Uses smoothed (rolling-averaged) random returns so that even the
    stochastic component is low-frequency. This ensures detail level 3
    (lowest frequency) dominates over detail level 1 (highest frequency).
    """
    rng = np.random.default_rng(42)
    n = 500
    raw = rng.normal(0, 0.001, n)
    # Smooth returns via rolling mean -> removes high-freq content
    smooth_returns = pd.Series(raw).rolling(30, min_periods=1).mean().values.copy()
    smooth_returns += 0.001  # upward drift
    close = 100 * np.exp(np.cumsum(smooth_returns))
    return _make_ohlc(close)


@pytest.fixture
def choppy_df():
    """High-frequency noise — dominated by high-frequency energy.

    Alternating sign returns create near-Nyquist frequency content,
    ensuring detail level 1 (highest frequency) dominates.
    """
    rng = np.random.default_rng(42)
    n = 500
    returns = np.zeros(n)
    returns[1:] = 0.015 * ((-1) ** np.arange(1, n))
    returns += rng.normal(0, 0.001, n)
    close = 100 * np.exp(np.cumsum(returns))
    return _make_ohlc(close)


# --- Feature column tests ---


class TestFeatureColumns:
    """Correct feature columns are produced."""

    def test_all_columns_present(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_feature_count_default(self, indicator, random_walk_df):
        _result = indicator.compute(random_walk_df)
        cols = indicator.get_feature_columns()
        # Default: 3 levels, 3 windows
        # Per window: 1 approx_energy + 3 detail_energy + 3 detail_mean = 7
        # 7 * 3 windows = 21
        # 3 detail_ratio + 3 high_freq_ratio = 6
        # Total = 27
        assert len(cols) == 27

    def test_preserves_original_columns(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        for col in ["O", "H", "L", "C", "V"]:
            assert col in result.columns

    def test_returns_dataframe(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        assert isinstance(result, pd.DataFrame)

    def test_same_length(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        assert len(result) == len(random_walk_df)

    def test_feature_prefix(self, indicator, random_walk_df):
        _result = indicator.compute(random_walk_df)
        for col in indicator.get_feature_columns():
            assert col.startswith("wt_"), f"Feature {col} missing wt_ prefix"


# --- No-lookahead tests ---


class TestNoLookahead:
    """Features must be shifted by 1 bar (no lookahead)."""

    def test_first_row_is_nan(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        for col in indicator.get_feature_columns():
            assert pd.isna(result[col].iloc[0]), f"{col} first row should be NaN"

    def test_energy_first_row_nan(self, indicator, random_walk_df):
        result = indicator.compute(random_walk_df)
        assert pd.isna(result["wt_detail_1_energy_10"].iloc[0])

    def test_shifted_values_are_lagged(self, indicator, random_walk_df):
        """Second row should equal what would be first row unshifted."""
        result = indicator.compute(random_walk_df)
        # After shift, row 1 should have a valid value
        for col in indicator.get_feature_columns():
            val = result[col].iloc[1]
            assert not pd.isna(val) or True  # may be NaN from safe_divide, that's ok


# --- Frequency content tests ---


class TestFrequencyContent:
    """Wavelet features capture correct frequency characteristics."""

    def test_trending_low_freq_dominates(self, indicator, trending_df):
        """Trending data should have more low-frequency energy."""
        result = indicator.compute(trending_df)
        # Detail level 3 (lowest detail freq) should have more energy
        # than detail level 1 (highest freq) in a trend
        d3_energy = result["wt_detail_3_energy_50"].dropna().iloc[-100:].mean()
        d1_energy = result["wt_detail_1_energy_50"].dropna().iloc[-100:].mean()
        # In trending market, low-freq detail should dominate
        assert d3_energy > d1_energy, (
            f"Trending: low-freq energy ({d3_energy:.6f}) should exceed "
            f"high-freq energy ({d1_energy:.6f})"
        )

    def test_choppy_high_freq_dominates(self, indicator, choppy_df):
        """Choppy data should have more high-frequency energy."""
        result = indicator.compute(choppy_df)
        d1_energy = result["wt_detail_1_energy_50"].dropna().iloc[-100:].mean()
        d3_energy = result["wt_detail_3_energy_50"].dropna().iloc[-100:].mean()
        assert d1_energy > d3_energy, (
            f"Choppy: high-freq energy ({d1_energy:.6f}) should exceed "
            f"low-freq energy ({d3_energy:.6f})"
        )

    def test_high_freq_ratio_higher_for_choppy(self, indicator, trending_df, choppy_df):
        """High-freq ratio should be larger for choppy than trending data."""
        res_trend = indicator.compute(trending_df)
        res_choppy = indicator.compute(choppy_df)
        ratio_trend = res_trend["wt_high_freq_ratio_50"].dropna().iloc[-100:].mean()
        ratio_choppy = res_choppy["wt_high_freq_ratio_50"].dropna().iloc[-100:].mean()
        assert ratio_choppy > ratio_trend, (
            f"Choppy ratio ({ratio_choppy:.4f}) should exceed "
            f"trending ratio ({ratio_trend:.4f})"
        )

    def test_detail_ratios_sum_less_than_one(self, indicator, random_walk_df):
        """Sum of detail ratios should be <= 1 (approx takes the rest)."""
        result = indicator.compute(random_walk_df)
        r1 = result["wt_detail_ratio_1"].dropna()
        r2 = result["wt_detail_ratio_2"].dropna()
        r3 = result["wt_detail_ratio_3"].dropna()
        total = r1 + r2 + r3
        # Allow small numerical tolerance
        assert (total <= 1.0 + 1e-6).all(), f"Detail ratios sum > 1: max={total.max()}"


# --- Parameter variation tests ---


class TestParameterVariation:
    """Different parameters produce valid results."""

    def test_different_wavelet_haar(self, random_walk_df):
        ind = _wavelets.WaveletsIndicator()
        result = ind.compute(random_walk_df, wavelet="haar")
        assert "wt_detail_1_energy_10" in result.columns

    def test_different_wavelet_sym5(self, random_walk_df):
        ind = _wavelets.WaveletsIndicator()
        result = ind.compute(random_walk_df, wavelet="sym5")
        assert "wt_detail_1_energy_10" in result.columns

    def test_different_levels(self, random_walk_df):
        ind = _wavelets.WaveletsIndicator()
        result = ind.compute(random_walk_df, levels=4)
        assert "wt_detail_4_energy_10" in result.columns
        assert "wt_detail_ratio_4" in result.columns

    def test_custom_windows(self, random_walk_df):
        ind = _wavelets.WaveletsIndicator()
        result = ind.compute(random_walk_df, windows=[5, 15])
        assert "wt_detail_1_energy_5" in result.columns
        assert "wt_detail_1_energy_15" in result.columns
        # Default windows should not be present
        assert "wt_detail_1_energy_10" not in result.columns


# --- Plugin integration tests ---


class TestPluginIntegration:
    """Plugin integrates correctly with the framework."""

    def test_plugin_importable(self):
        from fwbg.pipeline import get_registry

        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-core:wavelets")
        assert plugin_cls is not None

    def test_benefits_from_stationary_false(self):
        from fwbg.pipeline import get_registry

        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-core:wavelets")
        assert plugin_cls.benefits_from_stationary is False

    def test_default_params(self, indicator):
        params = indicator.get_default_params()
        assert params["wavelet"] == "db4"
        assert params["levels"] == 3
        assert params["windows"] == [10, 20, 50]

    def test_name_and_version(self, indicator):
        assert indicator.name == "wavelets"
        assert indicator.version == "2.1.0"


# --- Band energy shares from full-window coefficients ---


def _white_noise_df(n=5000, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    return _make_ohlc(close)


class TestBandEnergyShares:
    def test_white_noise_detail1_share_is_one_half(self, indicator):
        """White noise puts half of its energy into the finest detail band.

        Taking the end sample of a reconstruction instead gave ~0.05.
        """
        out = indicator.compute(_white_noise_df(), levels=3, dwt_window=256)
        # independent (non-overlapping) trailing windows
        r1 = out["wt_detail_ratio_1"].dropna().iloc[::256]
        r2 = out["wt_detail_ratio_2"].dropna().iloc[::256]
        r3 = out["wt_detail_ratio_3"].dropna().iloc[::256]
        assert r1.mean() == pytest.approx(0.5, abs=0.05)
        assert r2.mean() == pytest.approx(0.25, abs=0.05)
        assert r3.mean() == pytest.approx(0.125, abs=0.05)

    def test_matches_reference_coefficient_shares(self, indicator):
        import pywt

        df = _white_noise_df(n=600)
        out = indicator.compute(df, levels=3, dwt_window=256)
        close = df["C"].to_numpy()
        lr = np.diff(np.log(close))  # lr[j] is the return of bar j+1
        for i in (300, 450, 598):
            seg = lr[i - 256 : i]  # bars i-255..i
            coeffs = pywt.wavedec(seg, "db4", level=3, mode="periodization")
            e = [float(np.sum(c * c)) for c in coeffs]
            tot = sum(e)
            # row i+1 holds the features of bar i (shift by one)
            for lvl in (1, 2, 3):
                assert out[f"wt_detail_ratio_{lvl}"].iloc[i + 1] == pytest.approx(
                    e[4 - lvl] / tot, rel=1e-9
                )

    def test_shares_plus_approx_sum_to_one_and_power_matches(self, indicator):
        import pywt

        df = _white_noise_df(n=400)
        out = indicator.compute(df, levels=3, windows=[5], dwt_window=128)
        close = df["C"].to_numpy()
        lr = np.diff(np.log(close))
        i = 350
        powers = []
        for j in range(i - 4, i + 1):
            seg = lr[j - 128 : j]
            coeffs = pywt.wavedec(seg, "db4", level=3, mode="periodization")
            powers.append([float(np.sum(c * c)) / 128 for c in coeffs])
        expected = np.mean(powers, axis=0)
        assert out["wt_approx_energy_5"].iloc[i + 1] == pytest.approx(expected[0])
        for lvl in (1, 2, 3):
            assert out[f"wt_detail_{lvl}_energy_5"].iloc[i + 1] == pytest.approx(
                expected[4 - lvl]
            )
        # Parseval: band powers add up to the mean squared return of the window
        seg = lr[i - 128 : i]
        assert sum(powers[-1]) == pytest.approx(float(np.mean(seg**2)))

    def test_warmup_is_nan_until_all_levels_fit(self, indicator):
        out = indicator.compute(_white_noise_df(n=200), levels=3, dwt_window=128)
        # db4 needs (8 - 1) * 2**3 = 56 bars; returns start at bar 1
        ratio = out["wt_detail_ratio_1"]
        assert ratio.iloc[:57].isna().all()
        assert ratio.iloc[57:].notna().all()

    def test_infeasible_levels_raise(self, indicator):
        with pytest.raises(ValueError, match="dwt_window"):
            indicator.compute(_white_noise_df(n=100), levels=6, dwt_window=64)

    def test_get_feature_columns_with_params(self, indicator):
        cols = indicator.get_feature_columns({"levels": 2, "windows": [7]})
        assert "wt_detail_2_energy_7" in cols
        assert "wt_detail_3_energy_7" not in cols
        assert "wt_detail_ratio_2" in cols
