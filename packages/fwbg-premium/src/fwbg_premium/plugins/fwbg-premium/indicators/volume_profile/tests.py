"""Tests for VolumeProfileIndicator plugin."""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.registry import get_indicator


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_h1(n=500, seed=42):
    """Generate n H1 bars with realistic OHLCV data."""
    np.random.seed(seed)
    ret = np.random.randn(n) * 0.001
    c = 10000 * np.exp(np.cumsum(ret))
    o = np.roll(c, 1)
    o[0] = c[0]
    sp = c * 0.0001
    h = np.maximum(o, c) + np.abs(np.random.randn(n)) * sp
    low = np.minimum(o, c) - np.abs(np.random.randn(n)) * sp
    v = np.random.randint(500, 5000, n).astype(float)
    return pd.DataFrame(
        {"O": o, "H": h, "L": low, "C": c, "V": v},
        index=pd.date_range("2022-01-03 00:00", periods=n, freq="h"),
    )


def make_h1_no_volume(n=500, seed=42):
    """Like make_h1 but all V=0 to force TPO mode."""
    df = make_h1(n=n, seed=seed)
    df["V"] = 0.0
    return df


FEATURE_COLS = [
    "vp_poc_dist",
    "vp_vah_dist",
    "vp_val_dist",
    "vp_inside_va",
    "vp_poc_rel_pos",
    "vp_va_width_ratio",
]

# After day 0 is consumed as the first session, day 1 bars get profiles.
# With h=h, day 0 = 24 bars. Day 1 starts at bar index 24.
FIRST_DAY_BARS = 24


@pytest.fixture(scope="module")
def result():
    cls = get_indicator("volume_profile")
    indicator = cls()
    df = make_h1(n=500)
    return indicator.compute(df)


@pytest.fixture(scope="module")
def result_no_vol():
    cls = get_indicator("volume_profile")
    indicator = cls()
    df = make_h1_no_volume(n=500)
    return indicator.compute(df)


# ---------------------------------------------------------------------------
# TestVPFeaturePresence
# ---------------------------------------------------------------------------

class TestVPFeaturePresence:
    """Features exist after warmup (>1 day of data), first day rows are NaN."""

    def test_plugin_registered(self):
        cls = get_indicator("volume_profile")
        assert cls is not None

    def test_feature_columns_present(self, result):
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_get_feature_columns_matches(self):
        cls = get_indicator("volume_profile")
        indicator = cls()
        assert set(indicator.get_feature_columns()) == set(FEATURE_COLS)

    def test_first_day_is_nan(self, result):
        # First calendar day has no previous session profile -> all NaN
        first_day_slice = result.iloc[:FIRST_DAY_BARS]
        for col in FEATURE_COLS:
            all_nan = first_day_slice[col].isna().all()
            assert all_nan, (
                f"{col}: first day rows should all be NaN, "
                f"got non-NaN count={first_day_slice[col].notna().sum()}"
            )

    def test_features_populated_after_first_day(self, result):
        after_first_day = result.iloc[FIRST_DAY_BARS:]
        for col in FEATURE_COLS:
            non_nan = after_first_day[col].notna().sum()
            assert non_nan > 0, f"{col}: no values populated after first day"

    def test_original_columns_preserved(self, result):
        for col in ["O", "H", "L", "C", "V"]:
            assert col in result.columns

    def test_no_extra_vp_columns(self, result):
        # No unexpected vp_* columns should appear
        declared = set(FEATURE_COLS)
        vp_cols = {c for c in result.columns if c.startswith("vp_")}
        undeclared = vp_cols - declared
        assert not undeclared, f"Undeclared vp_ columns added: {undeclared}"


# ---------------------------------------------------------------------------
# TestVPValueArea
# ---------------------------------------------------------------------------

class TestVPValueArea:
    """vp_inside_va is binary (0/1), VAH > VAL structurally."""

    def test_inside_va_binary(self, result):
        vals = result["vp_inside_va"].dropna().unique()
        assert set(vals).issubset({0.0, 1.0}), (
            f"vp_inside_va has non-binary values: {vals}"
        )

    def test_vah_above_val_implied_by_width_ratio(self, result):
        # vp_va_width_ratio = (VAH - VAL) / session_range
        # If VAH >= VAL this should be >= 0
        width = result["vp_va_width_ratio"].dropna()
        assert (width >= 0).all(), "vp_va_width_ratio has negative values (VAH < VAL)"

    def test_va_width_ratio_bounded(self, result):
        # Value area cannot exceed full session range
        width = result["vp_va_width_ratio"].dropna()
        assert (width >= 0).all()
        assert (width <= 1.0 + 1e-9).all(), (
            f"vp_va_width_ratio > 1.0: max={width.max()}"
        )

    def test_poc_rel_pos_bounded(self, result):
        # POC position in previous session range: [0, 1]
        pos = result["vp_poc_rel_pos"].dropna()
        assert (pos >= 0).all(), "vp_poc_rel_pos < 0"
        assert (pos <= 1.0 + 1e-9).all(), f"vp_poc_rel_pos > 1.0: max={pos.max()}"

    def test_inside_va_consistent_with_distances(self, result):
        # When inside_va=1, vp_vah_dist >= 0 and vp_val_dist <= ... (close <= VAH => close-VAH<=0)
        # Actually: vp_vah_dist = (close - VAH)/ATR, if inside VA then close <= VAH => vp_vah_dist <= 0
        # and close >= VAL => vp_val_dist >= 0
        after = result.iloc[FIRST_DAY_BARS:].dropna(subset=["vp_inside_va", "vp_vah_dist", "vp_val_dist"])
        inside = after[after["vp_inside_va"] == 1.0]
        if len(inside) > 0:
            assert (inside["vp_vah_dist"] <= 1e-9).all(), (
                "inside VA but vp_vah_dist > 0 (close > VAH)"
            )
            assert (inside["vp_val_dist"] >= -1e-9).all(), (
                "inside VA but vp_val_dist < 0 (close < VAL)"
            )


# ---------------------------------------------------------------------------
# TestVPDistances
# ---------------------------------------------------------------------------

class TestVPDistances:
    """Distance columns exist and are finite after warmup."""

    DIST_COLS = ["vp_poc_dist", "vp_vah_dist", "vp_val_dist"]

    def test_distance_columns_present(self, result):
        for col in self.DIST_COLS:
            assert col in result.columns

    def test_distances_finite_after_warmup(self, result):
        after = result.iloc[FIRST_DAY_BARS:]
        for col in self.DIST_COLS:
            non_nan = after[col].dropna()
            assert len(non_nan) > 0, f"{col}: no values after first day"
            assert np.isfinite(non_nan).all(), f"{col}: contains inf/nan after warmup"

    def test_poc_between_vah_and_val(self, result):
        # POC must lie between VAL and VAH by construction
        # vp_poc_dist - vp_vah_dist = (close-POC)/ATR - (close-VAH)/ATR = (VAH-POC)/ATR >= 0
        # vp_poc_dist - vp_val_dist = (close-POC)/ATR - (close-VAL)/ATR = (VAL-POC)/ATR <= 0
        after = result.iloc[FIRST_DAY_BARS:].dropna(
            subset=["vp_poc_dist", "vp_vah_dist", "vp_val_dist"]
        )
        # VAH >= POC => (close - VAH) <= (close - POC) => vp_vah_dist <= vp_poc_dist
        vah_lte_poc = (after["vp_vah_dist"] <= after["vp_poc_dist"] + 1e-9).all()
        assert vah_lte_poc, "POC should be below or at VAH"
        # VAL <= POC => (close - VAL) >= (close - POC) => vp_val_dist >= vp_poc_dist
        val_gte_poc = (after["vp_val_dist"] >= after["vp_poc_dist"] - 1e-9).all()
        assert val_gte_poc, "POC should be above or at VAL"

    def test_no_inf_in_distances(self, result):
        for col in self.DIST_COLS:
            series = result[col].dropna()
            assert not np.isinf(series).any(), f"{col} contains inf values"


# ---------------------------------------------------------------------------
# TestVPWithVolume
# ---------------------------------------------------------------------------

class TestVPWithVolume:
    """vol-weighted POC differs from no-vol TPO in some cases."""

    def test_volume_result_columns_present(self, result):
        for col in FEATURE_COLS:
            assert col in result.columns

    def test_tpo_result_columns_present(self, result_no_vol):
        for col in FEATURE_COLS:
            assert col in result_no_vol.columns

    def test_both_produce_features_after_first_day(self, result, result_no_vol):
        for col in ["vp_poc_dist", "vp_inside_va"]:
            vol_vals = result.iloc[FIRST_DAY_BARS:][col].dropna()
            tpo_vals = result_no_vol.iloc[FIRST_DAY_BARS:][col].dropna()
            assert len(vol_vals) > 0, f"volume mode: {col} empty after first day"
            assert len(tpo_vals) > 0, f"TPO mode: {col} empty after first day"

    def test_poc_dist_differs_between_modes(self, result, result_no_vol):
        # With random volume the vol-weighted POC should differ from uniform-weight TPO
        # in at least some bars. They won't be identical over 500 bars.
        vol_poc = result.iloc[FIRST_DAY_BARS:]["vp_poc_dist"].dropna()
        tpo_poc = result_no_vol.iloc[FIRST_DAY_BARS:]["vp_poc_dist"].dropna()
        # Align on common index
        common = vol_poc.index.intersection(tpo_poc.index)
        if len(common) > 0:
            diff = (vol_poc.loc[common] - tpo_poc.loc[common]).abs()
            # At least one bar should differ between vol-weighted and TPO
            assert diff.max() > 0, (
                "Volume-weighted and TPO POC are identical for all bars — "
                "volume weighting has no effect"
            )

    def test_inside_va_binary_no_volume(self, result_no_vol):
        vals = result_no_vol["vp_inside_va"].dropna().unique()
        assert set(vals).issubset({0.0, 1.0}), (
            f"TPO mode vp_inside_va has non-binary values: {vals}"
        )

    def test_daily_data_returns_unchanged(self):
        # Plugin should return df unchanged for daily-frequency data
        cls = get_indicator("volume_profile")
        indicator = cls()
        n = 100
        close = 10000 * np.ones(n)
        daily_df = pd.DataFrame(
            {"O": close, "H": close * 1.01, "L": close * 0.99, "C": close, "V": close * 100},
            index=pd.date_range("2022-01-03", periods=n, freq="D"),
        )
        result = indicator.compute(daily_df)
        # No vp_ columns should be added on daily data
        vp_cols = [c for c in result.columns if c.startswith("vp_")]
        assert len(vp_cols) == 0, f"vp_ columns added on daily data: {vp_cols}"
