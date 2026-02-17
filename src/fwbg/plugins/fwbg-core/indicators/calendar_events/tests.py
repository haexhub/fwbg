"""Tests for calendar anomaly indicator plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_cal = import_plugin_module("fwbg-core", "indicators", "calendar_events")
if _cal is None:
    pytest.skip("fwbg-core calendar_events plugin not available", allow_module_level=True)


@pytest.fixture
def indicator():
    return _cal.CalendarEventsIndicator()


@pytest.fixture
def two_year_df():
    """2+ years of hourly data covering all calendar events."""
    n = 365 * 2 * 24  # 2 years of hourly data
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.01, n))
    return pd.DataFrame(
        {
            "O": close * 0.999,
            "H": close * 1.002,
            "L": close * 0.998,
            "C": close,
            "V": rng.integers(100, 1000, n),
        },
        index=idx,
    )


class TestCalendarFeatureColumns:
    """All expected columns are produced."""

    def test_all_columns_present(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_feature_count(self, indicator):
        assert len(indicator.get_feature_columns()) == 9

    def test_returns_dataframe(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_original_columns(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        for col in ["O", "H", "L", "C", "V"]:
            assert col in result.columns


class TestCalendarNoLookahead:
    """Features must be shifted by 1 bar (no lookahead)."""

    def test_first_row_is_nan(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        for col in indicator.get_feature_columns():
            assert pd.isna(result[col].iloc[0]), f"{col} first row should be NaN"

    def test_shift_by_one(self, indicator):
        """Verify feature at bar i reflects bar i-1's calendar info."""
        # Use a small DF spanning month boundary: Dec 31 -> Jan 1
        idx = pd.date_range("2023-12-31 22:00", periods=5, freq="h")
        close = np.array([100.0, 100.1, 100.2, 100.3, 100.4])
        df = pd.DataFrame(
            {"O": close, "H": close, "L": close, "C": close, "V": [100] * 5},
            index=idx,
        )
        result = indicator.compute(df)
        # Bar 0 (Dec 31 22:00) -> NaN (shifted)
        assert pd.isna(result["cal_year_boundary"].iloc[0])
        # Bar 1 (Dec 31 23:00) -> reflects bar 0 (Dec 31 22:00) -> should be 1
        assert result["cal_year_boundary"].iloc[1] == 1.0


class TestTurnOfMonth:
    """Turn-of-month: first 3 and last 2 calendar days."""

    def test_jan_first_three_days(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Check Jan 2, 2023 (shifted from Jan 1)
        jan_2 = result.loc["2023-01-02", "cal_turn_of_month"]
        assert (jan_2 == 1.0).all(), "Jan 2 should show turn_of_month=1 (from Jan 1)"

    def test_mid_month_is_zero(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Jan 16 should reflect Jan 15 -> day 15 is mid-month
        jan_16 = result.loc["2023-01-16", "cal_turn_of_month"]
        assert (jan_16 == 0.0).all(), "Mid-month should be 0"

    def test_month_end_days(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Feb 1 reflects Jan 31 (last day) -> should be 1
        feb_1 = result.loc["2023-02-01", "cal_turn_of_month"]
        assert (feb_1 == 1.0).all(), "Last day of month should be turn_of_month=1"


class TestTripleWitching:
    """Triple witching: within 2 days of 3rd Friday of Mar/Jun/Sep/Dec."""

    def test_march_2023_triple_witching(self, indicator, two_year_df):
        """3rd Friday of March 2023 is March 17."""
        result = indicator.compute(two_year_df)
        # Check shifted: Mar 17 reflects Mar 16 which is within 2 days of Mar 17
        mar_17 = result.loc["2023-03-17", "cal_triple_witching"]
        assert (mar_17 == 1.0).all(), "Mar 17 should be triple witching (from Mar 16)"

    def test_non_witching_month_is_zero(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Feb 15 reflects Feb 14 -> Feb is not a witching month
        feb_15 = result.loc["2023-02-15", "cal_triple_witching"]
        assert (feb_15 == 0.0).all(), "Feb is not a triple witching month"

    def test_june_2023_triple_witching(self, indicator, two_year_df):
        """3rd Friday of June 2023 is June 16."""
        result = indicator.compute(two_year_df)
        # Jun 16 reflects Jun 15 which is 1 day before 3rd Friday -> within 2 days
        jun_16 = result.loc["2023-06-16", "cal_triple_witching"]
        assert (jun_16 == 1.0).all(), "Jun 16 should be triple witching"


class TestMonthlyOpex:
    """Monthly OpEx: within 2 days of 3rd Friday of every month."""

    def test_january_opex(self, indicator, two_year_df):
        """3rd Friday of Jan 2023 is Jan 20."""
        result = indicator.compute(two_year_df)
        # Jan 20 reflects Jan 19 -> abs(19 - 20) = 1 <= 2 -> should be 1
        jan_20 = result.loc["2023-01-20", "cal_monthly_opex"]
        assert (jan_20 == 1.0).all(), "Jan 20 should be monthly opex"

    def test_far_from_opex_is_zero(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Jan 10 reflects Jan 9 -> 3rd Friday is Jan 20 -> abs(9-20) = 11 -> 0
        jan_10 = result.loc["2023-01-10", "cal_monthly_opex"]
        assert (jan_10 == 0.0).all(), "Far from opex should be 0"


class TestQuarterEnd:
    """Quarter-end: last 5 calendar days of Mar/Jun/Sep/Dec."""

    def test_march_end(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Mar 29 reflects Mar 28 -> day 28, days_in_month=31, 28 > 26 -> 1
        mar_29 = result.loc["2023-03-29", "cal_quarter_end"]
        assert (mar_29 == 1.0).all(), "Late March should be quarter_end=1"

    def test_non_quarter_month(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Feb 27 reflects Feb 26 -> Feb is not a quarter month -> 0
        feb_27 = result.loc["2023-02-27", "cal_quarter_end"]
        assert (feb_27 == 0.0).all(), "Feb is not a quarter-end month"


class TestYearBoundary:
    """Year boundary: last 5 days of Dec or first 5 days of Jan."""

    def test_late_december(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Dec 28 reflects Dec 27 -> day 27 > 26 -> 1
        dec_28 = result.loc["2023-12-28", "cal_year_boundary"]
        assert (dec_28 == 1.0).all(), "Late December should be year_boundary=1"

    def test_early_january(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Jan 4 reflects Jan 3 -> day 3 <= 5 -> 1
        jan_4 = result.loc["2023-01-04", "cal_year_boundary"]
        assert (jan_4 == 1.0).all(), "Early January should be year_boundary=1"

    def test_mid_year_is_zero(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Jul 15 reflects Jul 14 -> not Dec or Jan -> 0
        jul_15 = result.loc["2023-07-15", "cal_year_boundary"]
        assert (jul_15 == 0.0).all(), "Mid-year should be year_boundary=0"


class TestNfpWeek:
    """NFP week: first 5 calendar days of any month."""

    def test_first_five_days(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Mar 4 reflects Mar 3 -> day 3 <= 5 -> 1
        mar_4 = result.loc["2023-03-04", "cal_nfp_week"]
        assert (mar_4 == 1.0).all(), "First 5 days should be nfp_week=1"

    def test_after_fifth_day(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Mar 8 reflects Mar 7 -> day 7 > 5 -> 0
        mar_8 = result.loc["2023-03-08", "cal_nfp_week"]
        assert (mar_8 == 0.0).all(), "After day 5 should be nfp_week=0"


class TestWeekOfMonth:
    """Week of month: normalized 0..1."""

    def test_values_in_range(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        vals = result["cal_week_of_month"].dropna()
        assert vals.min() >= 0.0, f"week_of_month min={vals.min()}"
        assert vals.max() <= 1.0, f"week_of_month max={vals.max()}"

    def test_first_week_near_zero(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Jan 2 reflects Jan 1 -> day 1 -> (0) / 4 = 0.0
        jan_2 = result.loc["2023-01-02", "cal_week_of_month"]
        assert (jan_2 == 0.0).all(), "First day should be week_of_month=0"


class TestDaysToMonthEnd:
    """Days to month end: normalized 0..1."""

    def test_values_in_range(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        vals = result["cal_days_to_month_end"].dropna()
        assert vals.min() >= 0.0, f"days_to_month_end min={vals.min()}"
        assert vals.max() <= 1.0, f"days_to_month_end max={vals.max()}"

    def test_month_start_near_one(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Jan 2 reflects Jan 1 -> (31-1)/31 ≈ 0.968
        jan_2_vals = result.loc["2023-01-02", "cal_days_to_month_end"]
        assert (jan_2_vals > 0.9).all(), "Month start should be near 1.0"

    def test_month_end_near_zero(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # Feb 1 00:00 reflects Jan 31 23:00 -> day 31, days_in_month=31 -> 0.0
        feb_1_midnight = result.loc["2023-02-01 00:00", "cal_days_to_month_end"]
        assert feb_1_midnight == 0.0, "Last day of month should be 0.0"


class TestFomcProximity:
    """FOMC proximity: cyclic sine feature."""

    def test_values_in_range(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        vals = result["cal_fomc_proximity"].dropna()
        assert vals.min() >= -1.0 - 1e-10, f"fomc min={vals.min()}"
        assert vals.max() <= 1.0 + 1e-10, f"fomc max={vals.max()}"

    def test_is_cyclic(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        vals = result["cal_fomc_proximity"].dropna()
        # Should have both positive and negative values (cyclic)
        assert vals.min() < -0.5, "Should have negative values"
        assert vals.max() > 0.5, "Should have positive values"


class TestCalendarPluginIntegration:
    """Plugin integrates correctly with registry."""

    def test_plugin_importable(self):
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-core:calendar_events")
        assert plugin_cls is not None

    def test_benefits_from_stationary_false(self):
        from fwbg.pipeline import get_registry
        registry = get_registry()
        registry.auto_discover()
        plugin_cls = registry.get("fwbg-core:calendar_events")
        assert plugin_cls.benefits_from_stationary is False

    def test_default_params(self, indicator):
        params = indicator.get_default_params()
        assert "include_proximity" in params
        assert "include_binary" in params
        assert params["include_proximity"] is True
        assert params["include_binary"] is True

    def test_name_and_version(self, indicator):
        assert indicator.name == "calendar_events"
        assert indicator.version == "1.0.0"


class TestCalendarParameterFiltering:
    """Parameters control which features are generated."""

    def test_binary_only(self, indicator, two_year_df):
        result = indicator.compute(two_year_df, include_proximity=False)
        # Binary features should be present
        assert "cal_turn_of_month" in result.columns
        # Proximity features should be absent
        assert "cal_days_to_month_end" not in result.columns
        assert "cal_fomc_proximity" not in result.columns
        assert "cal_week_of_month" not in result.columns

    def test_proximity_only(self, indicator, two_year_df):
        result = indicator.compute(two_year_df, include_binary=False)
        # Proximity features should be present
        assert "cal_days_to_month_end" in result.columns
        assert "cal_fomc_proximity" in result.columns
        # Binary features should be absent
        assert "cal_turn_of_month" not in result.columns
        assert "cal_triple_witching" not in result.columns
