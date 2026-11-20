"""Tests for calendar anomaly indicator plugin."""

import holidays
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

_cal = import_plugin_module("fwbg-core", "indicators", "calendar_events")
if _cal is None:
    pytest.skip(
        "fwbg-core calendar_events plugin not available", allow_module_level=True
    )


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


def daily(start: str, end: str) -> pd.DataFrame:
    """Daily bars (every calendar day) from *start* to *end*."""
    idx = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"C": 1.0}, index=idx)


def flagged(result: pd.DataFrame, col: str) -> list[str]:
    """Dates (YYYY-MM-DD) where *col* is 1.0."""
    return list(result.index[result[col] == 1.0].strftime("%Y-%m-%d"))


class TestCalendarFeatureColumns:
    """All expected columns are produced."""

    def test_all_columns_present(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        for col in indicator.get_feature_columns():
            assert col in result.columns, f"Missing: {col}"

    def test_feature_count(self, indicator):
        assert len(indicator.get_feature_columns()) == 8

    def test_fomc_proximity_is_gone(self, indicator, two_year_df):
        """The old column was sin(2*pi*doy/46), unrelated to any FOMC date."""
        assert "cal_fomc_proximity" not in indicator.get_feature_columns()
        assert "cal_fomc_proximity" not in indicator.compute(two_year_df).columns

    def test_returns_dataframe(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_original_columns(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        for col in ["O", "H", "L", "C", "V"]:
            assert col in result.columns


class TestCalendarTiming:
    """Calendar features describe the bar's own date (known in advance)."""

    def test_first_row_is_nan(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        for col in indicator.get_feature_columns():
            assert pd.isna(result[col].iloc[0]), f"{col} first row should be NaN"
            assert result[col].iloc[1:].notna().all(), f"{col} has later NaN"

    def test_row_describes_its_own_timestamp_not_the_previous_bar(self, indicator):
        idx = pd.date_range("2023-12-31 22:00", periods=5, freq="h")
        df = pd.DataFrame({"C": 1.0}, index=idx)
        result = indicator.compute(df)
        # Dec 31 23:00 is a year-boundary bar; Jan 1 00:00 and later are too
        assert result["cal_year_boundary"].iloc[1:].tolist() == [1.0] * 4
        # Jan 1 02:00 is a new month: week_of_month 0, days_to_month_end (31-1)/31
        assert result["cal_week_of_month"].iloc[-1] == 0.0
        assert result["cal_days_to_month_end"].iloc[-1] == pytest.approx(30 / 31)
        # Dec 31 23:00 is the last day of December
        assert result["cal_days_to_month_end"].iloc[1] == 0.0

    def test_monday_does_not_inherit_fridays_value(self, indicator):
        # Fri 2023-03-03 is the NFP day, Mon 2023-03-06 starts the next week
        result = indicator.compute(daily("2023-03-01", "2023-03-08")).set_index(
            pd.date_range("2023-03-01", "2023-03-08")
        )
        assert result.loc["2023-03-03", "cal_nfp_week"] == 1.0
        assert result.loc["2023-03-06", "cal_nfp_week"] == 0.0

    def test_truncation_invariance_on_every_row(self, indicator):
        full = indicator.compute(daily("2023-12-15", "2024-03-20"))
        for cut in range(2, len(full)):
            part = indicator.compute(daily("2023-12-15", "2024-03-20").iloc[:cut])
            pd.testing.assert_frame_equal(part, full.iloc[:cut])


class TestTurnOfMonth:
    """Turn-of-month: last 2 and first 3 trading days (NYSE calendar)."""

    def test_new_year_holiday_first_three_trading_days(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        # 2023-01-02 is an NYSE holiday: the first trading days are Jan 3, 4 and 5
        assert (result.loc["2023-01-03", "cal_turn_of_month"] == 1.0).all()
        assert (result.loc["2023-01-05", "cal_turn_of_month"] == 1.0).all()
        assert (result.loc["2023-01-06", "cal_turn_of_month"] == 0.0).all()

    def test_mid_month_is_zero(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        assert (result.loc["2023-01-16", "cal_turn_of_month"] == 0.0).all()

    def test_month_ending_on_a_sunday_flags_the_last_friday(self, indicator):
        # 2023-04-30 is a Sunday: the last trading days are Thu Apr 27 and Fri Apr 28.
        # The old calendar-day rule (day >= 29) missed both.
        result = indicator.compute(daily("2023-04-20", "2023-05-08"))
        assert flagged(result, "cal_turn_of_month") == [
            "2023-04-27",
            "2023-04-28",
            "2023-04-29",  # weekend after the last trading day keeps its flag
            "2023-04-30",
            "2023-05-01",
            "2023-05-02",
            "2023-05-03",
        ]

    def test_good_friday_month_end(self, indicator):
        # 2024-03-29 (Good Friday) is closed: the last trading days are Mar 27 and 28
        result = indicator.compute(daily("2024-03-20", "2024-04-05"))
        assert flagged(result, "cal_turn_of_month")[:3] == [
            "2024-03-27",
            "2024-03-28",
            "2024-03-29",
        ]
        assert "2024-03-26" not in flagged(result, "cal_turn_of_month")

    def test_every_months_last_trading_day_is_flagged(self, indicator):
        result = indicator.compute(daily("2014-12-01", "2025-01-31"))
        closed = holidays.financial_holidays("NYSE", years=range(2014, 2025))
        open_days = pd.bdate_range("2015-01-01", "2024-12-31")
        open_days = open_days[~open_days.isin(pd.DatetimeIndex(list(closed.keys())))]
        last_per_month = (
            pd.Series(open_days, index=open_days)
            .groupby(open_days.to_period("M"))
            .max()
        )
        assert len(last_per_month) == 120
        flags = result["cal_turn_of_month"]
        assert (flags.loc[last_per_month.to_numpy()] == 1.0).all()

    def test_weekdays_calendar_ignores_exchange_holidays(self, indicator):
        result = indicator.compute(
            daily("2024-03-20", "2024-04-05"), calendar="weekdays"
        )
        # Good Friday is a weekday here, so it is the last trading day of March
        assert flagged(result, "cal_turn_of_month")[:3] == [
            "2024-03-28",
            "2024-03-29",
            "2024-03-30",
        ]

    def test_unknown_calendar_raises(self, indicator):
        with pytest.raises(ValueError, match="unknown calendar"):
            indicator.compute(daily("2024-03-20", "2024-04-05"), calendar="MOON")

    def test_intraday_bars_repeat_the_days_flag(self, indicator):
        idx = pd.date_range("2023-04-27 00:00", "2023-04-28 23:00", freq="h")
        result = indicator.compute(pd.DataFrame({"C": 1.0}, index=idx))
        assert (result["cal_turn_of_month"].iloc[1:] == 1.0).all()


class TestTripleWitching:
    """Triple witching: within 2 days of 3rd Friday of Mar/Jun/Sep/Dec."""

    def test_march_2023_triple_witching(self, indicator, two_year_df):
        """3rd Friday of March 2023 is March 17."""
        result = indicator.compute(two_year_df)
        assert (result.loc["2023-03-17", "cal_triple_witching"] == 1.0).all()
        assert (result.loc["2023-03-15", "cal_triple_witching"] == 1.0).all()
        assert (result.loc["2023-03-14", "cal_triple_witching"] == 0.0).all()

    def test_non_witching_month_is_zero(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        assert (result.loc["2023-02-17", "cal_triple_witching"] == 0.0).all()

    def test_june_2023_triple_witching(self, indicator, two_year_df):
        """3rd Friday of June 2023 is June 16."""
        result = indicator.compute(two_year_df)
        assert (result.loc["2023-06-16", "cal_triple_witching"] == 1.0).all()


class TestMonthlyOpex:
    """Monthly OpEx: within 2 days of 3rd Friday of every month."""

    def test_january_opex(self, indicator, two_year_df):
        """3rd Friday of Jan 2023 is Jan 20."""
        result = indicator.compute(two_year_df)
        assert (result.loc["2023-01-20", "cal_monthly_opex"] == 1.0).all()
        assert (result.loc["2023-01-18", "cal_monthly_opex"] == 1.0).all()
        assert (result.loc["2023-01-17", "cal_monthly_opex"] == 0.0).all()

    def test_far_from_opex_is_zero(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        assert (result.loc["2023-01-10", "cal_monthly_opex"] == 0.0).all()


class TestQuarterEnd:
    """Quarter-end: last 5 calendar days of Mar/Jun/Sep/Dec."""

    def test_march_end(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        assert (result.loc["2023-03-28", "cal_quarter_end"] == 1.0).all()
        assert (result.loc["2023-03-26", "cal_quarter_end"] == 0.0).all()

    def test_non_quarter_month(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        assert (result.loc["2023-02-27", "cal_quarter_end"] == 0.0).all()


class TestYearBoundary:
    """Year boundary: last 5 days of Dec or first 5 days of Jan."""

    def test_late_december(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        assert (result.loc["2023-12-27", "cal_year_boundary"] == 1.0).all()
        assert (result.loc["2023-12-26", "cal_year_boundary"] == 0.0).all()

    def test_early_january(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        assert (result.loc["2023-01-05", "cal_year_boundary"] == 1.0).all()
        assert (result.loc["2023-01-06", "cal_year_boundary"] == 0.0).all()

    def test_mid_year_is_zero(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        assert (result.loc["2023-07-14", "cal_year_boundary"] == 0.0).all()


class TestNfpWeek:
    """NFP week: Monday-Friday of the week containing the month's first Friday."""

    def test_week_with_first_friday_spans_the_month_boundary(self, indicator):
        # First Friday of March 2023 is Mar 3: the week starts on Mon Feb 27
        result = indicator.compute(daily("2023-02-20", "2023-03-12"))
        assert flagged(result, "cal_nfp_week") == [
            "2023-02-27",
            "2023-02-28",
            "2023-03-01",
            "2023-03-02",
            "2023-03-03",
        ]

    def test_first_friday_on_the_seventh(self, indicator):
        # Sep 1 2018 is a Saturday: the first Friday is Sep 7. The old rule
        # (day <= 5) flagged Sat Sep 1 .. Wed Sep 5 and missed Sep 6 and 7.
        assert pd.Timestamp("2018-09-07").dayofweek == 4
        result = indicator.compute(daily("2018-08-27", "2018-09-16"))
        assert flagged(result, "cal_nfp_week") == [
            "2018-09-03",
            "2018-09-04",
            "2018-09-05",
            "2018-09-06",
            "2018-09-07",
        ]

    def test_first_friday_on_the_first(self, indicator):
        # Jul 1 2022 is a Friday: its week starts on Mon Jun 27
        result = indicator.compute(daily("2022-06-20", "2022-07-10"))
        assert flagged(result, "cal_nfp_week") == [
            "2022-06-27",
            "2022-06-28",
            "2022-06-29",
            "2022-06-30",
            "2022-07-01",
        ]

    def test_matches_first_friday_of_every_month_over_a_decade(self, indicator):
        # starts on a Sunday so the first row (NaN) is not part of an NFP week
        result = indicator.compute(daily("2014-11-30", "2024-12-31"))
        first_fridays = pd.date_range("2014-12-01", "2025-01-31", freq="WOM-1FRI")
        expected = sorted(
            day
            for ff in first_fridays
            for k in range(5)
            if (day := (ff - pd.Timedelta(days=k)).strftime("%Y-%m-%d")) <= "2024-12-31"
        )
        assert len(first_fridays) == 122
        assert flagged(result, "cal_nfp_week") == expected


class TestWeekOfMonth:
    """Week of month: normalized 0..1."""

    def test_values_in_range(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        vals = result["cal_week_of_month"].dropna()
        assert vals.min() >= 0.0, f"week_of_month min={vals.min()}"
        assert vals.max() <= 1.0, f"week_of_month max={vals.max()}"

    def test_first_week_near_zero(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        assert (result.loc["2023-01-02", "cal_week_of_month"] == 0.0).all()


class TestDaysToMonthEnd:
    """Days to month end: normalized 0..1."""

    def test_values_in_range(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        vals = result["cal_days_to_month_end"].dropna()
        assert vals.min() >= 0.0, f"days_to_month_end min={vals.min()}"
        assert vals.max() <= 1.0, f"days_to_month_end max={vals.max()}"

    def test_month_start_near_one(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        jan_2_vals = result.loc["2023-01-02", "cal_days_to_month_end"]
        assert (jan_2_vals > 0.9).all(), "Month start should be near 1.0"

    def test_month_end_is_zero(self, indicator, two_year_df):
        result = indicator.compute(two_year_df)
        assert (result.loc["2023-01-31", "cal_days_to_month_end"] == 0.0).all()


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
        assert indicator.version == "1.1.0"


class TestCalendarParameterFiltering:
    """Parameters control which features are generated."""

    def test_binary_only(self, indicator, two_year_df):
        result = indicator.compute(two_year_df, include_proximity=False)
        # Binary features should be present
        assert "cal_turn_of_month" in result.columns
        # Proximity features should be absent
        assert "cal_days_to_month_end" not in result.columns
        assert "cal_week_of_month" not in result.columns

    def test_proximity_only(self, indicator, two_year_df):
        result = indicator.compute(two_year_df, include_binary=False)
        # Proximity features should be present
        assert "cal_days_to_month_end" in result.columns
        # Binary features should be absent
        assert "cal_turn_of_month" not in result.columns
        assert "cal_triple_witching" not in result.columns
