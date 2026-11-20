import inspect
import numpy as np
import pandas as pd
import pytest
from fwbg.plugins import import_plugin_module

_ts = import_plugin_module("fwbg-core", "indicators", "time_season")
if _ts is None:
    pytest.skip("time_season plugin not available", allow_module_level=True)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_df_at_hour(hour, n=10):
    """Hourly bars whose first valid row (row 1; row 0 is NaN) is at the given UTC hour."""
    start = pd.Timestamp("2022-01-03") + pd.Timedelta(hours=hour - 1)
    idx = pd.date_range(start, periods=n, freq="h")
    close = np.full(n, 100.0)
    return pd.DataFrame(
        {"O": close, "H": close + 0.1, "L": close - 0.1, "C": close}, index=idx
    )


def _get_ind(**params):
    for name in dir(_ts):
        obj = getattr(_ts, name)
        if (
            isinstance(obj, type)
            and not inspect.isabstract(obj)
            and hasattr(obj, "compute")
            and hasattr(obj, "get_feature_columns")
        ):
            return obj(**params) if params else obj()
    raise RuntimeError("Could not find indicator class in time_season module")


# ── cyclical encoding ─────────────────────────────────────────────────────────


class TestCyclicalEncoding:
    def test_hour_sin_cos_at_midnight(self):
        """Hour=0: sin(2pi*0/24)=0, cos(2pi*0/24)=1."""
        df = _make_df_at_hour(0)
        result = _get_ind().compute(df)
        valid = result.dropna()
        if "time_hour_sin" in valid.columns:
            assert abs(valid["time_hour_sin"].iloc[0]) < 0.01, (
                f"sin at hour 0 should be 0, got {valid['time_hour_sin'].iloc[0]:.4f}"
            )
        if "time_hour_cos" in valid.columns:
            assert abs(valid["time_hour_cos"].iloc[0] - 1.0) < 0.01, (
                f"cos at hour 0 should be 1, got {valid['time_hour_cos'].iloc[0]:.4f}"
            )

    def test_hour_sin_cos_at_hour_6(self):
        """Hour=6: sin(2pi*6/24)=1, cos(2pi*6/24)=0."""
        df = _make_df_at_hour(6)
        result = _get_ind().compute(df)
        valid = result.dropna()
        if "time_hour_sin" in valid.columns:
            assert abs(valid["time_hour_sin"].iloc[0] - 1.0) < 0.01, (
                f"sin at hour 6 should be 1, got {valid['time_hour_sin'].iloc[0]:.4f}"
            )
        if "time_hour_cos" in valid.columns:
            assert abs(valid["time_hour_cos"].iloc[0]) < 0.01, (
                f"cos at hour 6 should be 0, got {valid['time_hour_cos'].iloc[0]:.4f}"
            )

    def test_hour_sin_cos_at_noon(self):
        """Hour=12: sin(2pi*12/24)=0, cos(2pi*12/24)=-1."""
        df = _make_df_at_hour(12)
        result = _get_ind().compute(df)
        valid = result.dropna()
        if "time_hour_sin" in valid.columns:
            assert abs(valid["time_hour_sin"].iloc[0]) < 0.01, (
                f"sin at hour 12 should be 0, got {valid['time_hour_sin'].iloc[0]:.4f}"
            )
        if "time_hour_cos" in valid.columns:
            assert abs(valid["time_hour_cos"].iloc[0] + 1.0) < 0.01, (
                f"cos at hour 12 should be -1, got {valid['time_hour_cos'].iloc[0]:.4f}"
            )

    def test_hours_23_and_0_are_close_in_circular_space(self):
        """Hours 23 and 0 should be close neighbors on the circular encoding."""
        df23 = _make_df_at_hour(23)
        df0 = _make_df_at_hour(0)
        ind = _get_ind()
        r23 = ind.compute(df23).dropna()
        r0 = ind.compute(df0).dropna()
        if "time_hour_sin" in r23.columns:
            sin_diff = abs(r23["time_hour_sin"].iloc[0] - r0["time_hour_sin"].iloc[0])
            cos_diff = abs(r23["time_hour_cos"].iloc[0] - r0["time_hour_cos"].iloc[0])
            assert sin_diff < 0.3, (
                f"Hour 23 and 0 sin should be close, diff={sin_diff:.3f}"
            )
            assert cos_diff < 0.3, (
                f"Hour 23 and 0 cos should be close, diff={cos_diff:.3f}"
            )


# ── session detection ─────────────────────────────────────────────────────────


class TestSessionDetection:
    def test_london_session_at_10_utc(self):
        """10:00 UTC is London session (8-16 UTC)."""
        df = _make_df_at_hour(10)
        result = _get_ind().compute(df)
        if "time_session_london" in result.columns:
            assert result["time_session_london"].dropna().iloc[0] == 1, (
                "10 UTC should be London session"
            )

    def test_not_london_at_3_utc(self):
        """3:00 UTC is NOT London session."""
        df = _make_df_at_hour(3)
        result = _get_ind().compute(df)
        if "time_session_london" in result.columns:
            assert result["time_session_london"].dropna().iloc[0] == 0, (
                "3 UTC should not be London"
            )

    def test_ny_session_at_15_utc(self):
        """15:00 UTC is NY session (13-21 UTC)."""
        df = _make_df_at_hour(15)
        result = _get_ind().compute(df)
        if "time_session_ny" in result.columns:
            assert result["time_session_ny"].dropna().iloc[0] == 1, (
                "15 UTC should be NY session"
            )

    def test_overlap_at_14_utc(self):
        """14:00 UTC overlaps London (8-16) and NY (13-21)."""
        df = _make_df_at_hour(14)
        result = _get_ind().compute(df)
        if "time_session_overlap" in result.columns:
            assert result["time_session_overlap"].dropna().iloc[0] == 1, (
                "14 UTC should be overlap session"
            )

    def test_asia_session_at_2_utc(self):
        """2:00 UTC is Asia session (0-8 UTC)."""
        df = _make_df_at_hour(2)
        result = _get_ind().compute(df)
        if "time_session_asia" in result.columns:
            assert result["time_session_asia"].dropna().iloc[0] == 1, (
                "2 UTC should be Asia session"
            )

    def test_not_ny_at_6_utc(self):
        """6:00 UTC is not NY session."""
        df = _make_df_at_hour(6)
        result = _get_ind().compute(df)
        if "time_session_ny" in result.columns:
            assert result["time_session_ny"].dropna().iloc[0] == 0, (
                "6 UTC should not be NY session"
            )


# ── calendar features ─────────────────────────────────────────────────────────


class TestCalendarFeatures:
    def test_year_progress_low_in_january(self):
        """time_year_progress near 0 on Jan 2."""
        idx = pd.date_range("2023-01-02 08:00", periods=5, freq="h")
        df = pd.DataFrame({"O": 100.0, "H": 100.5, "L": 99.5, "C": 100.0}, index=idx)
        result = _get_ind().compute(df)
        if "time_year_progress" in result.columns:
            val = result["time_year_progress"].dropna().iloc[0]
            assert val < 0.05, f"Year progress on Jan 2 should be < 0.05, got {val:.4f}"

    def test_year_progress_high_in_december(self):
        """time_year_progress near 1 on Dec 31."""
        idx = pd.date_range("2023-12-31 08:00", periods=5, freq="h")
        df = pd.DataFrame({"O": 100.0, "H": 100.5, "L": 99.5, "C": 100.0}, index=idx)
        result = _get_ind().compute(df)
        if "time_year_progress" in result.columns:
            val = result["time_year_progress"].dropna().iloc[0]
            assert val > 0.95, (
                f"Year progress on Dec 31 should be > 0.95, got {val:.4f}"
            )

    def test_year_progress_increases_over_time(self):
        """Year progress in December > January."""
        idx_jan = pd.date_range("2023-01-02", periods=10, freq="h")
        idx_dec = pd.date_range("2023-12-31", periods=10, freq="h")
        df_jan = pd.DataFrame({"O": 100, "H": 100, "L": 100, "C": 100}, index=idx_jan)
        df_dec = pd.DataFrame({"O": 100, "H": 100, "L": 100, "C": 100}, index=idx_dec)
        ind = _get_ind()
        r_jan = ind.compute(df_jan)
        r_dec = ind.compute(df_dec)
        if "time_year_progress" in r_jan.columns:
            v_jan = r_jan["time_year_progress"].dropna().iloc[0]
            v_dec = r_dec["time_year_progress"].dropna().iloc[0]
            assert v_dec > v_jan, (
                f"Year progress: Dec ({v_dec:.3f}) should > Jan ({v_jan:.3f})"
            )


# ── plugin attributes ─────────────────────────────────────────────────────────


class TestPluginAttributes:
    def test_feature_columns_include_sin_cos(self):
        ind = _get_ind()
        cols = ind.get_feature_columns()
        assert "time_hour_sin" in cols or any("sin" in c for c in cols), (
            "Sin encoding should be declared"
        )

    def test_session_columns_declared(self):
        ind = _get_ind()
        cols = ind.get_feature_columns()
        session_cols = [c for c in cols if "session" in c]
        assert len(session_cols) >= 3, (
            f"Expected >= 3 session columns, got {session_cols}"
        )

    def test_no_inf_values(self):
        n = 500
        idx = pd.date_range("2022-01-03", periods=n, freq="h")
        df = pd.DataFrame({"O": 100, "H": 100.1, "L": 99.9, "C": 100}, index=idx)
        ind = _get_ind()
        result = ind.compute(df)
        for col in ind.get_feature_columns():
            if col in result.columns:
                bad = result[col].isin([float("inf"), float("-inf")])
                assert not bad.any(), f"{col} contains inf"

    def test_only_declared_features_added(self):
        n = 200
        idx = pd.date_range("2022-01-03", periods=n, freq="h")
        df = pd.DataFrame({"O": 100, "H": 100.1, "L": 99.9, "C": 100}, index=idx)
        ind = _get_ind()
        result = ind.compute(df)
        original = set(df.columns)
        new_cols = set(result.columns) - original
        declared = set(ind.get_feature_columns())
        undeclared = new_cols - declared
        assert not undeclared, f"Undeclared columns added: {undeclared}"


# ── timing: a row describes its own timestamp ─────────────────────────────────


def _frame(index) -> pd.DataFrame:
    return pd.DataFrame({"C": 1.0}, index=pd.DatetimeIndex(index))


class TestOwnTimestamp:
    def test_first_row_nan_and_rows_are_not_shifted(self):
        idx = pd.date_range("2023-03-03 22:00", periods=4, freq="h")  # Fri -> Sat
        out = _get_ind().compute(_frame(idx))
        cols = _get_ind().get_feature_columns()
        assert out[cols].iloc[0].isna().all()
        assert out[cols].iloc[1:].notna().all().all()
        # Fri 23:00 is hour 23 on a Friday, Sat 00:00 hour 0 on a Saturday
        assert out["time_hour"].iloc[1:].tolist() == [23.0, 0.0, 1.0]
        assert out["time_day"].iloc[1:].tolist() == [4.0, 5.0, 5.0]
        assert out["time_week_end"].iloc[1:].tolist() == [1.0, 0.0, 0.0]

    def test_monday_row_is_monday_not_friday(self):
        out = _get_ind().compute(_frame(pd.date_range("2023-03-02", "2023-03-07")))
        by_day = out.set_index(out.index.strftime("%a"))
        assert by_day.loc["Mon", "time_week_start"] == 1.0
        assert by_day.loc["Mon", "time_week_end"] == 0.0
        assert by_day.loc["Fri", "time_week_end"] == 1.0

    def test_truncation_invariance_on_every_row(self):
        index = pd.date_range("2023-04-20", "2023-05-12", freq="6h")
        full = _get_ind().compute(_frame(index))
        for cut in range(2, len(index)):
            pd.testing.assert_frame_equal(
                _get_ind().compute(_frame(index[:cut])), full.iloc[:cut]
            )


# ── month ends are counted in trading days ───────────────────────────────────


def _flags(out: pd.DataFrame, col: str) -> list[str]:
    return list(out.index[out[col] == 1.0].strftime("%Y-%m-%d"))


class TestTradingDayMonthEdges:
    def test_month_ending_on_sunday_flags_the_last_three_weekdays(self):
        # 2023-04-30 is a Sunday; the old rule (day >= 28) flagged only Apr 28 and
        # the weekend, February 2021 (ends on a Sunday) was not flagged at all.
        out = _get_ind().compute(_frame(pd.date_range("2023-04-20", "2023-05-08")))
        assert _flags(out, "time_month_end") == [
            "2023-04-26",
            "2023-04-27",
            "2023-04-28",
            "2023-04-29",
            "2023-04-30",
        ]
        assert _flags(out, "time_month_start") == [
            "2023-05-01",
            "2023-05-02",
            "2023-05-03",
        ]

    def test_february_ending_on_a_sunday_is_flagged(self):
        # 2021-02-28 is a Sunday: the last trading day is Fri Feb 26 (< 28)
        out = _get_ind().compute(_frame(pd.date_range("2021-02-20", "2021-03-03")))
        assert "2021-02-26" in _flags(out, "time_month_end")
        assert _flags(out, "time_month_end")[0] == "2021-02-24"

    def test_every_month_end_has_a_flagged_last_weekday_over_a_decade(self):
        out = _get_ind().compute(_frame(pd.date_range("2014-12-01", "2024-12-31")))
        weekdays = out[out.index.dayofweek < 5]
        last = weekdays.groupby(weekdays.index.to_period("M")).tail(1)
        last = last.loc["2015-01-01":"2024-12-31"]
        assert len(last) == 120
        assert (last["time_month_end"] == 1.0).all()
        assert (
            weekdays.groupby(weekdays.index.to_period("M"))["time_month_end"]
            .sum()
            .iloc[1:-1]
            == 3
        ).all()

    def test_exchange_calendar_skips_holidays(self):
        # Good Friday 2024-03-29 is closed on NYSE: Mar 28 is the last trading day
        days = pd.date_range("2024-03-20", "2024-04-05")
        nyse = _get_ind().compute(_frame(days), calendar="NYSE")
        weekdays = _get_ind().compute(_frame(days))
        assert _flags(nyse, "time_month_end")[:3] == [
            "2024-03-26",
            "2024-03-27",
            "2024-03-28",
        ]
        assert _flags(weekdays, "time_month_end")[:3] == [
            "2024-03-27",
            "2024-03-28",
            "2024-03-29",
        ]

    def test_quarter_end_follows_month_end(self):
        out = _get_ind().compute(_frame(pd.date_range("2021-02-20", "2021-03-03")))
        assert _flags(out, "time_quarter_end") == []
        out = _get_ind().compute(_frame(pd.date_range("2021-12-20", "2022-01-05")))
        assert _flags(out, "time_quarter_end")[:3] == [
            "2021-12-29",
            "2021-12-30",
            "2021-12-31",
        ]
        assert _flags(out, "time_quarter_end")[-1] == "2021-12-31"

    def test_month_edge_days_param(self):
        out = _get_ind().compute(
            _frame(pd.date_range("2023-04-20", "2023-05-08")), month_edge_days=1
        )
        assert _flags(out, "time_month_start") == ["2023-05-01"]
        assert _flags(out, "time_month_end") == [
            "2023-04-28",
            "2023-04-29",
            "2023-04-30",
        ]

    def test_unknown_calendar_raises(self):
        with pytest.raises(ValueError, match="unknown calendar"):
            _get_ind().compute(
                _frame(pd.date_range("2023-04-20", "2023-05-08")), calendar="MOON"
            )


# ── ISO week encoding ─────────────────────────────────────────────────────────


class TestIsoWeekEncoding:
    def test_week_53_is_not_week_1(self):
        # 2020 has 53 ISO weeks: Thu 2020-12-31 is in week 53, Mon 2021-01-04 in week 1 of 2021
        idx = pd.DatetimeIndex(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"])
        out = _get_ind().compute(_frame(idx), include_sessions=False)
        assert out["season_week"].iloc[1:].tolist() == [53.0, 1.0, 1.0]
        angle_1 = np.arctan2(
            out["season_week_sin"].iloc[2], out["season_week_cos"].iloc[2]
        )
        # week 53 of the 53-week year 2020 sits at angle 0; week 1 of the 52-week
        # year 2021 is 1/52 of a turn later. The old week / 52 put week 53 on week 1.
        assert out["season_week_cos"].iloc[1] == pytest.approx(1.0)
        assert out["season_week_sin"].iloc[1] == pytest.approx(0.0, abs=1e-12)
        assert angle_1 == pytest.approx(2 * np.pi / 52)
        assert out["season_week_sin"].iloc[1] != pytest.approx(
            out["season_week_sin"].iloc[2]
        )

    def test_52_week_year_unchanged(self):
        # 2021 has 52 ISO weeks: week 26 -> sin(2*pi*26/52)
        out = _get_ind().compute(
            _frame(pd.DatetimeIndex(["2021-06-27", "2021-06-30"])),
            include_sessions=False,
        )
        assert out["season_week"].iloc[1] == 26.0
        assert out["season_week_cos"].iloc[1] == pytest.approx(
            np.cos(2 * np.pi * 26 / 52)
        )

    def test_iso_weeks_in_year(self):
        weeks = _ts.iso_weeks_in_year([2015, 2016, 2020, 2021, 2026])
        assert weeks.tolist() == [53.0, 52.0, 53.0, 52.0, 53.0]


# ── session time zones ────────────────────────────────────────────────────────


class TestSessionTimeZones:
    def test_ny_session_follows_daylight_saving_time(self):
        # NY cash session 9-17 local: 14-22 UTC in winter (EST), 13-21 UTC in summer (EDT)
        sessions = {"ny": [9, 17, "America/New_York"]}
        winter = _frame(pd.date_range("2023-01-17 12:00", periods=12, freq="h"))
        summer = _frame(pd.date_range("2023-07-18 12:00", periods=12, freq="h"))
        w = _get_ind().compute(winter, sessions=sessions, include_seasonality=False)
        s = _get_ind().compute(summer, sessions=sessions, include_seasonality=False)
        assert w.loc[w["time_session_ny"] == 1].index.hour.tolist() == list(
            range(14, 22)
        )
        assert s.loc[s["time_session_ny"] == 1].index.hour.tolist() == list(
            range(13, 21)
        )

    def test_plain_sessions_stay_in_utc(self):
        idx = pd.date_range("2023-07-18 12:00", periods=12, freq="h")
        out = _get_ind().compute(_frame(idx), sessions={"ny": [13, 21]})
        assert out.loc[out["time_session_ny"] == 1].index.hour.tolist() == list(
            range(13, 21)
        )

    def test_tz_aware_index_is_converted(self):
        idx = pd.date_range(
            "2023-07-18 12:00", periods=12, freq="h", tz="UTC"
        ).tz_convert("Asia/Tokyo")
        out = _get_ind().compute(
            _frame(idx),
            sessions={"ny": [9, 17, "America/New_York"]},
            include_seasonality=False,
        )
        assert len(out.loc[out["time_session_ny"] == 1]) == 8

    def test_unknown_zone_and_bad_spec_raise(self):
        idx = _frame(pd.date_range("2023-07-18 12:00", periods=4, freq="h"))
        with pytest.raises(ValueError, match="time zone"):
            _get_ind().compute(idx, sessions={"x": [9, 17, "Mars/Olympus"]})
        with pytest.raises(ValueError, match="must be"):
            _get_ind().compute(idx, sessions={"x": [9]})
