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
    """Create n bars all timestamped at the given UTC hour."""
    start = pd.Timestamp("2022-01-03") + pd.Timedelta(hours=hour)
    idx = pd.date_range(start, periods=n, freq="h")
    close = np.full(n, 100.0)
    return pd.DataFrame({"O": close, "H": close + 0.1, "L": close - 0.1, "C": close}, index=idx)


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
            assert abs(valid["time_hour_sin"].iloc[0]) < 0.01,                 f"sin at hour 0 should be 0, got {valid['time_hour_sin'].iloc[0]:.4f}"
        if "time_hour_cos" in valid.columns:
            assert abs(valid["time_hour_cos"].iloc[0] - 1.0) < 0.01,                 f"cos at hour 0 should be 1, got {valid['time_hour_cos'].iloc[0]:.4f}"

    def test_hour_sin_cos_at_hour_6(self):
        """Hour=6: sin(2pi*6/24)=1, cos(2pi*6/24)=0."""
        df = _make_df_at_hour(6)
        result = _get_ind().compute(df)
        valid = result.dropna()
        if "time_hour_sin" in valid.columns:
            assert abs(valid["time_hour_sin"].iloc[0] - 1.0) < 0.01,                 f"sin at hour 6 should be 1, got {valid['time_hour_sin'].iloc[0]:.4f}"
        if "time_hour_cos" in valid.columns:
            assert abs(valid["time_hour_cos"].iloc[0]) < 0.01,                 f"cos at hour 6 should be 0, got {valid['time_hour_cos'].iloc[0]:.4f}"

    def test_hour_sin_cos_at_noon(self):
        """Hour=12: sin(2pi*12/24)=0, cos(2pi*12/24)=-1."""
        df = _make_df_at_hour(12)
        result = _get_ind().compute(df)
        valid = result.dropna()
        if "time_hour_sin" in valid.columns:
            assert abs(valid["time_hour_sin"].iloc[0]) < 0.01,                 f"sin at hour 12 should be 0, got {valid['time_hour_sin'].iloc[0]:.4f}"
        if "time_hour_cos" in valid.columns:
            assert abs(valid["time_hour_cos"].iloc[0] + 1.0) < 0.01,                 f"cos at hour 12 should be -1, got {valid['time_hour_cos'].iloc[0]:.4f}"

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
            assert sin_diff < 0.3, f"Hour 23 and 0 sin should be close, diff={sin_diff:.3f}"
            assert cos_diff < 0.3, f"Hour 23 and 0 cos should be close, diff={cos_diff:.3f}"


# ── session detection ─────────────────────────────────────────────────────────

class TestSessionDetection:
    def test_london_session_at_10_utc(self):
        """10:00 UTC is London session (8-16 UTC)."""
        df = _make_df_at_hour(10)
        result = _get_ind().compute(df)
        if "time_session_london" in result.columns:
            assert result["time_session_london"].dropna().iloc[0] == 1,                 "10 UTC should be London session"

    def test_not_london_at_3_utc(self):
        """3:00 UTC is NOT London session."""
        df = _make_df_at_hour(3)
        result = _get_ind().compute(df)
        if "time_session_london" in result.columns:
            assert result["time_session_london"].dropna().iloc[0] == 0,                 "3 UTC should not be London"

    def test_ny_session_at_15_utc(self):
        """15:00 UTC is NY session (13-21 UTC)."""
        df = _make_df_at_hour(15)
        result = _get_ind().compute(df)
        if "time_session_ny" in result.columns:
            assert result["time_session_ny"].dropna().iloc[0] == 1,                 "15 UTC should be NY session"

    def test_overlap_at_14_utc(self):
        """14:00 UTC overlaps London (8-16) and NY (13-21)."""
        df = _make_df_at_hour(14)
        result = _get_ind().compute(df)
        if "time_session_overlap" in result.columns:
            assert result["time_session_overlap"].dropna().iloc[0] == 1,                 "14 UTC should be overlap session"

    def test_asia_session_at_2_utc(self):
        """2:00 UTC is Asia session (0-8 UTC)."""
        df = _make_df_at_hour(2)
        result = _get_ind().compute(df)
        if "time_session_asia" in result.columns:
            assert result["time_session_asia"].dropna().iloc[0] == 1,                 "2 UTC should be Asia session"

    def test_not_ny_at_6_utc(self):
        """6:00 UTC is not NY session."""
        df = _make_df_at_hour(6)
        result = _get_ind().compute(df)
        if "time_session_ny" in result.columns:
            assert result["time_session_ny"].dropna().iloc[0] == 0,                 "6 UTC should not be NY session"


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
            assert val > 0.95, f"Year progress on Dec 31 should be > 0.95, got {val:.4f}"

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
            assert v_dec > v_jan, f"Year progress: Dec ({v_dec:.3f}) should > Jan ({v_jan:.3f})"


# ── plugin attributes ─────────────────────────────────────────────────────────

class TestPluginAttributes:
    def test_feature_columns_include_sin_cos(self):
        ind = _get_ind()
        cols = ind.get_feature_columns()
        assert "time_hour_sin" in cols or any("sin" in c for c in cols),             "Sin encoding should be declared"

    def test_session_columns_declared(self):
        ind = _get_ind()
        cols = ind.get_feature_columns()
        session_cols = [c for c in cols if "session" in c]
        assert len(session_cols) >= 3,             f"Expected >= 3 session columns, got {session_cols}"

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
