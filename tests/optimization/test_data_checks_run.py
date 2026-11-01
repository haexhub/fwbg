"""Integration: a run's loading step applies the strategy's data checks."""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.config import StrategyConfig
from fwbg.data import trading_hours
from fwbg.data.validation import DataCheckError
from fwbg.optimization import process
from fwbg.optimization.process import load_symbol_frame, process_symbol


def around_the_clock(days: int = 14) -> pd.DataFrame:
    """Hourly EURUSD bars including weekends, with a few flat zero-volume filler bars."""
    idx = pd.date_range("2026-01-05", periods=days * 24, freq="h")
    c = 1.1 + np.cumsum(np.random.default_rng(3).normal(0, 0.0005, len(idx)))
    df = pd.DataFrame({"O": c, "H": c + 0.0004, "L": c - 0.0004, "C": c, "V": 10.0}, index=idx)
    filler = idx[(idx >= "2026-01-10 00:00") & (idx < "2026-01-10 03:00")]
    df.loc[filler, ["O", "H", "L", "C"]] = 1.1
    df.loc[filler, "V"] = 0.0
    return df


@pytest.fixture
def weekend_data(monkeypatch):
    monkeypatch.setattr(process, "load_data_aligned", lambda path: around_the_clock())
    monkeypatch.setattr(process.data_config, "TIMEFRAME", "HOUR_1")
    monkeypatch.setattr(trading_hours, "load_asset_overrides", lambda: {})


def test_strict_mode_stops_on_bars_outside_the_trading_hours(weekend_data):
    with pytest.raises(DataCheckError) as exc:
        load_symbol_frame("EURUSD_HOUR_1.csv", StrategyConfig(), "EURUSD")
    assert "outside_trading_hours" in {f.code for f in exc.value.report.errors}


def test_warn_mode_runs_without_the_filler_bars_and_keeps_the_report(weekend_data):
    df = load_symbol_frame("EURUSD_HOUR_1.csv", StrategyConfig(data_checks="warn"), "EURUSD")
    assert len(df) == len(around_the_clock()) - 3
    report = df.attrs["data_checks"]
    assert [f["code"] for f in report["errors"]] == ["outside_trading_hours"]
    assert [f["code"] for f in report["infos"]][0] == "filler_bars_dropped"


def test_off_mode_leaves_the_data_alone(weekend_data):
    df = load_symbol_frame("EURUSD_HOUR_1.csv", StrategyConfig(data_checks="off"), "EURUSD")
    pd.testing.assert_frame_equal(df, around_the_clock())
    assert "data_checks" not in df.attrs


def test_process_symbol_reports_data_invalid(weekend_data):
    result = process_symbol("EURUSD_HOUR_1.csv", StrategyConfig())
    assert result["status"] == "data_invalid"
    assert result["data_checks"]["errors"][0]["code"] == "outside_trading_hours"


@pytest.mark.parametrize("bad", ["loose", None, 1])
def test_data_checks_mode_is_validated(bad):
    with pytest.raises(ValueError, match="data_checks"):
        StrategyConfig.from_dict({"name": "x", "data_checks": bad})
