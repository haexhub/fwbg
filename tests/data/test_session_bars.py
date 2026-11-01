"""Tests for fwbg.data.session_bars and `fwbg data session-bars`."""

import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from fwbg.data import trading_hours
from fwbg.data.session_bars import build_session_bars
from fwbg.data.trading_hours import parse_trading_hours
from fwbg.data.validation import DataCheckError

# ``fwbg.cli`` re-exports ``main()``, which shadows the module of the same name.
cli_main = importlib.import_module("fwbg.cli.main")
CASH = parse_trading_hours(
    {"tz": "America/New_York", "windows": [{"days": "Mon-Fri", "start": "09:30", "end": "16:00"}]}
)
GLOBEX = parse_trading_hours(
    {"tz": "America/New_York", "windows": [{"days": "Sun-Thu", "start": "18:00", "end": "17:00"}]}
)


def m15(start: str, end: str) -> pd.DataFrame:
    """15-minute UTC bars, close = 1000 + minutes since start / 100, volume 1."""
    idx = pd.date_range(start, end, freq="15min", inclusive="left")
    c = 1000 + np.arange(len(idx)) * 0.15
    return pd.DataFrame({"O": c - 0.05, "H": c + 0.1, "L": c - 0.1, "C": c, "V": 1.0}, index=idx)


def test_cash_bar_ends_at_the_cash_close_not_at_midnight():
    # Monday 2026-03-02 (EST): the whole day in UTC, so 23:45 New York exists too
    df = m15("2026-03-02 05:00", "2026-03-03 05:00")
    df.loc["2026-03-03 04:45", ["H", "C"]] = 5000.0  # the 23:45-24:00 New York bar
    built = build_session_bars(df, symbol="NAS100", hours=CASH, bar_minutes=15)
    bar = built.bars.loc["2026-03-02"]
    session = df.loc["2026-03-02 14:30":"2026-03-02 20:45"]  # 09:30 .. 15:45 New York
    assert bar["O"] == session["O"].iloc[0] and bar["C"] == session["C"].iloc[-1]
    assert bar["H"] == session["H"].max() and bar["H"] < 5000.0
    assert bar["V"] == len(session) == 26


def test_incomplete_sessions_are_dropped():
    df = m15("2026-03-02 05:00", "2026-03-05 05:00")
    early_close = (df.index >= "2026-03-03 18:00") & (
        df.index < "2026-03-04 05:00"
    )  # Tue closes 13:00 NY
    late_open = (df.index >= "2026-03-04 05:00") & (
        df.index < "2026-03-04 15:00"
    )  # Wed opens 10:00 NY
    built = build_session_bars(
        df[~early_close & ~late_open], symbol="NAS100", hours=CASH, bar_minutes=15
    )
    assert list(built.bars.index.strftime("%Y-%m-%d")) == ["2026-03-02"]
    assert (built.sessions, built.incomplete) == (3, 2)


def test_overnight_session_is_labelled_with_its_close_date_across_dst():
    # Globex: Sunday 18:00 -> Monday 17:00 New York; US clocks change on 2026-03-08
    df = m15("2026-03-08 00:00", "2026-03-10 00:00")
    built = build_session_bars(df, symbol="NAS100", hours=GLOBEX, bar_minutes=15)
    assert list(built.bars.index.strftime("%Y-%m-%d")) == ["2026-03-09"]
    monday = df.loc["2026-03-08 22:00":"2026-03-09 20:45"]  # 18:00 EDT .. 16:45 EDT
    assert built.bars["C"].iloc[0] == monday["C"].iloc[-1]


def test_filler_bars_do_not_make_a_holiday_session():
    df = m15("2026-03-02 05:00", "2026-03-04 05:00")
    holiday = df.index >= "2026-03-03 05:00"
    df.loc[holiday, ["O", "H", "L", "C"]] = 1000.0
    df.loc[holiday, "V"] = 0.0
    built = build_session_bars(df, symbol="NAS100", hours=CASH, bar_minutes=15)
    assert list(built.bars.index.strftime("%Y-%m-%d")) == ["2026-03-02"]
    assert built.source_checks.to_dict()["infos"][0]["code"] == "filler_bars_dropped"


def test_bad_source_and_daily_source_raise():
    df = m15("2026-03-02 05:00", "2026-03-03 05:00")
    with pytest.raises(DataCheckError):
        build_session_bars(
            pd.concat([df, df.iloc[[3]]]), symbol="NAS100", hours=CASH, bar_minutes=15
        )
    with pytest.raises(ValueError, match="intraday"):
        build_session_bars(df, symbol="NAS100", hours=CASH, bar_minutes=1440)


def test_cli_writes_a_datasource_with_provenance(tmp_path, monkeypatch):
    source = tmp_path / "NAS100_MINUTE_15.csv"
    frame = m15("2026-03-02 05:00", "2026-03-04 05:00")
    frame.index.name = "T"
    frame.to_csv(source, date_format="%Y-%m-%d %H:%M:%S")
    root = tmp_path / "data"
    monkeypatch.setattr("fwbg.core.data_sources.get_data_root", lambda: root)
    monkeypatch.setattr(trading_hours, "load_asset_overrides", lambda: {})
    monkeypatch.setattr(
        sys,
        "argv",
        ["fwbg", "data", "session-bars", "--input", str(source), "--datasource", "cash"],
    )
    cli_main.main()
    out = root / "cash" / "datasource" / "NAS100_DAY_1.csv"
    bars = pd.read_csv(out, index_col="T")
    assert list(bars.index) == ["2026-03-02 00:00:00", "2026-03-03 00:00:00"]
    provenance = json.loads(out.with_suffix(".session.json").read_text())
    assert provenance["session"] == "cash" and provenance["bars"] == 2
    assert json.loads((root / "cash" / "config.json").read_text())["path"] == str(out.parent)


def test_cli_rejects_an_unknown_session(tmp_path, monkeypatch):
    source = tmp_path / "EURUSD_MINUTE_15.csv"
    m15("2026-03-02 05:00", "2026-03-02 06:00").rename_axis("T").to_csv(source)
    monkeypatch.setattr(trading_hours, "load_asset_overrides", lambda: {})
    monkeypatch.setattr(
        sys, "argv", ["fwbg", "data", "session-bars", "--input", str(source), "--datasource", "x"]
    )
    with pytest.raises(SystemExit, match="no 'cash' session"):
        cli_main.main()
