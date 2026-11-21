"""Tests for POST /api/data/ensure and GET /api/data/ensure/{task_id}."""

import pytest
from fastapi.testclient import TestClient

from fwbg.api import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_ensure_unknown_task_returns_404(client):
    resp = client.get("/api/data/ensure/nosuchid")
    assert resp.status_code == 404


def test_ensure_invalid_timeframe_returns_422(client):
    resp = client.post(
        "/api/data/ensure", json={"symbol": "EURUSD", "timeframe": "BAD_TF"}
    )
    assert resp.status_code == 422
    assert "timeframe" in resp.json()["detail"].lower()


def test_ensure_non_dukascopy_symbol_returns_404(client, monkeypatch):
    from fwbg.api import data as data_mod

    monkeypatch.setattr(
        data_mod, "_find_existing_file", lambda s, t, d=None, e=None: None
    )
    resp = client.post(
        "/api/data/ensure", json={"symbol": "FAKEXYZ999", "timeframe": "HOUR_1"}
    )
    assert resp.status_code == 404
    assert "Dukascopy" in resp.json()["detail"]


def test_ensure_returns_ready_when_file_exists(client, monkeypatch, tmp_path):
    from fwbg.api import data as data_mod

    fake_path = tmp_path / "EURUSD_HOUR_1.csv"
    fake_path.write_text("T,O,H,L,C,V\n")
    monkeypatch.setattr(
        data_mod,
        "_find_existing_file",
        lambda s, t, d=None, e=None: ("src1", fake_path),
    )

    resp = client.post(
        "/api/data/ensure", json={"symbol": "EURUSD", "timeframe": "HOUR_1"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["symbol"] == "EURUSD"
    assert body["source"] == "src1"
    assert body["path"] == str(fake_path)


def test_ensure_normalises_symbol(client, monkeypatch, tmp_path):
    from fwbg.api import data as data_mod

    fake_path = tmp_path / "EURUSD_HOUR_1.csv"
    fake_path.write_text("T,O,H,L,C,V\n")
    seen: dict = {}

    def _fake_find(s, t, d=None, e=None):
        seen["symbol"] = s
        return ("src1", fake_path)

    monkeypatch.setattr(data_mod, "_find_existing_file", _fake_find)
    client.post("/api/data/ensure", json={"symbol": "EUR/USD", "timeframe": "HOUR_1"})
    assert seen["symbol"] == "EURUSD"


def test_ensure_no_csv_source_returns_503(client, monkeypatch):
    from fwbg.api import data as data_mod

    monkeypatch.setattr(
        data_mod, "_find_existing_file", lambda s, t, d=None, e=None: None
    )
    monkeypatch.setattr(data_mod, "_first_csv_source", lambda: None)
    resp = client.post(
        "/api/data/ensure", json={"symbol": "EURUSD", "timeframe": "HOUR_1"}
    )
    assert resp.status_code == 503


def test_ensure_starts_download_returns_202_and_task_is_pollable(
    client, monkeypatch, tmp_path
):
    from fwbg.core.data_sources import CSVSourceConfig
    from fwbg.api import data as data_mod
    import fwbg.data.dukascopy as dk_mod

    fake_source = CSVSourceConfig(name="dukas_src", path=tmp_path)
    monkeypatch.setattr(
        data_mod, "_find_existing_file", lambda s, t, d=None, e=None: None
    )
    monkeypatch.setattr(data_mod, "_first_csv_source", lambda: fake_source)
    monkeypatch.setattr(
        dk_mod,
        "download",
        lambda out, symbols, timeframe, start, end, **kw: [
            {
                "symbol": symbols[0],
                "file": str(out / f"{symbols[0]}_{timeframe}.csv"),
                "rows": 42,
                "spread": 0.0,
            }
        ],
    )

    resp = client.post(
        "/api/data/ensure",
        json={
            "symbol": "EURUSD",
            "timeframe": "HOUR_1",
            "date_from": "2023-01-01",
            "date_to": "2023-03-01",
        },
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "downloading"
    assert body["symbol"] == "EURUSD"
    assert body["source"] == "dukas_src"
    assert "task_id" in body
    assert "poll_url" in body

    # Task must be immediately pollable.
    poll = client.get(f"/api/data/ensure/{body['task_id']}")
    assert poll.status_code == 200
    assert poll.json()["task_id"] == body["task_id"]


def test_ensure_date_to_before_date_from_returns_422(client, monkeypatch, tmp_path):
    from fwbg.core.data_sources import CSVSourceConfig
    from fwbg.api import data as data_mod

    monkeypatch.setattr(
        data_mod, "_find_existing_file", lambda s, t, d=None, e=None: None
    )
    monkeypatch.setattr(
        data_mod,
        "_first_csv_source",
        lambda: CSVSourceConfig(name="src", path=tmp_path),
    )
    resp = client.post(
        "/api/data/ensure",
        json={
            "symbol": "EURUSD",
            "timeframe": "HOUR_1",
            "date_from": "2023-06-01",
            "date_to": "2023-01-01",
        },
    )
    assert resp.status_code == 422
    assert "date_to" in resp.json()["detail"].lower()


# ── Cache-coverage: implicit full-history accepts any existing file ─────────


def test_ensure_implicit_request_accepts_partial_cache(client, monkeypatch, tmp_path):
    """A request without date_from must NOT force a full-history re-download when
    a file already exists — _find_existing_file is called with date_from=None so
    the coverage check is skipped (else a 2012 cache triggers a 2003→now refetch
    that stalls the whole run)."""
    from fwbg.api import data as data_mod

    fake_path = tmp_path / "GBPUSD_MINUTE_15.csv"
    fake_path.write_text("T,O,H,L,C,V\n")
    seen: dict = {}

    def _fake_find(s, t, d=None, e=None):
        seen["date_from"] = d
        return ("src1", fake_path)

    monkeypatch.setattr(data_mod, "_find_existing_file", _fake_find)
    resp = client.post(
        "/api/data/ensure", json={"symbol": "GBPUSD", "timeframe": "MINUTE_15"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    assert seen["date_from"] is None  # coverage check skipped for implicit request


def test_ensure_explicit_date_from_enforces_coverage(client, monkeypatch, tmp_path):
    """An explicit date_from still requires the cached file to cover it —
    _find_existing_file receives that date so a too-late file is rejected."""
    from fwbg.api import data as data_mod

    fake_path = tmp_path / "GBPUSD_MINUTE_15.csv"
    fake_path.write_text("T,O,H,L,C,V\n")
    seen: dict = {}

    def _fake_find(s, t, d=None, e=None):
        seen["date_from"] = d
        return ("src1", fake_path)

    monkeypatch.setattr(data_mod, "_find_existing_file", _fake_find)
    client.post(
        "/api/data/ensure",
        json={"symbol": "GBPUSD", "timeframe": "MINUTE_15", "date_from": "2015-01-01"},
    )
    assert seen["date_from"] == "2015-01-01"


# ── Full-history defaults + timeframe listing ───────────────────────────────


def test_timeframes_endpoint_lists_supported(client):
    resp = client.get("/api/data/timeframes")
    assert resp.status_code == 200
    tfs = resp.json()["timeframes"]
    assert "MINUTE_1" in tfs and "HOUR_1" in tfs and "DAY_1" in tfs


def test_default_history_start_uses_catalogue_per_granularity():
    from fwbg.api.data import _default_history_start

    daily = _default_history_start("EURUSD", "DAY_1")
    hourly = _default_history_start("EURUSD", "HOUR_1")
    minute = _default_history_start("EURUSD", "MINUTE_15")
    # Daily FX history reaches decades further back than intraday candles.
    assert daily < hourly <= minute
    assert daily < "2000-01-01"  # EURUSD daily starts in the 1970s


def test_default_history_start_falls_back_for_unknown_symbol():
    from fwbg.api.data import _FALLBACK_HISTORY_START, _default_history_start

    assert _default_history_start("NOSUCHSYM", "HOUR_1") == _FALLBACK_HISTORY_START


def test_ensure_download_defaults_to_full_history(client, monkeypatch, tmp_path):
    """No date_from in the request -> the background download is started with
    the catalogue's history start, not the old 2020-01-01 default."""
    import fwbg.data.dukascopy as dk_mod
    from fwbg.api import data as data_mod
    from fwbg.core.data_sources import CSVSourceConfig

    captured = {}

    def fake_download(path, symbols, timeframe, start, end, **kw):
        captured["start"] = start
        return [{"symbol": symbols[0], "file": "x.csv", "rows": 1}]

    monkeypatch.setattr(
        data_mod, "_find_existing_file", lambda s, t, d=None, e=None: None
    )
    monkeypatch.setattr(
        data_mod,
        "_first_csv_source",
        lambda: CSVSourceConfig(name="dukas_src", path=tmp_path),
    )
    monkeypatch.setattr(dk_mod, "download", fake_download)

    resp = client.post(
        "/api/data/ensure", json={"symbol": "EURUSD", "timeframe": "DAY_1"}
    )
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]
    # The daemon thread runs fake_download almost immediately; poll until done.
    for _ in range(50):
        if client.get(f"/api/data/ensure/{task_id}").json()["status"] != "running":
            break
    assert captured["start"].year < 2000


# ── Coverage of cached files: start, end and gaps ───────────────────────────


def _write_bars(path, days):
    """One bar per given ISO date, in the fwbg T,O,H,L,C,V layout."""
    rows = "".join(f"{d} 00:00:00,1,1,1,1,0\n" for d in days)
    path.write_text("T,O,H,L,C,V\n" + rows)


def _daterange(start, end):
    import pandas as pd

    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end)]


@pytest.fixture
def cached_source(monkeypatch, tmp_path):
    from fwbg.api import data as data_mod
    from fwbg.core.data_sources import CSVSourceConfig

    source = CSVSourceConfig(name="cache", path=tmp_path)
    monkeypatch.setattr(data_mod, "_DATA_SOURCES", {"cache": source})
    return source


def test_cached_file_must_reach_an_explicit_date_to(cached_source):
    from fwbg.api.data import _find_existing_file

    path = cached_source.get_file_path("EURUSD", "DAY_1")
    _write_bars(path, _daterange("2024-01-01", "2024-01-31"))

    assert (
        _find_existing_file("EURUSD", "DAY_1", "2024-01-01", "2024-02-03") is not None
    )
    assert _find_existing_file("EURUSD", "DAY_1", "2024-01-01", "2024-06-01") is None
    assert _find_existing_file("EURUSD", "DAY_1", None, "2024-06-01") is None


def test_implicit_request_still_accepts_any_cached_file(cached_source):
    from fwbg.api.data import _find_existing_file

    path = cached_source.get_file_path("EURUSD", "DAY_1")
    days = _daterange("2024-01-01", "2024-01-05") + _daterange(
        "2024-06-01", "2024-06-05"
    )
    _write_bars(path, days)

    assert _find_existing_file("EURUSD", "DAY_1") is not None


def test_cached_file_with_a_large_hole_in_the_range_is_rejected(cached_source):
    from fwbg.api.data import _find_existing_file

    path = cached_source.get_file_path("EURUSD", "DAY_1")
    days = _daterange("2024-01-01", "2024-02-01") + _daterange(
        "2024-04-01", "2024-04-30"
    )
    _write_bars(path, days)

    assert _find_existing_file("EURUSD", "DAY_1", "2024-01-01", "2024-04-30") is None
    # The hole lies outside this window, so the file is fine for it.
    assert (
        _find_existing_file("EURUSD", "DAY_1", "2024-04-01", "2024-04-30") is not None
    )


def test_holiday_sized_gap_is_not_a_hole(cached_source):
    from fwbg.api.data import _find_existing_file

    path = cached_source.get_file_path("EURUSD", "DAY_1")
    days = _daterange("2024-01-01", "2024-03-28") + _daterange(
        "2024-04-02", "2024-04-30"
    )
    _write_bars(path, days)

    assert (
        _find_existing_file("EURUSD", "DAY_1", "2024-01-01", "2024-04-30") is not None
    )


def test_fresh_quality_report_without_gaps_is_trusted_over_a_scan(cached_source):
    import os
    import time

    from fwbg.api.data import _find_existing_file

    path = cached_source.get_file_path("EURUSD", "DAY_1")
    days = _daterange("2024-01-01", "2024-01-05") + _daterange(
        "2024-05-01", "2024-05-05"
    )
    _write_bars(path, days)  # has a hole a scan would find ...
    report = path.with_suffix(".quality.json")
    report.write_text('{"max_gap_seconds": 86400.0}')  # ... but the report says none
    os.utime(path, (time.time() - 100, time.time() - 100))  # report is newer

    assert (
        _find_existing_file("EURUSD", "DAY_1", "2024-01-01", "2024-05-05") is not None
    )


def test_stale_quality_report_is_ignored(cached_source):
    from fwbg.api.data import _find_existing_file

    path = cached_source.get_file_path("EURUSD", "DAY_1")
    report = path.with_suffix(".quality.json")
    report.write_text('{"max_gap_seconds": 86400.0}')
    import os
    import time

    os.utime(report, (time.time() - 100, time.time() - 100))
    days = _daterange("2024-01-01", "2024-01-05") + _daterange(
        "2024-05-01", "2024-05-05"
    )
    _write_bars(path, days)  # newer than the report, and it has a hole

    assert _find_existing_file("EURUSD", "DAY_1", "2024-01-01", "2024-05-05") is None


def test_ensure_redownload_keeps_the_span_of_the_file_it_replaces(
    client, monkeypatch, cached_source
):
    from fwbg.api import data as data_mod
    import fwbg.data.dukascopy as dk_mod

    path = cached_source.get_file_path("EURUSD", "DAY_1")
    _write_bars(path, _daterange("2020-01-01", "2020-01-31"))
    seen: dict = {}

    def _fake_download(out, symbols, timeframe, start, end, **kw):
        seen["start"], seen["end"] = start, end
        return []

    monkeypatch.setattr(dk_mod, "download", _fake_download)
    monkeypatch.setattr(data_mod, "_first_csv_source", lambda: cached_source)

    resp = client.post(
        "/api/data/ensure",
        json={
            "symbol": "EURUSD",
            "timeframe": "DAY_1",
            "date_from": "2020-01-10",
            "date_to": "2020-06-01",
        },
    )
    assert resp.status_code == 202
    import time

    for _ in range(50):
        if seen:
            break
        time.sleep(0.05)
    assert seen["start"].strftime("%Y-%m-%d") == "2020-01-01"  # not 2020-01-10
    assert seen["end"].strftime("%Y-%m-%d") == "2020-06-01"


def test_ensure_hands_out_an_incomplete_file_for_a_symbol_that_cannot_be_downloaded(
    client, cached_source
):
    path = cached_source.get_file_path("FAKEXYZ999", "DAY_1")
    _write_bars(path, _daterange("2024-01-01", "2024-01-31"))

    resp = client.post(
        "/api/data/ensure",
        json={
            "symbol": "FAKEXYZ999",
            "timeframe": "DAY_1",
            "date_from": "2024-01-01",
            "date_to": "2024-06-01",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["path"] == str(path)
    assert "may not cover" in body["warning"]


def test_hole_threshold_is_fourteen_days(cached_source):
    """The documented rule: a hole of up to 14 days passes, a longer one is a
    failed download (docs/data-checks.md, "When a cached file is re-downloaded")."""
    from fwbg.api.data import _find_existing_file

    path = cached_source.get_file_path("EURUSD", "DAY_1")
    # 2024-01-10 -> 2024-01-24 is exactly 14 days between bars
    days = _daterange("2024-01-01", "2024-01-10") + _daterange(
        "2024-01-24", "2024-02-10"
    )
    _write_bars(path, days)
    assert (
        _find_existing_file("EURUSD", "DAY_1", "2024-01-01", "2024-02-10") is not None
    )

    # one day more: 15 days between bars
    days = _daterange("2024-01-01", "2024-01-10") + _daterange(
        "2024-01-25", "2024-02-10"
    )
    _write_bars(path, days)
    assert _find_existing_file("EURUSD", "DAY_1", "2024-01-01", "2024-02-10") is None


def test_open_ended_request_does_not_apply_the_hole_rule(cached_source):
    from fwbg.api.data import _find_existing_file

    path = cached_source.get_file_path("EURUSD", "DAY_1")
    days = _daterange("2024-01-01", "2024-01-10") + _daterange(
        "2024-03-01", "2024-03-10"
    )
    _write_bars(path, days)
    assert _find_existing_file("EURUSD", "DAY_1") is not None
