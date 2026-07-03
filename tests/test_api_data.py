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
    resp = client.post("/api/data/ensure", json={"symbol": "EURUSD", "timeframe": "BAD_TF"})
    assert resp.status_code == 422
    assert "timeframe" in resp.json()["detail"].lower()


def test_ensure_non_dukascopy_symbol_returns_404(client, monkeypatch):
    from fwbg.api import data as data_mod

    monkeypatch.setattr(data_mod, "_find_existing_file", lambda s, t: None)
    resp = client.post("/api/data/ensure", json={"symbol": "FAKEXYZ999", "timeframe": "HOUR_1"})
    assert resp.status_code == 404
    assert "Dukascopy" in resp.json()["detail"]


def test_ensure_returns_ready_when_file_exists(client, monkeypatch, tmp_path):
    from fwbg.api import data as data_mod

    fake_path = tmp_path / "EURUSD_HOUR_1.csv"
    fake_path.write_text("T,O,H,L,C,V\n")
    monkeypatch.setattr(data_mod, "_find_existing_file", lambda s, t: ("src1", fake_path))

    resp = client.post("/api/data/ensure", json={"symbol": "EURUSD", "timeframe": "HOUR_1"})
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

    def _fake_find(s, t):
        seen["symbol"] = s
        return ("src1", fake_path)

    monkeypatch.setattr(data_mod, "_find_existing_file", _fake_find)
    client.post("/api/data/ensure", json={"symbol": "EUR/USD", "timeframe": "HOUR_1"})
    assert seen["symbol"] == "EURUSD"


def test_ensure_no_csv_source_returns_503(client, monkeypatch):
    from fwbg.api import data as data_mod

    monkeypatch.setattr(data_mod, "_find_existing_file", lambda s, t: None)
    monkeypatch.setattr(data_mod, "_first_csv_source", lambda: None)
    resp = client.post("/api/data/ensure", json={"symbol": "EURUSD", "timeframe": "HOUR_1"})
    assert resp.status_code == 503


def test_ensure_starts_download_returns_202_and_task_is_pollable(client, monkeypatch, tmp_path):
    from fwbg.core.data_sources import CSVSourceConfig
    from fwbg.api import data as data_mod
    import fwbg.data.dukascopy as dk_mod

    fake_source = CSVSourceConfig(name="dukas_src", path=tmp_path)
    monkeypatch.setattr(data_mod, "_find_existing_file", lambda s, t: None)
    monkeypatch.setattr(data_mod, "_first_csv_source", lambda: fake_source)
    monkeypatch.setattr(
        dk_mod,
        "download",
        lambda out, symbols, timeframe, start, end, **kw: [
            {"symbol": symbols[0], "file": str(out / f"{symbols[0]}_{timeframe}.csv"), "rows": 42, "spread": 0.0}
        ],
    )

    resp = client.post(
        "/api/data/ensure",
        json={"symbol": "EURUSD", "timeframe": "HOUR_1", "date_from": "2023-01-01", "date_to": "2023-03-01"},
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

    monkeypatch.setattr(data_mod, "_find_existing_file", lambda s, t: None)
    monkeypatch.setattr(
        data_mod, "_first_csv_source",
        lambda: CSVSourceConfig(name="src", path=tmp_path),
    )
    resp = client.post(
        "/api/data/ensure",
        json={"symbol": "EURUSD", "timeframe": "HOUR_1", "date_from": "2023-06-01", "date_to": "2023-01-01"},
    )
    assert resp.status_code == 422
    assert "date_to" in resp.json()["detail"].lower()


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

    monkeypatch.setattr(data_mod, "_find_existing_file", lambda s, t: None)
    monkeypatch.setattr(
        data_mod, "_first_csv_source",
        lambda: CSVSourceConfig(name="dukas_src", path=tmp_path),
    )
    monkeypatch.setattr(dk_mod, "download", fake_download)

    resp = client.post("/api/data/ensure", json={"symbol": "EURUSD", "timeframe": "DAY_1"})
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]
    # The daemon thread runs fake_download almost immediately; poll until done.
    for _ in range(50):
        if client.get(f"/api/data/ensure/{task_id}").json()["status"] != "running":
            break
    assert captured["start"].year < 2000
