"""Request-supplied names must not move data-file paths out of their source."""

import io

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from fwbg.api import create_app
from fwbg.api import datasources as ds_api
from fwbg.core.data_sources import CSVSourceConfig


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def source(monkeypatch, tmp_path):
    root = tmp_path / "data"
    src = CSVSourceConfig(name="t1", path=root / "t1" / "datasource")
    (root / "t1" / "raw").mkdir(parents=True)
    src.path.mkdir(parents=True)
    monkeypatch.setattr(ds_api, "get_data_root", lambda: root)
    monkeypatch.setattr(ds_api, "save_source_config", lambda _s: None)
    monkeypatch.setitem(ds_api._DATA_SOURCES, "t1", src)
    return src


def _process_body(**overrides):
    body = {
        "filename": "raw.csv", "symbol": "EURUSD", "timeframe": "HOUR",
        "date_col": "d", "open_col": "o", "high_col": "h", "low_col": "l", "close_col": "c",
    }
    body.update(overrides)
    return body


def test_get_file_path_rejects_escaping_symbol(tmp_path):
    src = CSVSourceConfig(name="t", path=tmp_path)
    assert src.get_file_path("EURUSD", "HOUR") == tmp_path / "EURUSD_HOUR.csv"
    for bad in ("/etc/x", "../x", "a/../../x"):
        with pytest.raises(ValueError):
            src.get_file_path(bad, "HOUR")


@pytest.mark.parametrize("override", [
    {"filename": "/etc/passwd"},
    {"filename": "../config.json"},
    {"symbol": "/tmp/evil"},
    {"timeframe": "../x"},
])
def test_process_rejects_paths_outside_source(client, source, tmp_path, override):
    outside = tmp_path / "outside.csv"
    outside.write_text("d,o,h,l,c\n2024-01-01,1,1,1,1\n")
    if "filename" in override and override["filename"].startswith("/"):
        override = {"filename": str(outside)}

    r = client.post("/api/datasources/t1/process", json=_process_body(**override))

    assert r.status_code == 400
    assert list(tmp_path.glob("*.csv")) == [outside]


def test_update_rejects_path_outside_data_root(client, source, tmp_path):
    r = client.put("/api/datasources/t1", json={"path": str(tmp_path / "elsewhere")})

    assert r.status_code == 400
    assert ds_api._DATA_SOURCES["t1"].path == source.path


def test_update_keeps_unchanged_path(client, source):
    r = client.put("/api/datasources/t1", json={"path": str(source.path), "description": "x"})

    assert r.status_code == 200


@pytest.mark.parametrize("name", [".", ".."])
def test_create_source_rejects_dot_segments_before_creating_dirs(client, monkeypatch, tmp_path, name):
    root = tmp_path / "data"
    monkeypatch.setattr(ds_api, "get_data_root", lambda: root)

    response = client.post("/api/datasources", json={"name": name, "type": "csv"})

    assert response.status_code == 400
    assert not root.exists()


def test_chart_rejects_symbol_path(client):
    r = client.get("/api/chart/ohlcv", params={"symbol": "/etc/x", "timeframe": "HOUR"})

    assert r.status_code == 400


def test_upload_writes_raw_file_without_leftovers(client, source, tmp_path):
    r = client.post(
        "/api/datasources/t1/raw",
        files={"files": ("prices.csv", b"d,o,h,l,c\n", "text/csv")},
    )

    assert r.status_code == 201
    raw = tmp_path / "data" / "t1" / "raw"
    assert [p.name for p in raw.iterdir()] == ["prices.csv"]
    assert (raw / "prices.csv").read_bytes() == b"d,o,h,l,c\n"


class _CountingReader(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.bytes_read = 0

    def read(self, size=-1):
        chunk = super().read(size)
        self.bytes_read += len(chunk)
        return chunk


def test_oversized_upload_is_rejected_before_it_is_read_whole(source, tmp_path, monkeypatch):
    monkeypatch.setattr(ds_api, "MAX_UPLOAD_SIZE", 1024)
    upload = _CountingReader(b"x" * (8 * 1024 * 1024))

    with pytest.raises(HTTPException) as exc_info:
        ds_api.upload_raw_files("t1", [UploadFile(file=upload, filename="big.csv")])

    assert exc_info.value.status_code == 413
    assert upload.bytes_read < len(upload.getvalue())
    assert list((tmp_path / "data" / "t1" / "raw").iterdir()) == []
