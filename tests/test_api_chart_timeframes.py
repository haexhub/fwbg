"""Tests für Plan 020 WP4: Timeframe-Pfade der Chart-API.

Deckt ab: kanonische Timeframe-Ausgabe bei /chart/sources (legacy + kanonische
Dateinamen gemischt), Legacy-Schreibweisen-Requests gegen kanonisch benannte
Dateien, MTF-Regression aus PR #133 (1a24ad0), und dass ein am Broker-Adapter
nicht unterstützter Timeframe als HTTP 400 durchgereicht wird (nicht 500).
"""
import pytest
from fastapi.testclient import TestClient

from fwbg.api import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def _write_csv(path, rows):
    lines = ["T,O,H,L,C,V"]
    for ts, o, h, low, c in rows:
        lines.append(f"{ts},{o},{h},{low},{c},0")
    path.write_text("\n".join(lines) + "\n")


def _hourly_rows(n, start="2024-01-01 00:00:00", step_hours=1):
    import datetime

    base = datetime.datetime.fromisoformat(start)
    rows = []
    for i in range(n):
        ts = base + datetime.timedelta(hours=i * step_hours)
        rows.append((ts.isoformat(sep=" "), 1.10, 1.11, 1.09, 1.105))
    return rows


@pytest.fixture
def csv_source(tmp_path, monkeypatch):
    import fwbg.core.data_sources as ds_mod
    from fwbg.core.data_sources import CSVSourceConfig

    # Legacy-benannte Datei (Kurzform) neben kanonisch benannter Datei.
    _write_csv(tmp_path / "EURUSD_HOUR.csv", _hourly_rows(24))
    _write_csv(tmp_path / "DAX_DAY_1.csv", _hourly_rows(5, step_hours=24))
    _write_csv(tmp_path / "EURUSD_HOUR_1.csv", _hourly_rows(24))
    _write_csv(tmp_path / "EURUSD_HOUR_4.csv", _hourly_rows(10, step_hours=4))

    source = CSVSourceConfig(name="test_src", path=tmp_path)
    monkeypatch.setattr(ds_mod, "_DATA_SOURCES", {"test_src": source})
    return source


# ---------------------------------------------------------------------------
# GET /api/chart/sources — canonical timeframes regardless of on-disk naming
# ---------------------------------------------------------------------------


def test_sources_reports_canonical_timeframes_for_legacy_and_canonical_files(csv_source, client):
    resp = client.get("/api/chart/sources")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    symbols = {s["symbol"]: s["timeframes"] for s in body[0]["symbols"]}

    # EURUSD_HOUR.csv (legacy) must surface as canonical "HOUR_1".
    assert "HOUR_1" in symbols["EURUSD"]
    # DAX_DAY_1.csv (already canonical) must surface as "DAY_1".
    assert "DAY_1" in symbols["DAX"]


# ---------------------------------------------------------------------------
# GET /api/chart/ohlcv — legacy timeframe spelling resolves via canonicalization
# ---------------------------------------------------------------------------


def test_ohlcv_legacy_timeframe_spelling_resolves_canonical_file(csv_source, client):
    """Request with legacy "HOUR" must find EURUSD_HOUR_1.csv via
    _best_native_file's canonicalization, not 404."""
    resp = client.get("/api/chart/ohlcv", params={"symbol": "EURUSD", "timeframe": "HOUR", "source": "test_src"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] > 0


# ---------------------------------------------------------------------------
# POST /api/chart/indicator — MTF timeframe handling
# ---------------------------------------------------------------------------


class _FakeIndicatorPlugin:
    def get_default_params(self):
        return {}

    def compute(self, df, **params):
        df = df.copy()
        df["indicator_value"] = df["C"]
        return df

    def get_feature_columns(self, params=None):
        return ["indicator_value"]


@pytest.fixture
def fake_plugin_registry(monkeypatch):
    import fwbg.api.deps as deps_mod

    class _FakeRegistry:
        def get(self, fqn):
            return _FakeIndicatorPlugin

    monkeypatch.setattr(deps_mod, "get_plugin_registry", lambda: _FakeRegistry())
    return _FakeRegistry()


def _indicator_body(**overrides):
    body = {
        "symbol": "EURUSD",
        "timeframe": "HOUR",
        "source": "test_src",
        "fqn": "fake.indicator",
        "limit": 100,
    }
    body.update(overrides)
    return body


def test_indicator_same_canonical_timeframe_is_single_tf_no_400(csv_source, fake_plugin_registry, client):
    """Regression on the PR #133 review-fix (1a24ad0): chart timeframe "HOUR"
    plus indicator_timeframe "HOUR_1" is the SAME timeframe in different
    spellings — must take the single-TF path, not 400."""
    resp = client.post(
        "/api/chart/indicator",
        json=_indicator_body(indicator_timeframe="HOUR_1"),
    )
    assert resp.status_code == 200


def test_indicator_timeframe_below_chart_timeframe_returns_400(csv_source, fake_plugin_registry, client):
    """indicator_timeframe must be higher than the chart timeframe."""
    resp = client.post(
        "/api/chart/indicator",
        json=_indicator_body(timeframe="HOUR_4", indicator_timeframe="HOUR_1"),
    )
    assert resp.status_code == 400


def test_indicator_unknown_timeframe_returns_400(csv_source, fake_plugin_registry, client):
    resp = client.post(
        "/api/chart/indicator",
        json=_indicator_body(indicator_timeframe="NOT_A_TIMEFRAME"),
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/chart/ohlcv (broker path) — unsupported timeframe -> 400, not 500
# ---------------------------------------------------------------------------


class _StubBrokerAdapter:
    """Stub in place of a real fwbg_broker_ig adapter — mirrors the
    fail-loud ValueError the real adapter now raises for an unmapped
    timeframe (Plan 020 WP3), without needing IG credentials/network."""

    def connect(self):
        return True

    def disconnect(self):
        return True

    def get_historical_bars(self, symbol, timeframe, limit=1000):
        raise ValueError(f"unsupported timeframe for yfinance interval: {timeframe!r}")


def test_post_ohlcv_broker_unsupported_timeframe_returns_400_not_500(client, monkeypatch):
    import fwbg.api.chart as chart_mod

    monkeypatch.setattr(
        chart_mod, "_create_broker_adapter", lambda broker_type, credentials: _StubBrokerAdapter()
    )

    resp = client.post(
        "/api/chart/ohlcv",
        json={
            "symbol": "EURUSD",
            "timeframe": "HOUR_2",
            "broker_type": "ig",
            "credentials": {"username": "u", "password": "p", "api_key": "k"},
        },
    )
    assert resp.status_code == 400
    assert resp.status_code != 500
