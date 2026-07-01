"""Tests for the asset-registry endpoints (/api/assets, /api/assets/classes)."""

import pytest
from fastapi.testclient import TestClient

from fwbg.api import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_list_asset_classes(client):
    resp = client.get("/api/assets/classes")
    assert resp.status_code == 200
    data = resp.json()

    assert data["classes"] == ["FOREX", "INDEX", "COMMODITY", "CRYPTO"]
    # EURUSD is a FOREX major and must be listed under FOREX.
    assert "EURUSD" in data["by_class"]["FOREX"]
    # Internal TEST bucket must not leak into the public vocabulary.
    assert "TEST" not in data["by_class"]
    # by_class is symbol-sorted for stable dropdowns.
    for symbols in data["by_class"].values():
        assert symbols == sorted(symbols)


def test_list_assets(client):
    resp = client.get("/api/assets")
    assert resp.status_code == 200
    assets = resp.json()["assets"]

    by_symbol = {a["symbol"]: a for a in assets}
    assert by_symbol["EURUSD"]["asset_class"] == "FOREX"
    assert by_symbol["BTCUSD"]["asset_class"] == "CRYPTO"
    assert set(by_symbol["EURUSD"]["currencies"]) == {"EUR", "USD"}
    # No internal/test assets in the flat list.
    assert all(a["asset_class"] in {"FOREX", "INDEX", "COMMODITY", "CRYPTO"} for a in assets)
    assert "TESTUSD" not in by_symbol
