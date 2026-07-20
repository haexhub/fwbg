"""Tests for RunStartRequest field validation on POST /api/runs/start.

`cost_multiplier` must be > 0 — a zero or negative value would make the
cost-stress run trade for free (Plan 014).
"""
import pytest
from fastapi.testclient import TestClient

from fwbg.api import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("bad_value", [0, -1])
def test_cost_multiplier_non_positive_returns_422(client, bad_value):
    resp = client.post(
        "/api/runs/start",
        json={"strategy_name": "does-not-matter", "cost_multiplier": bad_value},
    )
    assert resp.status_code == 422
    assert "cost_multiplier" in resp.json()["detail"][0]["loc"]


def test_cost_multiplier_positive_passes_validation(client, monkeypatch):
    # A positive value must clear request validation (may still 404 later
    # because the strategy doesn't exist — that's a different failure mode).
    resp = client.post(
        "/api/runs/start",
        json={"strategy_name": "does-not-exist-xyz", "cost_multiplier": 1.5},
    )
    assert resp.status_code != 422
