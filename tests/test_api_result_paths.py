"""Regression tests for API filesystem boundaries around run results."""

import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from fwbg.api import create_app
from fwbg.api import _paths as paths_mod
from fwbg.api import deps as deps_mod
from fwbg.api import runs as runs_mod


@pytest.fixture
def result_client(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    monkeypatch.setattr(deps_mod, "get_test_results_dir", lambda: results_dir)
    monkeypatch.setattr(runs_mod, "get_test_results_dir", lambda: results_dir)
    monkeypatch.setattr(paths_mod, "get_test_results_dir", lambda: results_dir)
    monkeypatch.setattr(runs_mod, "_active_jobs", {})

    app = create_app()
    with TestClient(app) as client:
        yield client, results_dir


def test_results_root_is_reserved_and_cannot_be_deleted(result_client):
    client, results_dir = result_client

    with pytest.raises(HTTPException) as exc_info:
        paths_mod.safe_results_path(".")
    assert exc_info.value.status_code == 400

    response = client.delete("/api/runs/%2E")
    assert response.status_code == 400
    assert results_dir.exists()


def test_results_symlink_outside_root_cannot_be_read_or_deleted(result_client, tmp_path):
    client, results_dir = result_client
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.json").write_text(json.dumps({"secret": True}))
    (results_dir / "escape").symlink_to(outside, target_is_directory=True)

    get_response = client.get("/api/runs/escape")
    delete_response = client.delete("/api/runs/escape")

    assert get_response.status_code == 400
    assert delete_response.status_code == 400
    assert outside.exists()
    assert (outside / "secret.json").exists()


def test_normal_run_can_be_read_and_deleted_without_touching_neighbors(result_client):
    client, results_dir = result_client
    run_dir = results_dir / "run-1"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"description": "normal"}))
    neighbor = results_dir / "neighbor"
    neighbor.mkdir()
    (neighbor / "keep.txt").write_text("keep")

    get_response = client.get("/api/runs/run-1")
    delete_response = client.delete("/api/runs/run-1")

    assert get_response.status_code == 200
    assert get_response.json()["run_id"] == "run-1"
    assert delete_response.status_code == 200
    assert not run_dir.exists()
    assert neighbor.exists()
    assert (neighbor / "keep.txt").exists()
    assert results_dir.exists()


def test_compare_uses_the_safe_results_resolver(result_client):
    client, results_dir = result_client
    (results_dir / "run-1").mkdir()

    valid_response = client.post("/api/runs/compare", json={"run_ids": ["run-1"]})
    invalid_response = client.post("/api/runs/compare", json={"run_ids": ["."]})

    assert valid_response.status_code == 200
    assert valid_response.json()["runs"][0]["run_id"] == "run-1"
    assert invalid_response.status_code == 400


def test_preview_rejects_invalid_strategy_identifier(result_client):
    client, _ = result_client

    response = client.post(
        "/api/runs/preview",
        json={"strategy_name": "../outside", "symbol": "DAX"},
    )

    assert response.status_code == 400


def test_active_run_cannot_be_deleted(result_client):
    client, results_dir = result_client
    run_dir = results_dir / "active-run"
    run_dir.mkdir()
    (run_dir / "keep.txt").write_text("keep")

    class ActiveProcess:
        def poll(self):
            return None

    runs_mod._active_jobs["active-run"] = {
        "status": "running",
        "process": ActiveProcess(),
    }

    response = client.delete("/api/runs/active-run")

    assert response.status_code == 409
    assert run_dir.exists()
    assert (run_dir / "keep.txt").exists()
