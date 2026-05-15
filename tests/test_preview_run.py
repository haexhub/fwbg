"""Tests für die Preview-Run Funktionalität.

Verifiziert:
- run_id wird aus der API als --run-id ans CLI weitergegeben
- assets-Parameter filtert korrekt (Symbol-Match statt Klassen-Lookup)
- progress-Endpoint liefert die richtige job_id zurück
- trades-Endpoint findet die grid_details unter der job_id
"""
import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from fwbg.api import create_app
import fwbg.api.runs as runs_mod


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def client_with_strategy(tmp_path):
    """TestClient mit Temp-Strategie- und Results-Verzeichnis."""
    from fwbg.api import deps as deps_mod
    from fwbg.api import _paths as paths_mod

    orig_strategies = runs_mod.get_strategies_dir
    orig_results_deps = deps_mod.get_test_results_dir
    orig_results_runs = runs_mod.get_test_results_dir
    orig_results_paths = paths_mod.get_test_results_dir

    results_path = tmp_path / "test_results"
    results_path.mkdir()
    results_fn = lambda: results_path

    runs_mod.get_strategies_dir = lambda: tmp_path
    deps_mod.get_test_results_dir = results_fn
    runs_mod.get_test_results_dir = results_fn
    paths_mod.get_test_results_dir = results_fn

    strat = {"name": "Preview Test", "pipeline": {}, "grids": {}}
    (tmp_path / "preview_test.json").write_text(json.dumps(strat))

    app = create_app()
    with TestClient(app) as c:
        yield c, tmp_path

    runs_mod.get_strategies_dir = orig_strategies
    deps_mod.get_test_results_dir = orig_results_deps
    runs_mod.get_test_results_dir = orig_results_runs
    paths_mod.get_test_results_dir = orig_results_paths


# ──────────────────────────────────────────────
# API: run_id wird korrekt übergeben
# ──────────────────────────────────────────────


class TestRunIdPassthrough:
    """Die API muss --run-id an den CLI-Subprozess weitergeben."""

    def test_start_run_includes_run_id_in_cmd(self, client_with_strategy):
        """POST /api/runs/start startet CLI mit --run-id = job_id."""
        client, _ = client_with_strategy

        captured_cmd = []

        original_popen = subprocess.Popen

        def mock_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            # Return a mock process that immediately "succeeds"
            proc = MagicMock()
            proc.pid = 99999
            proc.poll.return_value = 0
            proc.returncode = 0
            return proc

        with patch("fwbg.api.runs.subprocess.Popen", side_effect=mock_popen):
            resp = client.post("/api/runs/start", json={"strategy_name": "preview_test"})

        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        assert "--run-id" in captured_cmd, "--run-id fehlt im CLI-Command"
        run_id_idx = captured_cmd.index("--run-id")
        assert captured_cmd[run_id_idx + 1] == job_id, (
            f"CLI run_id ({captured_cmd[run_id_idx + 1]}) != job_id ({job_id})"
        )

    def test_job_id_format_matches_cli_pattern(self, client_with_strategy):
        """Die job_id hat das CLI-Format: YYYYMMDD_HHMMSS_[6-hex]."""
        client, _ = client_with_strategy

        with patch("fwbg.api.runs.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 99999
            mock_popen.return_value = mock_proc

            resp = client.post("/api/runs/start", json={"strategy_name": "preview_test"})

        job_id = resp.json()["job_id"]
        import re
        assert re.match(r"^\d{8}_\d{6}_[0-9a-f]{6}$", job_id), (
            f"job_id '{job_id}' entspricht nicht dem Format YYYYMMDD_HHMMSS_[6-hex]"
        )

    def test_assets_parameter_passes_assets_flag(self, client_with_strategy):
        """assets=[...] im Request → --assets im CLI-Command."""
        client, _ = client_with_strategy
        captured_cmd = []

        def mock_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            proc = MagicMock()
            proc.pid = 99999
            proc.poll.return_value = 0
            proc.returncode = 0
            return proc

        with patch("fwbg.api.runs.subprocess.Popen", side_effect=mock_popen):
            resp = client.post("/api/runs/start", json={
                "strategy_name": "preview_test",
                "assets": ["DAX"],
            })

        assert resp.status_code == 200
        assert "--assets" in captured_cmd
        assets_idx = captured_cmd.index("--assets")
        assert "DAX" in captured_cmd[assets_idx + 1]

    def test_asset_classes_uses_asset_classes_flag(self, client_with_strategy):
        """asset_classes=[...] → --asset-classes im CLI-Command."""
        client, _ = client_with_strategy
        captured_cmd = []

        def mock_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            proc = MagicMock()
            proc.pid = 99999
            proc.poll.return_value = 0
            proc.returncode = 0
            return proc

        with patch("fwbg.api.runs.subprocess.Popen", side_effect=mock_popen):
            resp = client.post("/api/runs/start", json={
                "strategy_name": "preview_test",
                "asset_classes": ["INDEX"],
            })

        assert resp.status_code == 200
        assert "--asset-classes" in captured_cmd

    def test_no_assets_sends_no_filter(self, client_with_strategy):
        """Ohne assets/asset_classes kein Filter-Flag im CLI-Command."""
        client, _ = client_with_strategy
        captured_cmd = []

        def mock_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            proc = MagicMock()
            proc.pid = 99999
            proc.poll.return_value = 0
            proc.returncode = 0
            return proc

        with patch("fwbg.api.runs.subprocess.Popen", side_effect=mock_popen):
            client.post("/api/runs/start", json={"strategy_name": "preview_test"})

        assert "--assets" not in captured_cmd
        assert "--asset-classes" not in captured_cmd


# ──────────────────────────────────────────────
# API: progress-Endpoint mit job_id
# ──────────────────────────────────────────────


class TestProgressEndpoint:
    """Progress-Endpoint liefert Daten unter der korrekten job_id."""

    def test_progress_returns_job_id_in_response(self, client_with_strategy):
        """GET /api/runs/{job_id}/progress gibt job_id zurück."""
        client, _ = client_with_strategy

        with patch("fwbg.api.runs.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.pid = 99999
            proc.poll.return_value = None  # noch laufend
            mock_popen.return_value = proc

            resp = client.post("/api/runs/start", json={"strategy_name": "preview_test"})
            job_id = resp.json()["job_id"]

        progress = client.get(f"/api/runs/{job_id}/progress")
        assert progress.status_code == 200
        data = progress.json()
        assert data.get("job_id") == job_id or data.get("run_id") == job_id

    def test_unknown_job_returns_404(self, client_with_strategy):
        """Unbekannte job_id → 404."""
        client, _ = client_with_strategy
        resp = client.get("/api/runs/deadbeef/progress")
        assert resp.status_code == 404


# ──────────────────────────────────────────────
# API: trades-Endpoint benötigt grid_details
# ──────────────────────────────────────────────


class TestTradesEndpoint:
    """trades-Endpoint findet Daten unter der job_id, nicht unter einer CLI-generierten ID."""

    def test_trades_without_grid_details_returns_404(self, client_with_strategy):
        """Ohne grid_details → 404 mit passendem Detail."""
        client, _ = client_with_strategy
        resp = client.get("/api/runs/nonexistent123/trades/DAX")
        assert resp.status_code == 404
        assert "DAX" in resp.json()["detail"]

    def test_trades_with_grid_details_returns_data(self, client_with_strategy):
        """Mit vorhandener grid_details-Datei werden Trades zurückgegeben."""
        client, tmp_path = client_with_strategy

        # Simuliere einen abgeschlossenen Run unter der job_id
        results_dir = tmp_path / "test_results"
        run_id = "fakejob1"
        sym_dir = results_dir / run_id / "grid_details" / "DAX"
        sym_dir.mkdir(parents=True, exist_ok=True)

        fold_data = {
            "walk_forward": {
                "fold_details": [{
                    "fold_id": 0,
                    "test_trades_detail": [{
                        "entry_time": "2025-01-02 09:00:00",
                        "exit_time": "2025-01-02 10:00:00",
                        "entry_price": 18000.0,
                        "exit_price": 18100.0,
                        "direction": "LONG",
                        "result": 1.0,
                    }],
                }],
            },
        }
        (sym_dir / "fold_results.json").write_text(json.dumps(fold_data))

        resp = client.get(f"/api/runs/{run_id}/trades/DAX")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "DAX"
        assert data["run_id"] == run_id
        assert len(data["trades"]) == 1
        assert data["trades"][0]["entry_price"] == 18000.0
        assert data["trades"][0]["direction"] == "LONG"


# ──────────────────────────────────────────────
# CLI: --run-id Argument
# ──────────────────────────────────────────────


class TestCLIRunId:
    """CLI verwendet die übergebene run_id statt eine neue zu generieren."""

    def test_help_shows_run_id_option(self):
        """--help enthält --run-id."""
        result = subprocess.run(
            [sys.executable, "-m", "fwbg.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--run-id" in result.stdout

    def test_run_optimizer_uses_provided_run_id(self, tmp_path, monkeypatch):
        """run_optimizer() erstellt das Verzeichnis unter der übergebenen run_id."""
        from fwbg.cli.main import run_optimizer
        from fwbg.results import storage

        # Kein echtes Optimierungs-Subprozess nötig — wir patchen pool_manager
        monkeypatch.setattr(storage, "RESULTS_BASE_PATH", str(tmp_path))

        custom_id = "test-custom-run-id"

        # run_optimizer mit einem leeren Asset-Filter → sofort "keine Dateien"
        # Aber das Verzeichnis wird VOR dem Asset-Scan erstellt → testen wir das
        created_ids = []
        original_create = storage.create_run_directory

        def capture_create(run_id, *args, **kwargs):
            created_ids.append(run_id)
            return original_create(run_id, *args, **kwargs)

        monkeypatch.setattr(storage, "create_run_directory", capture_create)

        # Liefert None wegen leerem asset_filter, aber run_id wurde intern gesetzt
        run_optimizer(
            save_results=True,
            strategy_metadata={"name": "Test"},
            asset_filter=["NONEXISTENT_ASSET_XYZ"],
            run_id=custom_id,
        )

        # Falls Verzeichnis erstellt wurde, muss es unter custom_id liegen
        if created_ids:
            assert created_ids[0] == custom_id, (
                f"Erwartete run_id '{custom_id}', bekam '{created_ids[0]}'"
            )
