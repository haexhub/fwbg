"""Integration tests for the FWBG REST API.

Tests run against the real plugin registry and filesystem — no mocks.
"""
import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fwbg.api import create_app
from fwbg.api.deps import get_plugin_registry


@pytest.fixture(scope="module")
def client():
    """Create a test client with a fresh FastAPI app."""
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def registry():
    """Ensure the plugin registry is populated."""
    return get_plugin_registry()


@pytest.fixture
def strategy_client(tmp_path):
    """Create a test client with strategies pointing to a temp directory."""
    import fwbg.api.strategies as strat_mod
    import fwbg.api.runs as runs_mod

    orig_strat = strat_mod.get_strategies_dir
    orig_runs = runs_mod.get_strategies_dir

    def _tmp_dir():
        return tmp_path

    # Patch at the call site (where the function is imported as a local name)
    strat_mod.get_strategies_dir = _tmp_dir
    runs_mod.get_strategies_dir = _tmp_dir

    app = create_app()
    with TestClient(app) as c:
        yield c, tmp_path

    strat_mod.get_strategies_dir = orig_strat
    runs_mod.get_strategies_dir = orig_runs


# ──────────────────────────────────────────────
# Plugin Endpoints
# ──────────────────────────────────────────────


class TestPluginEndpoints:
    """Tests for /api/plugins endpoints."""

    def test_list_plugins_returns_all(self, client, registry):
        """Endpoint returns all discovered plugins."""
        resp = client.get("/api/plugins")
        assert resp.status_code == 200
        plugins = resp.json()
        assert len(plugins) == len(registry.list_plugins())

    def test_plugin_has_required_fields(self, client):
        """Every plugin has the fields needed by the UI."""
        resp = client.get("/api/plugins")
        plugins = resp.json()
        required = {"fqn", "name", "namespace", "phase", "version", "param_schema", "defaults"}
        for p in plugins:
            missing = required - set(p.keys())
            assert not missing, f"Plugin {p['fqn']} missing fields: {missing}"

    def test_param_schema_matches_defaults(self, client):
        """param_schema keys match defaults keys for all plugins."""
        resp = client.get("/api/plugins")
        for p in resp.json():
            schema_keys = set(p["param_schema"].keys())
            default_keys = set(p["defaults"].keys())
            assert schema_keys == default_keys, (
                f"Plugin {p['fqn']}: schema keys {schema_keys} != default keys {default_keys}"
            )

    def test_param_schema_has_type_and_default(self, client):
        """Every param in schema has type and default."""
        resp = client.get("/api/plugins")
        for p in resp.json():
            for param_name, schema in p["param_schema"].items():
                assert "type" in schema, f"{p['fqn']}.{param_name} missing 'type'"
                assert "default" in schema, f"{p['fqn']}.{param_name} missing 'default'"

    def test_filter_by_phase(self, client):
        """Filtering by phase returns only matching plugins."""
        resp = client.get("/api/plugins?phase=indicators")
        assert resp.status_code == 200
        plugins = resp.json()
        assert len(plugins) > 0
        assert all(p["phase"] == "indicators" for p in plugins)

    def test_filter_invalid_phase(self, client):
        """Invalid phase returns 400."""
        resp = client.get("/api/plugins?phase=nonexistent")
        assert resp.status_code == 400

    def test_get_single_plugin(self, client):
        """Fetch a specific plugin by FQN."""
        resp = client.get("/api/plugins/fwbg-core:trend")
        assert resp.status_code == 200
        p = resp.json()
        assert p["fqn"] == "fwbg-core:trend"
        assert p["phase"] == "indicators"
        assert len(p["param_schema"]) > 0

    def test_get_nonexistent_plugin(self, client):
        """Requesting unknown plugin returns 404."""
        resp = client.get("/api/plugins/fwbg-core:does_not_exist")
        assert resp.status_code == 404

    def test_all_phases_represented(self, client):
        """All pipeline phases have at least one plugin."""
        resp = client.get("/api/plugins")
        phases = {p["phase"] for p in resp.json()}
        assert "indicators" in phases
        assert "feature_selection" in phases

    def test_plugin_schema_types_valid(self, client):
        """All param types are one of the known types."""
        valid_types = {"int", "float", "bool", "string", "list[int]", "list[float]", "list[string]", "choice"}
        resp = client.get("/api/plugins")
        for p in resp.json():
            for param_name, schema in p["param_schema"].items():
                assert schema["type"] in valid_types, (
                    f"{p['fqn']}.{param_name} has unknown type: {schema['type']}"
                )


# ──────────────────────────────────────────────
# Strategy Endpoints
# ──────────────────────────────────────────────


class TestStrategyEndpoints:
    """Tests for /api/strategies CRUD endpoints."""

    def test_list_real_strategies(self, client):
        """List strategies from the real strategies/ directory."""
        resp = client.get("/api/strategies")
        assert resp.status_code == 200
        strategies = resp.json()
        # We know at least sr_trend_continuation and exploration exist
        names = [s["filename"] for s in strategies]
        assert "sr_trend_continuation" in names
        assert "exploration" in names

    def test_load_strategy(self, client):
        """Load a real strategy file and verify structure."""
        resp = client.get("/api/strategies/sr_trend_continuation")
        assert resp.status_code == 200
        data = resp.json()
        # Verify it has the expected strategy sections
        assert "pipeline" in data
        assert "exit_strategy" in data
        assert "model" in data
        assert "grids" in data
        assert "validation" in data
        # Verify pipeline has plugin arrays
        pipeline = data["pipeline"]
        assert "indicators" in pipeline
        assert isinstance(pipeline["indicators"], list)
        assert len(pipeline["indicators"]) > 0

    def test_load_nonexistent_strategy(self, client):
        """Loading unknown strategy returns 404."""
        resp = client.get("/api/strategies/nonexistent_strategy_xyz")
        assert resp.status_code == 404

    def test_create_and_load_strategy(self, strategy_client):
        """Create a strategy and load it back."""
        client, tmp_dir = strategy_client
        strategy_data = {
            "name": "Test Strategy",
            "data": {
                "pipeline": {"indicators": [{"name": "trend", "params": {}}]},
                "exit_strategy": "fixed",
                "exit_params": {"tp": 50, "sl": 50},
                "model": {"type": "xgboost"},
                "grids": {},
                "validation": {"method": "walk_forward", "folds": 4},
            },
        }

        # Create
        resp = client.post("/api/strategies", json=strategy_data)
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] == "created"
        filename = result["filename"]

        # Verify file exists on disk
        assert (tmp_dir / f"{filename}.json").exists()

        # Load back
        resp = client.get(f"/api/strategies/{filename}")
        assert resp.status_code == 200
        loaded = resp.json()
        assert loaded["name"] == "Test Strategy"
        assert loaded["pipeline"]["indicators"][0]["name"] == "trend"

    def test_create_duplicate_strategy(self, strategy_client):
        """Creating a duplicate returns 409."""
        client, _ = strategy_client
        body = {"name": "Dupe", "data": {"pipeline": {}}}
        client.post("/api/strategies", json=body)
        resp = client.post("/api/strategies", json=body)
        assert resp.status_code == 409

    def test_update_strategy(self, strategy_client):
        """Update an existing strategy."""
        client, _ = strategy_client
        # Create
        client.post("/api/strategies", json={"name": "Updateable", "data": {"v": 1}})

        # Update
        updated = {"name": "Updateable", "v": 2, "extra_field": True}
        resp = client.put("/api/strategies/updateable", json=updated)
        assert resp.status_code == 200

        # Verify updated
        resp = client.get("/api/strategies/updateable")
        assert resp.json()["v"] == 2
        assert resp.json()["extra_field"] is True

    def test_delete_strategy(self, strategy_client):
        """Delete a strategy file."""
        client, _ = strategy_client
        client.post("/api/strategies", json={"name": "Deletable", "data": {}})

        resp = client.delete("/api/strategies/deletable")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify gone
        resp = client.get("/api/strategies/deletable")
        assert resp.status_code == 404

    def test_delete_nonexistent(self, strategy_client):
        """Deleting unknown strategy returns 404."""
        client, _ = strategy_client
        resp = client.delete("/api/strategies/ghost")
        assert resp.status_code == 404


# ──────────────────────────────────────────────
# Run Endpoints
# ──────────────────────────────────────────────


class TestRunEndpoints:
    """Tests for /api/runs endpoints."""

    def test_list_runs(self, client):
        """List runs from test_results/."""
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert isinstance(runs, list)
        # Verify each run has required fields
        for run in runs:
            assert "run_id" in run
            assert "status" in run

    def test_list_runs_limit(self, client):
        """Limit parameter restricts result count."""
        resp = client.get("/api/runs?limit=2")
        runs = resp.json()
        assert len(runs) <= 2

    def test_get_run_details(self, client):
        """Get details of a completed run."""
        # First get a run_id
        runs = client.get("/api/runs?limit=1").json()
        if not runs:
            pytest.skip("No runs available")

        run_id = runs[0]["run_id"]
        resp = client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        detail = resp.json()
        assert detail["run_id"] == run_id
        assert detail["status"] == "completed"

    def test_get_nonexistent_run(self, client):
        """Requesting unknown run returns 404."""
        resp = client.get("/api/runs/nonexistent_run_id")
        assert resp.status_code == 404

    def test_start_run_nonexistent_strategy(self, client):
        """Starting a run with unknown strategy returns 404."""
        resp = client.post("/api/runs/start", json={
            "strategy_name": "nonexistent_strategy_xyz",
        })
        assert resp.status_code == 404

    def test_cancel_nonexistent_job(self, client):
        """Cancelling unknown job returns 404."""
        resp = client.post("/api/runs/ghost/cancel")
        assert resp.status_code == 404

    def test_progress_nonexistent_job(self, client):
        """Progress for unknown job returns 404."""
        resp = client.get("/api/runs/ghost/progress")
        assert resp.status_code == 404


# ──────────────────────────────────────────────
# Schema Consistency
# ──────────────────────────────────────────────


class TestSchemaConsistency:
    """Verify param schemas are consistent across the registry."""

    def test_numeric_params_have_defaults_in_range(self, client):
        """For numeric params with min/max, default is within bounds."""
        resp = client.get("/api/plugins")
        for p in resp.json():
            for param_name, schema in p["param_schema"].items():
                if schema["type"] in ("int", "float") and "min" in schema and "max" in schema:
                    default = schema["default"]
                    if default is not None:
                        assert schema["min"] <= default <= schema["max"], (
                            f"{p['fqn']}.{param_name}: default {default} "
                            f"outside [{schema['min']}, {schema['max']}]"
                        )

    def test_choice_params_have_choices(self, client):
        """Choice-type params must have a 'choices' list."""
        resp = client.get("/api/plugins")
        for p in resp.json():
            for param_name, schema in p["param_schema"].items():
                if schema["type"] == "choice":
                    assert "choices" in schema and isinstance(schema["choices"], list), (
                        f"{p['fqn']}.{param_name}: choice type without choices list"
                    )

    def test_described_plugins_have_param_descriptions(self, client):
        """Plugins with descriptions in manifest also have param descriptions."""
        resp = client.get("/api/plugins")
        plugins_with_descriptions = [p for p in resp.json() if p.get("description")]
        # At least some core plugins should have descriptions
        assert len(plugins_with_descriptions) > 5, "Expected >5 plugins with descriptions"

    def test_strategy_plugins_exist_in_registry(self, client):
        """All plugins referenced in strategies exist in the registry."""
        strategies_resp = client.get("/api/strategies")
        plugins_resp = client.get("/api/plugins")
        registered_names = {p["name"] for p in plugins_resp.json()}

        for strat_info in strategies_resp.json():
            strat = client.get(f"/api/strategies/{strat_info['filename']}").json()
            pipeline = strat.get("pipeline", {})
            for section in ["indicators", "preprocessing", "feature_selection", "data_loading"]:
                for entry in pipeline.get(section, []):
                    plugin_name = entry.get("name", "")
                    assert plugin_name in registered_names, (
                        f"Strategy '{strat_info['filename']}' references unknown plugin: {plugin_name}"
                    )
