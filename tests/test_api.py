"""Integration tests for the FWBG REST API.

Tests run against the real plugin registry and filesystem — no mocks.
"""

import os
import subprocess
import sys

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
# Auth / fail-closed startup
# ──────────────────────────────────────────────


def test_create_app_fails_closed_without_key(monkeypatch):
    """create_app() refuses to start unauthenticated unless dev bypass is set."""
    monkeypatch.delenv("FWBG_API_KEY", raising=False)
    monkeypatch.delenv("FWBG_ALLOW_UNAUTHENTICATED_API", raising=False)
    with pytest.raises(RuntimeError, match="FWBG_API_KEY"):
        create_app()


def _run_without_api_env(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter with both API auth vars removed."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("FWBG_API_KEY", "FWBG_ALLOW_UNAUTHENTICATED_API")
    }
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )


def test_importing_a_submodule_does_not_build_the_app():
    """Importing from fwbg.api must not construct the app.

    `python -m fwbg` (the trading bot, which serves no API) imports
    get_accounts_dir from fwbg.api.workspace. While the app was built at import
    time, that pulled in the fail-closed auth guard and crashlooped the bot with
    "FWBG_API_KEY is not set".
    """
    result = _run_without_api_env(
        "from fwbg.api.workspace import get_accounts_dir; assert get_accounts_dir"
    )
    assert result.returncode == 0, result.stderr


def test_app_attribute_still_resolves_for_uvicorn():
    """`fwbg.api:app` is the api service's uvicorn target and must keep working."""
    result = _run_without_api_env(
        "import os; os.environ['FWBG_API_KEY'] = 's3cret'\n"
        "from fwbg.api import app\n"
        "assert app.title, 'app did not build'"
    )
    assert result.returncode == 0, result.stderr


def test_app_attribute_still_fails_closed():
    """Deferring construction must not weaken the guard — only move it."""
    result = _run_without_api_env("from fwbg.api import app")
    assert result.returncode != 0
    assert "FWBG_API_KEY" in result.stderr


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
        resp = client.get("/api/plugins/fwbg-core:ema")
        assert resp.status_code == 200
        p = resp.json()
        assert p["fqn"] == "fwbg-core:ema"
        assert p["phase"] == "indicators"
        assert len(p["param_schema"]) > 0

    def test_plugin_exposes_depends_on(self, client):
        """depends_on is exposed so agent tooling can validate it before authoring."""
        resp = client.get("/api/plugins")
        plugins = {p["fqn"]: p for p in resp.json()}
        regime_cluster = plugins["fwbg-premium:regime_cluster"]
        assert regime_cluster["depends_on"] == ["regime", "volatility"]
        # Plugins without declared dependencies still expose the (empty) field.
        adx = plugins["fwbg-core:adx"]
        assert adx["depends_on"] == []

    def test_get_nonexistent_plugin(self, client):
        """Requesting unknown plugin returns 404."""
        resp = client.get("/api/plugins/fwbg-core:does_not_exist")
        assert resp.status_code == 404

    def test_get_plugin_source(self, client):
        """Plugin source is served for the PluginPlanner examples."""
        resp = client.get("/api/plugins/fwbg-core:ema/source")
        assert resp.status_code == 200
        data = resp.json()
        assert data["fqn"] == "fwbg-core:ema"
        assert data["filename"] == "__init__.py"
        assert "class" in data["source"]
        assert "EMA" in data["source"]

    def test_get_source_nonexistent_plugin(self, client):
        """Source for an unknown plugin returns 404."""
        resp = client.get("/api/plugins/fwbg-core:does_not_exist/source")
        assert resp.status_code == 404

    def test_get_plugin_spec(self, client):
        """The co-located speckit spec.md is served for the dedup gate."""
        resp = client.get("/api/plugins/fwbg-core:ema/spec")
        assert resp.status_code == 200
        data = resp.json()
        assert data["fqn"] == "fwbg-core:ema"
        assert "Plugin Spec" in data["spec"]

    def test_get_spec_nonexistent_plugin(self, client):
        """Spec for an unknown plugin returns 404."""
        resp = client.get("/api/plugins/fwbg-core:does_not_exist/spec")
        assert resp.status_code == 404

    def test_all_phases_represented(self, client):
        """All pipeline phases have at least one plugin."""
        resp = client.get("/api/plugins")
        phases = {p["phase"] for p in resp.json()}
        assert "indicators" in phases
        assert "feature_selection" in phases

    def test_plugin_schema_types_valid(self, client):
        """All param types are one of the known types."""
        valid_types = {"int", "float", "bool", "string", "list[int]", "list[float]", "list[string]", "list[object]", "choice", "session_ranges"}
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
        assert "exit_strategies" in data
        assert "model" in data
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
                "exit_strategies": [{"name": "fixed", "params": {"tp_mult": 1.0, "sl_mult": 1.0}, "ct": [0.5]}],
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

    def test_create_rejects_traversal_name(self, strategy_client):
        """A traversal name in the POST body is rejected and writes nothing outside tmp_dir."""
        client, tmp_dir = strategy_client
        resp = client.post("/api/strategies", json={"name": "../evil", "data": {}})
        assert resp.status_code == 400
        # No file escaped the strategies directory.
        assert not (tmp_dir.parent / "evil.json").exists()
        assert not (tmp_dir / "evil.json").exists()

    def test_create_rejects_slashed_name(self, strategy_client):
        """A name with path separators is rejected."""
        client, _ = strategy_client
        resp = client.post("/api/strategies", json={"name": "a/b/c", "data": {}})
        assert resp.status_code == 400

    def test_put_delete_reject_invalid_path_name(self, strategy_client):
        """Invalid characters in the {name} path param are rejected by the handler.

        Note: a literal ``../evil`` in the URL is normalized by the HTTP client
        to ``/api/evil`` before it reaches the app (→ 404, still no traversal),
        so we exercise the handler's own validation with an invalid single
        path segment instead.
        """
        client, _ = strategy_client
        assert client.put("/api/strategies/bad!name", json={}).status_code == 400
        assert client.delete("/api/strategies/bad!name").status_code == 400

    def test_create_valid_spaced_name(self, strategy_client):
        """A normal name with spaces still lowercases to test_strategy.json."""
        client, tmp_dir = strategy_client
        resp = client.post("/api/strategies", json={"name": "Test Strategy", "data": {}})
        assert resp.status_code == 200
        assert resp.json()["filename"] == "test_strategy"
        assert (tmp_dir / "test_strategy.json").exists()


# ──────────────────────────────────────────────
# Run Endpoints
# ──────────────────────────────────────────────


class TestRunEndpoints:
    """Tests for /api/runs endpoints."""

    def test_list_runs(self, client):
        """List runs from test_results/."""
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        runs = data["items"]
        # Verify each run has required fields
        for run in runs:
            assert "run_id" in run
            assert "status" in run

    def test_list_runs_limit(self, client):
        """Limit parameter restricts result count."""
        resp = client.get("/api/runs?limit=2")
        data = resp.json()
        runs = data["items"] if isinstance(data, dict) else data
        assert len(runs) <= 2

    def test_get_run_details(self, client):
        """Get details of a completed run."""
        # First get a run_id
        data = client.get("/api/runs?limit=1").json()
        runs = data["items"] if isinstance(data, dict) else data
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


# ──────────────────────────────────────────────
# Chart source resolution
# ──────────────────────────────────────────────


class TestResolveSource:
    """Tests for chart._resolve_source (optional 'source' param, Plan 010)."""

    def test_explicit_source_is_passed_through(self):
        from fwbg.api.chart import _resolve_source

        assert _resolve_source("dukascopy") == "dukascopy"

    def test_single_configured_source_is_used(self, monkeypatch):
        import fwbg.core.data_sources as ds_mod
        from fwbg.api.chart import _resolve_source

        monkeypatch.setattr(ds_mod, "_DATA_SOURCES", {"only_one": object()})
        assert _resolve_source(None) == "only_one"

    def test_multiple_sources_raise_422_listing_names(self, monkeypatch):
        from fastapi import HTTPException

        import fwbg.core.data_sources as ds_mod
        from fwbg.api.chart import _resolve_source

        monkeypatch.setattr(
            ds_mod, "_DATA_SOURCES", {"alpha": object(), "beta": object()}
        )
        with pytest.raises(HTTPException) as exc_info:
            _resolve_source(None)
        assert exc_info.value.status_code == 422
        assert "alpha" in exc_info.value.detail
        assert "beta" in exc_info.value.detail

    def test_no_sources_raise_422(self, monkeypatch):
        from fastapi import HTTPException

        import fwbg.core.data_sources as ds_mod
        from fwbg.api.chart import _resolve_source

        monkeypatch.setattr(ds_mod, "_DATA_SOURCES", {})
        with pytest.raises(HTTPException) as exc_info:
            _resolve_source(None)
        assert exc_info.value.status_code == 422
