"""Tests for POST /api/plugins (agent-authored plugin registration)."""
import importlib
import json

import pytest
from fastapi.testclient import TestClient

from fwbg.api import create_app
from fwbg.api.deps import get_plugin_registry
from fwbg.pipeline.registry import reset_registry


_VALID_PLUGIN_CODE = """\
from fwbg_sdk import BasePlugin, PluginPhase
import pandas as pd


class TestIndicatorPlugin(BasePlugin):
    name = "test_reg_indicator"
    version = "1.0.0"
    phase = PluginPhase.INDICATORS

    def run(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        df = df.copy()
        df["test_reg_indicator_value"] = 42.0
        return df
"""

_VALID_TESTS_CODE = """\
import pandas as pd
from fwbg_sdk import PluginPhase


def test_plugin_runs():
    import importlib.util, sys, pathlib
    # Import the plugin from the same directory as this test file.
    init = pathlib.Path(__file__).parent / "__init__.py"
    spec = importlib.util.spec_from_file_location("_test_treg", init)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_test_treg"] = module
    spec.loader.exec_module(module)
    cls = module.TestIndicatorPlugin
    df = pd.DataFrame({"close": [1.0, 2.0]})
    result = cls().run(df, {})
    assert "test_reg_indicator_value" in result.columns
"""

_FAILING_TESTS_CODE = """\
def test_always_fails():
    assert False, "deliberate failure"
"""


@pytest.fixture(autouse=True)
def clean_registry(tmp_path, monkeypatch):
    """Redirect user-plugins dir to tmp_path and reset registry around each test."""
    import fwbg.pipeline.registry as reg_mod
    import fwbg.api.plugins as plugins_mod

    monkeypatch.setattr(reg_mod, "get_user_plugins_dir", lambda: tmp_path)
    monkeypatch.setattr(plugins_mod, "get_user_plugins_dir", lambda: tmp_path, raising=False)

    reset_registry()
    get_plugin_registry.cache_clear()
    yield
    reset_registry()
    get_plugin_registry.cache_clear()


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def _register(client, slug="test_reg_indicator", kind="indicator", **kwargs):
    payload = {
        "slug": slug,
        "python_code": _VALID_PLUGIN_CODE.replace("test_reg_indicator", slug),
        "kind": kind,
        "description": "A test indicator",
        **kwargs,
    }
    return client.post("/api/plugins", json=payload)


class TestRegisterPlugin:
    def test_valid_plugin_appears_in_list(self, client):
        resp = _register(client)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["fqn"] == "agent-authored:test_reg_indicator"
        assert data["category"] == "indicators"

        list_resp = client.get("/api/plugins")
        assert list_resp.status_code == 200
        fqns = [p["fqn"] for p in list_resp.json()]
        assert "agent-authored:test_reg_indicator" in fqns

    def test_get_plugin_detail_after_register(self, client):
        _register(client)
        resp = client.get("/api/plugins/agent-authored:test_reg_indicator")
        assert resp.status_code == 200
        assert resp.json()["name"] == "test_reg_indicator"

    def test_duplicate_returns_409(self, client):
        _register(client)
        resp = _register(client)
        assert resp.status_code == 409

    def test_overwrite_replaces_plugin(self, client):
        _register(client)
        resp = _register(client, overwrite=True)
        assert resp.status_code == 200

    def test_invalid_code_returns_422(self, client):
        resp = client.post("/api/plugins", json={
            "slug": "bad_plugin",
            "python_code": "this is not valid python!!!",
            "kind": "indicator",
        })
        assert resp.status_code == 422

    def test_wrong_slug_in_code_returns_422(self, client):
        resp = client.post("/api/plugins", json={
            "slug": "other_name",
            "python_code": _VALID_PLUGIN_CODE,
            "kind": "indicator",
        })
        assert resp.status_code == 422
        assert "other_name" in resp.text

    def test_unknown_kind_returns_422(self, client):
        resp = client.post("/api/plugins", json={
            "slug": "test_reg_indicator",
            "python_code": _VALID_PLUGIN_CODE,
            "kind": "unknownkind",
        })
        assert resp.status_code == 422

    def test_failing_tests_returns_422(self, client):
        resp = client.post("/api/plugins", json={
            "slug": "test_reg_indicator",
            "python_code": _VALID_PLUGIN_CODE,
            "kind": "indicator",
            "tests_code": _FAILING_TESTS_CODE,
        })
        assert resp.status_code == 422
        assert "failed" in resp.text.lower()

    def test_passing_tests_accepted(self, client):
        resp = client.post("/api/plugins", json={
            "slug": "test_reg_indicator",
            "python_code": _VALID_PLUGIN_CODE,
            "kind": "indicator",
            "tests_code": _VALID_TESTS_CODE,
        })
        assert resp.status_code == 200

    def test_spec_md_written_and_served(self, client):
        _register(client, spec_md="# Spec\ncapability: test signal")
        resp = client.get("/api/plugins/agent-authored:test_reg_indicator/spec")
        assert resp.status_code == 200
        assert "capability" in resp.json()["spec"]

    def test_source_served_after_register(self, client):
        _register(client)
        resp = client.get("/api/plugins/agent-authored:test_reg_indicator/source")
        assert resp.status_code == 200
        assert "TestIndicatorPlugin" in resp.json()["source"]

    def test_bundle_manifest_created(self, client, tmp_path):
        _register(client)
        bundle_manifest = tmp_path / "agent-authored" / "manifest.json"
        assert bundle_manifest.exists()
        data = json.loads(bundle_manifest.read_text())
        assert data["name"] == "agent-authored"

    def test_exit_strategy_kind_routes_to_correct_category(self, client):
        exit_code = _VALID_PLUGIN_CODE.replace(
            "test_reg_indicator", "test_reg_exit"
        ).replace(
            "PluginPhase.INDICATORS", "PluginPhase.EXIT_STRATEGIES"
        )
        resp = client.post("/api/plugins", json={
            "slug": "test_reg_exit",
            "python_code": exit_code,
            "kind": "exit_strategy",
        })
        assert resp.status_code == 200
        assert resp.json()["category"] == "exit_strategies"


class TestNamespaceFilter:
    def test_namespace_agent_authored_returns_only_agent_plugins(self, client):
        _register(client)
        resp = client.get("/api/plugins", params={"namespace": "agent-authored"})
        assert resp.status_code == 200
        fqns = [p["fqn"] for p in resp.json()]
        assert fqns, "expected at least the registered agent plugin"
        assert all(fqn.startswith("agent-authored:") for fqn in fqns)
        assert "agent-authored:test_reg_indicator" in fqns

    def test_namespace_fwbg_core_excludes_agent_plugins(self, client):
        _register(client)
        resp = client.get("/api/plugins", params={"namespace": "fwbg-core"})
        assert resp.status_code == 200
        fqns = [p["fqn"] for p in resp.json()]
        assert all(fqn.startswith("fwbg-core:") for fqn in fqns)
        assert "agent-authored:test_reg_indicator" not in fqns

    def test_unknown_namespace_returns_empty(self, client):
        resp = client.get("/api/plugins", params={"namespace": "does-not-exist"})
        assert resp.status_code == 200
        assert resp.json() == []
