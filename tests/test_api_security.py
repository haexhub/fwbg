"""Security behaviour of the API: auth coverage, CORS order, plugin opt-in, secret masking."""

import pytest
from fastapi.testclient import TestClient

from fwbg.api import create_app
from fwbg.api import datasources as ds_api
from fwbg.core.data_sources import DBSourceConfig, RESTSourceConfig

_ORIGIN = "http://localhost:3000"
_SECRET = "very-secret-value"


@pytest.fixture
def keyed_client(monkeypatch):
    monkeypatch.setenv("FWBG_API_KEY", "s3cret")
    with TestClient(create_app()) as c:
        yield c


# ── auth coverage / CORS ordering ────────────────────────────────────────────


def test_unknown_and_non_api_paths_require_the_key(keyed_client):
    assert keyed_client.get("/api/strategies").status_code == 401
    assert keyed_client.get("/nonexistent").status_code == 401
    assert keyed_client.get("//api/strategies").status_code == 401


def test_docs_paths_stay_public(keyed_client):
    assert keyed_client.get("/openapi.json").status_code == 200


def test_preflight_is_answered_without_key_and_grants_no_data(keyed_client):
    resp = keyed_client.options(
        "/api/strategies",
        headers={"Origin": _ORIGIN, "Access-Control-Request-Method": "GET"},
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == _ORIGIN
    # The actual request still needs the key.
    resp = keyed_client.get("/api/strategies", headers={"Origin": _ORIGIN})
    assert resp.status_code == 401
    # ... and the rejection carries CORS headers, so a browser can read it.
    assert resp.headers["access-control-allow-origin"] == _ORIGIN


def test_preflight_from_untrusted_origin_does_not_reach_a_route(keyed_client):
    resp = keyed_client.options(
        "/api/strategies",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code in (400, 401)
    assert "access-control-allow-origin" not in resp.headers


def test_valid_key_is_accepted(keyed_client):
    assert (
        keyed_client.get("/api/strategies", headers={"X-API-Key": "s3cret"}).status_code
        == 200
    )


# ── plugin registration opt-in ───────────────────────────────────────────────

_PLUGIN = {
    "slug": "sec_probe",
    "python_code": "raise SystemExit('executed')",
    "kind": "indicator",
}


def test_plugin_registration_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FWBG_ALLOW_PLUGIN_UPLOAD", raising=False)
    with TestClient(create_app()) as c:
        resp = c.post("/api/plugins", json=_PLUGIN)
    assert resp.status_code == 403
    assert "FWBG_ALLOW_PLUGIN_UPLOAD" in resp.json()["detail"]


@pytest.mark.parametrize("value", ["0", "", "no", "false"])
def test_plugin_registration_off_values(monkeypatch, value):
    monkeypatch.setenv("FWBG_ALLOW_PLUGIN_UPLOAD", value)
    with TestClient(create_app()) as c:
        assert c.post("/api/plugins", json=_PLUGIN).status_code == 403


def test_plugin_registration_enabled_reaches_validation(monkeypatch, tmp_path):
    import fwbg.pipeline.registry as reg_mod

    monkeypatch.setenv("FWBG_ALLOW_PLUGIN_UPLOAD", "1")
    monkeypatch.setattr(reg_mod, "get_user_plugins_dir", lambda: tmp_path)
    with TestClient(create_app()) as c:
        resp = c.post("/api/plugins", json={**_PLUGIN, "python_code": "x = 1"})
    # Past the gate: the code is rejected for lacking a plugin class, not for the flag.
    assert resp.status_code == 422


def test_plugin_registration_needs_the_key_even_when_enabled(monkeypatch):
    monkeypatch.setenv("FWBG_API_KEY", "s3cret")
    monkeypatch.setenv("FWBG_ALLOW_PLUGIN_UPLOAD", "1")
    with TestClient(create_app()) as c:
        assert c.post("/api/plugins", json=_PLUGIN).status_code == 401


# ── datasource secret masking ────────────────────────────────────────────────


@pytest.fixture
def secret_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(ds_api, "save_source_config", lambda source: None)
    sources = {
        "rest1": RESTSourceConfig(
            name="rest1",
            base_url="https://example.invalid",
            api_key=_SECRET,
            headers={"Authorization": f"Bearer {_SECRET}", "Accept": "text/csv"},
        ),
        "db1": DBSourceConfig(
            name="db1", connection_string=f"postgresql://user:{_SECRET}@host:5432/db"
        ),
    }
    monkeypatch.setattr(ds_api, "_DATA_SOURCES", sources)
    with TestClient(create_app()) as c:
        yield c, sources


def test_datasource_list_masks_secrets(secret_sources):
    client, _ = secret_sources
    resp = client.get("/api/datasources")
    assert resp.status_code == 200
    assert _SECRET not in resp.text
    by_name = {d["name"]: d for d in resp.json()}
    assert by_name["rest1"]["api_key"] == ds_api.SECRET_MASK
    assert by_name["rest1"]["headers"] == {
        "Authorization": ds_api.SECRET_MASK,
        "Accept": "text/csv",
    }
    assert (
        by_name["db1"]["connection_string"]
        == f"postgresql://user:{ds_api.SECRET_MASK}@host:5432/db"
    )


def test_put_with_masked_values_keeps_stored_secrets(secret_sources):
    client, sources = secret_sources
    body = client.get("/api/datasources").json()
    rest = next(d for d in body if d["name"] == "rest1")
    db = next(d for d in body if d["name"] == "db1")
    rest["description"] = "edited"
    assert client.put("/api/datasources/rest1", json=rest).status_code == 200
    assert client.put("/api/datasources/db1", json=db).status_code == 200
    assert sources["rest1"].api_key == _SECRET
    assert sources["rest1"].headers["Authorization"] == f"Bearer {_SECRET}"
    assert sources["rest1"].description == "edited"
    assert (
        sources["db1"].connection_string == f"postgresql://user:{_SECRET}@host:5432/db"
    )


def test_put_with_new_secret_replaces_it(secret_sources):
    client, sources = secret_sources
    resp = client.put("/api/datasources/rest1", json={"api_key": "new-key"})
    assert resp.status_code == 200
    assert sources["rest1"].api_key == "new-key"
    assert "new-key" not in resp.text


def test_put_mask_for_unknown_header_is_rejected(secret_sources):
    client, _ = secret_sources
    resp = client.put(
        "/api/datasources/rest1",
        json={"headers": {"X-Other-Token": ds_api.SECRET_MASK}},
    )
    assert resp.status_code == 400


# ── compose contract ─────────────────────────────────────────────────────────


def test_compose_requires_api_key_and_has_no_unauthenticated_bypass():
    from pathlib import Path

    import yaml

    compose = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text()
    )
    env = compose["services"]["api"]["environment"]
    assert not any("FWBG_ALLOW_UNAUTHENTICATED_API" in e for e in env)
    key = next(e for e in env if e.startswith("FWBG_API_KEY="))
    assert ":?" in key
    assert "FWBG_ALLOW_PLUGIN_UPLOAD=${FWBG_ALLOW_PLUGIN_UPLOAD:-0}" in env
