"""Offline contract tests for the authenticated FWBG MCP HTTP client."""

import httpx
import pytest

from fwbg_mcp import server

def _install_transport(monkeypatch, handler):
    """Route the module's synchronous helpers through an in-memory transport."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server.httpx, "get", client.get)
    monkeypatch.setattr(server.httpx, "post", client.post)
    monkeypatch.setattr(server.httpx, "put", client.put)
    return client


def test_http_helpers_send_x_api_key(monkeypatch):
    monkeypatch.setenv("FWBG_API_KEY", "synthetic-key")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"ok": True})

    client = _install_transport(monkeypatch, handler)

    server._get("/one", query="value")
    server._post("/two", {"value": 2})
    server._put("/three", {"value": 3})
    client.close()

    assert len(calls) == 3
    assert all(call.headers["X-API-Key"] == "synthetic-key" for call in calls)
    assert all("synthetic-key" not in str(call.url) for call in calls)


def test_save_strategy_creates_missing_strategy_with_post(monkeypatch):
    monkeypatch.setenv("FWBG_API_KEY", "synthetic-key")
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(404, json={"detail": "missing"})
        return httpx.Response(200, json={"status": "created"})

    client = _install_transport(monkeypatch, handler)

    result = server.save_strategy("new_strategy", {"model": {"type": "xgboost"}})
    client.close()

    assert result == {"status": "created"}
    assert [call.method for call in calls] == ["GET", "POST"]
    assert calls[1].content == b'{"name":"new_strategy","data":{"model":{"type":"xgboost"}}}'
    assert calls[1].headers["X-API-Key"] == "synthetic-key"


def test_save_strategy_updates_existing_strategy_with_put(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"name": "existing"})
        return httpx.Response(200, json={"status": "updated"})

    client = _install_transport(monkeypatch, handler)

    assert server.save_strategy("existing", {"v": 2}) == {"status": "updated"}
    client.close()
    assert [call.method for call in calls] == ["GET", "PUT"]


@pytest.mark.parametrize("status_code", [401, 500])
def test_save_strategy_does_not_mask_auth_or_server_errors(monkeypatch, status_code):
    def handler(request):
        return httpx.Response(status_code, json={"detail": "failure"})

    client = _install_transport(monkeypatch, handler)
    with pytest.raises(httpx.HTTPStatusError):
        server.save_strategy("protected", {})
    client.close()


def test_start_run_normalizes_job_id_to_run_id(monkeypatch):
    monkeypatch.setattr(
        server,
        "_post",
        lambda path, body: {"job_id": "run-123", "status": "running"},
    )

    result = server.start_run("strategy")

    assert result["job_id"] == result["run_id"] == "run-123"


def test_wait_for_run_handles_cancelled_status(monkeypatch):
    monkeypatch.setattr(
        server,
        "_get",
        lambda path, **params: {"status": "cancelled", "message": "user stopped"},
    )

    with pytest.raises(RuntimeError, match="cancelled"):
        server.wait_for_run("run-123", timeout_minutes=1)


def test_wait_for_run_logs_progress_to_stderr(monkeypatch, capsys):
    responses = iter(
        [
            {"status": "running", "current_stage": "fit", "progress_fraction": 0.5},
            {"status": "completed"},
        ]
    )
    monkeypatch.setattr(server, "_get", lambda path, **params: next(responses))
    monkeypatch.setattr(server, "get_run_results", lambda run_id: {"run_id": run_id})
    monkeypatch.setattr(server.time, "sleep", lambda seconds: None)

    assert server.wait_for_run("run-123", timeout_minutes=1) == {"run_id": "run-123"}
    captured = capsys.readouterr()
    assert "fit" in captured.err
    assert captured.out == ""


def test_list_presets_uses_api_endpoint(monkeypatch):
    calls = []

    def fake_get(path, **params):
        calls.append((path, params))
        return [{"id": "pipeline_v1", "name": "Pipeline"}]

    monkeypatch.setattr(server, "_get", fake_get)

    assert server.list_presets("pipelines") == ["pipeline_v1"]
    assert calls == [("/presets/pipelines", {})]
