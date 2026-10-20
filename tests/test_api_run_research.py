"""POST /api/runs/start passes the research fields and preflights the gate.

The preflight runs the CLI with ``--check-json`` before the real run is
spawned, so a refused run becomes an HTTP error instead of a run that dies.
"""

import json
import subprocess

import pytest
from fastapi.testclient import TestClient

import fwbg.api.runs as runs_mod
from fwbg.api import create_app
from fwbg.research.runs import CHECK_PREFIX

DENIAL_409 = {"code": "DUPLICATE_RUN", "status": 409, "message": "already ran"}
DENIAL_422 = {"code": "MISSING_NOTE", "status": 422, "message": "a note is required"}


class _FakeProcess:
    pid = 4242

    def poll(self):
        return None


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A client on a tmp workspace with a faked preflight and spawn."""
    strategies = tmp_path / "ws" / "strategies" / "configs"
    strategies.mkdir(parents=True)
    (strategies / "strat.json").write_text(json.dumps({"name": "strat"}))
    monkeypatch.setenv("FWBG_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("FWBG_STRATEGIES_DIR", str(strategies))
    monkeypatch.setenv("FWBG_TEST_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setattr(runs_mod, "_active_jobs", {})
    monkeypatch.setattr(runs_mod, "MAX_CONCURRENT_RUNS", 1)
    monkeypatch.setattr(runs_mod, "gate_enabled", lambda: True)

    calls = {"preflight": [], "spawn": [], "result": None}

    def fake_run(cmd, **kwargs):
        calls["preflight"].append(cmd)
        result = calls["result"]
        if isinstance(result, BaseException):
            raise result
        return result

    def fake_spawn(cmd, env, run_dir):
        calls["spawn"].append(cmd)
        return _FakeProcess(), run_dir / "cli_stdout.log", run_dir / "cli_stderr.log"

    monkeypatch.setattr("fwbg.api.run_preflight.subprocess.run", fake_run)
    monkeypatch.setattr(runs_mod, "_spawn_cli_process", fake_spawn)

    def respond(payload=None, returncode=0, stdout="", stderr=""):
        if payload is not None:
            stdout += CHECK_PREFIX + json.dumps(payload) + "\n"
        calls["result"] = subprocess.CompletedProcess([], returncode, stdout, stderr)

    calls["respond"] = respond
    with TestClient(create_app()) as client:
        yield client, calls


def _start(client, **extra):
    return client.post("/api/runs/start", json={"strategy_name": "strat", **extra})


def _refusal(*denials):
    return {"allowed": False, "denials": list(denials)}


def _assert_slot_released(client, calls):
    calls["respond"]({"allowed": True, "denials": []})
    assert _start(client).status_code == 200


def test_refusal_with_a_409_denial_returns_409_and_releases_the_slot(api):
    client, calls = api
    calls["respond"](_refusal(DENIAL_409, DENIAL_422), returncode=2)

    resp = _start(client, research_id="S-0001", note="n")

    assert resp.status_code == 409
    assert resp.json()["detail"]["denials"] == [DENIAL_409, DENIAL_422]
    assert calls["spawn"] == []
    _assert_slot_released(client, calls)


def test_refusal_with_only_422_denials_returns_422(api):
    client, calls = api
    calls["respond"](_refusal(DENIAL_422), returncode=2)

    resp = _start(client)

    assert resp.status_code == 422
    assert resp.json()["detail"]["denials"] == [DENIAL_422]
    assert calls["spawn"] == []


def test_allowed_run_spawns_with_the_research_fields(api):
    client, calls = api
    calls["respond"]({"allowed": True, "denials": []})

    resp = _start(client, research_id="S-0001", note="tighter stop, see card", reason="rerun")

    assert resp.status_code == 200
    assert len(calls["spawn"]) == 1
    cmd = calls["spawn"][0]
    assert cmd[cmd.index("--research-id") + 1] == "S-0001"
    assert cmd[cmd.index("--note") + 1] == "tighter stop, see card"
    assert cmd[cmd.index("--reason") + 1] == "rerun"
    assert cmd[cmd.index("--run-id") + 1] == resp.json()["job_id"]
    assert "--check-json" not in cmd
    assert calls["preflight"] == [[*cmd, "--check-json"]]


def test_allowed_run_without_reason_omits_the_flag(api):
    client, calls = api
    calls["respond"]({"allowed": True, "denials": []})

    assert _start(client, research_id="S-0001", note="n").status_code == 200

    assert "--reason" not in calls["spawn"][0]


def test_cli_failing_before_its_gate_returns_422_with_both_stream_tails(api):
    client, calls = api
    calls["respond"](
        returncode=1,
        stdout="Keine Datendateien gefunden\n",
        stderr="Traceback: boom\n",
    )

    resp = _start(client)

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail.startswith("research preflight failed")
    assert "Keine Datendateien gefunden" in detail
    assert "Traceback: boom" in detail
    assert calls["spawn"] == []
    _assert_slot_released(client, calls)


def test_payload_contradicting_the_exit_code_is_a_preflight_failure(api):
    client, calls = api
    calls["respond"]({"allowed": True, "denials": []}, returncode=1)

    resp = _start(client)

    assert resp.status_code == 422
    assert CHECK_PREFIX not in resp.json()["detail"]


def test_preflight_timeout_returns_504_and_releases_the_slot(api):
    client, calls = api
    calls["result"] = subprocess.TimeoutExpired(["fwbg"], 120)

    resp = _start(client)

    assert resp.status_code == 504
    assert calls["spawn"] == []
    _assert_slot_released(client, calls)


def test_gate_off_spawns_without_a_preflight(api, monkeypatch):
    client, calls = api
    monkeypatch.setattr(runs_mod, "gate_enabled", lambda: False)

    assert _start(client).status_code == 200

    assert calls["preflight"] == []
    assert len(calls["spawn"]) == 1
