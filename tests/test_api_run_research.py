"""POST /api/runs/start passes the research fields and preflights the gate.

The preflight runs the CLI with ``--check-json`` before the real run is
spawned, so a refused run becomes an HTTP error instead of a run that dies.
"""

import argparse
import json
import subprocess

import pytest
from fastapi.testclient import TestClient

import fwbg.api.run_preflight as preflight_mod
import fwbg.api.runs as runs_mod
from fwbg.api import create_app
from fwbg.research.runs import CHECK_PREFIX, add_run_arguments

DENIAL_409 = {"code": "DUPLICATE_RUN", "status": 409, "message": "already ran"}
DENIAL_422 = {"code": "MISSING_NOTE", "status": 422, "message": "a note is required"}
ALLOWED = {
    "allowed": True,
    "kind": "iteration",
    "fingerprint": "sha256:" + "0" * 64,
    "symbols": ["EURUSD"],
    "window": {"start": None, "end": "2025-01-01"},
    "denials": [],
}


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

    calls = {
        "preflight": [],
        "preflight_env": [],
        "spawn": [],
        "spawn_env": [],
        "result": None,
        "during_preflight": None,
        "results_dir": tmp_path / "results",
    }

    def fake_run(cmd, **kwargs):
        calls["preflight"].append(cmd)
        calls["preflight_env"].append(kwargs["env"])
        if calls["during_preflight"]:
            calls["during_preflight"](cmd)
        result = calls["result"]
        if isinstance(result, BaseException):
            raise result
        return result

    def fake_spawn(cmd, env, run_dir):
        calls["spawn"].append(cmd)
        calls["spawn_env"].append(env)
        return _FakeProcess(), run_dir / "cli_stdout.log", run_dir / "cli_stderr.log"

    monkeypatch.setattr(preflight_mod, "_run", fake_run)
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


def _job_id(cmd):
    return cmd[cmd.index("--run-id") + 1]


def _assert_slot_released(client, calls):
    refused_job = _job_id(calls["preflight"][-1])
    assert not (calls["results_dir"] / refused_job).exists()
    calls["respond"](ALLOWED)
    assert _start(client).status_code == 200


def _assert_preflight_failed(resp, *fragments):
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["message"] == "research preflight failed"
    assert "denials" not in detail
    for fragment in fragments:
        assert fragment in detail["output"]


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
    calls["respond"](ALLOWED)

    resp = _start(client, research_id="S-0001", note="tighter stop, see card", reason="rerun")

    assert resp.status_code == 200
    assert len(calls["spawn"]) == 1
    cmd = calls["spawn"][0]
    assert "--research-id=S-0001" in cmd
    assert "--note=tighter stop, see card" in cmd
    assert "--reason=rerun" in cmd
    assert _job_id(cmd) == resp.json()["job_id"]
    assert "--check-json" not in cmd
    assert calls["preflight"] == [[*cmd, "--check-json"]]
    assert calls["preflight_env"] == calls["spawn_env"]


def test_research_text_starting_with_a_dash_parses_in_the_cli(api):
    client, calls = api
    calls["respond"](ALLOWED)

    assert (
        _start(
            client, research_id="S-0001", note="-5% tighter stop", reason="-x", holdout=True
        ).status_code
        == 200
    )

    cmd = calls["spawn"][0]
    assert "--note=-5% tighter stop" in cmd
    parser = argparse.ArgumentParser()
    add_run_arguments(parser)
    args, _ = parser.parse_known_args(cmd[3:])
    assert (args.research_id, args.note, args.reason) == ("S-0001", "-5% tighter stop", "-x")
    assert args.holdout is True


def test_holdout_reaches_the_cli_and_its_preflight(api):
    client, calls = api
    calls["respond"]({**ALLOWED, "kind": "holdout", "window": {"start": None, "end": None}})

    assert _start(client, research_id="S-0001", note="n", holdout=True).status_code == 200

    cmd = calls["spawn"][0]
    assert "--holdout" in cmd
    assert calls["preflight"] == [[*cmd, "--check-json"]]


def test_holdout_is_off_by_default(api):
    client, calls = api
    calls["respond"](ALLOWED)

    assert _start(client, research_id="S-0001", note="n").status_code == 200

    assert "--holdout" not in calls["spawn"][0]


def test_allowed_run_without_reason_omits_the_flag(api):
    client, calls = api
    calls["respond"](ALLOWED)

    assert _start(client, research_id="S-0001", note="n").status_code == 200

    assert not any(item.startswith("--reason") for item in calls["spawn"][0])


def test_cli_failing_before_its_gate_returns_422_with_both_stream_tails(api):
    client, calls = api
    calls["respond"](
        returncode=1,
        stdout="Keine Datendateien gefunden\n",
        stderr="Traceback: boom\n",
    )

    resp = _start(client)

    _assert_preflight_failed(resp, "Keine Datendateien gefunden", "Traceback: boom")
    assert calls["spawn"] == []
    _assert_slot_released(client, calls)


def test_payload_contradicting_the_exit_code_is_a_preflight_failure(api):
    client, calls = api
    calls["respond"](ALLOWED, returncode=1)

    resp = _start(client)

    _assert_preflight_failed(resp)
    assert CHECK_PREFIX not in resp.json()["detail"]["output"]


@pytest.mark.parametrize("denials", ["nope", [DENIAL_409, "nope"], None])
def test_malformed_denials_are_a_preflight_failure(api, denials):
    client, calls = api
    calls["respond"]({"allowed": False, "denials": denials}, returncode=2)

    _assert_preflight_failed(_start(client))
    assert calls["spawn"] == []
    _assert_slot_released(client, calls)


def test_preflight_timeout_returns_504_and_releases_the_slot(api):
    client, calls = api
    calls["result"] = subprocess.TimeoutExpired(["fwbg"], 120)

    resp = _start(client)

    assert resp.status_code == 504
    assert resp.json()["detail"] == "research preflight timed out"
    assert calls["spawn"] == []
    _assert_slot_released(client, calls)


def test_run_cancelled_during_the_preflight_is_not_spawned(api):
    client, calls = api
    calls["respond"](ALLOWED)

    def cancel(cmd):
        job_id = _job_id(cmd)
        response = client.post(f"/api/runs/{job_id}/cancel")
        assert response.status_code == 200

    calls["during_preflight"] = cancel

    resp = _start(client)

    assert resp.status_code == 409
    assert resp.json()["detail"] == "run was cancelled before it started"
    assert calls["spawn"] == []
    job_id = _job_id(calls["preflight"][0])
    assert client.get(f"/api/runs/{job_id}").json()["status"] == "cancelled"


def test_gate_off_spawns_without_a_preflight(api, monkeypatch):
    client, calls = api
    monkeypatch.setattr(runs_mod, "gate_enabled", lambda: False)

    assert _start(client).status_code == 200

    assert calls["preflight"] == []
    assert len(calls["spawn"]) == 1
