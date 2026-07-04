"""Regression tests for the run-subprocess spawn path.

The API used to start CLI runs with `stdout=subprocess.PIPE` and never read
the pipes. Once the CLI had written ~64KB the OS pipe filled up and the CLI
blocked on write — freezing its progress display thread, the progress-queue
reader behind it, and finally the pool workers: the whole backtest hung
forever at a frozen progress fraction. Observed live on the first long
full-history run (GBPUSD M15). stdout/stderr now go to files in the run dir.
"""
import os

import pytest
import sys

from fwbg.api.runs import _job_error_output, _spawn_cli_process


def test_child_flooding_stdout_does_not_deadlock(tmp_path):
    # Write far more than any OS pipe buffer (64KB) to stdout. With the old
    # PIPE wiring this child would block forever; with file redirection it
    # must exit promptly.
    cmd = [sys.executable, "-c", "import sys; sys.stdout.write('x' * 1_000_000)"]
    process, stdout_path, stderr_path = _spawn_cli_process(
        cmd, os.environ.copy(), tmp_path / "run"
    )
    assert process.wait(timeout=30) == 0
    assert stdout_path.stat().st_size == 1_000_000
    assert stderr_path.stat().st_size == 0


def test_run_dir_is_created_and_logs_land_there(tmp_path):
    run_dir = tmp_path / "test_results" / "20990101_000000_abc123"
    cmd = [sys.executable, "-c", "print('hello run')"]
    process, stdout_path, stderr_path = _spawn_cli_process(cmd, os.environ.copy(), run_dir)
    process.wait(timeout=30)
    assert run_dir.is_dir()
    assert stdout_path == run_dir / "cli_stdout.log"
    assert "hello run" in stdout_path.read_text()


def test_job_error_output_prefers_stderr_tail(tmp_path):
    cmd = [
        sys.executable,
        "-c",
        "import sys; print('noise on stdout'); sys.stderr.write('boom: ' + 'y' * 600); sys.exit(3)",
    ]
    process, stdout_path, stderr_path = _spawn_cli_process(
        cmd, os.environ.copy(), tmp_path / "run"
    )
    assert process.wait(timeout=30) == 3

    job = {"stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}
    output = _job_error_output(job, limit=500)
    assert len(output) == 500  # tail-limited
    assert output.endswith("y")


def test_job_error_output_falls_back_to_stdout_then_empty(tmp_path):
    cmd = [sys.executable, "-c", "print('only stdout info'); import sys; sys.exit(1)"]
    process, stdout_path, stderr_path = _spawn_cli_process(
        cmd, os.environ.copy(), tmp_path / "run"
    )
    assert process.wait(timeout=30) == 1
    job = {"stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}
    assert "only stdout info" in _job_error_output(job)

    assert _job_error_output({"stdout_path": str(tmp_path / "missing.log")}) == ""
    assert _job_error_output({}) == ""


# ---------------------------------------------------------------------------
# Concurrency gate: with FWBG_MAX_CONCURRENT_RUNS=1 exactly one backtest may
# run; a job whose process already exited must not occupy the slot (statuses
# are otherwise only refreshed when a status endpoint happens to be polled).
# The gate sits before the strategy lookup, so a passed gate shows as 404
# for a nonexistent strategy while a full slot shows as 429.
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, returncode):
        self._rc = returncode
        self.returncode = returncode

    def poll(self):
        return self._rc


@pytest.fixture
def _single_slot(monkeypatch):
    import fwbg.api.runs as runs_mod

    monkeypatch.setattr(runs_mod, "MAX_CONCURRENT_RUNS", 1)
    monkeypatch.setattr(runs_mod, "_active_jobs", {})
    return runs_mod


def _post_start(client):
    return client.post("/api/runs/start", json={"strategy_name": "no_such_strategy"})


def test_active_run_occupies_the_single_slot(_single_slot):
    from fastapi.testclient import TestClient

    from fwbg.api import create_app

    _single_slot._active_jobs["job_a"] = {
        "job_id": "job_a", "status": "running", "process": _FakeProc(None),
    }
    with TestClient(create_app()) as client:
        resp = _post_start(client)
    assert resp.status_code == 429


def test_stale_finished_job_does_not_block_the_slot(_single_slot):
    from fastapi.testclient import TestClient

    from fwbg.api import create_app

    _single_slot._active_jobs["job_a"] = {
        "job_id": "job_a", "status": "running", "process": _FakeProc(0),
    }
    with TestClient(create_app()) as client:
        resp = _post_start(client)
    # Gate passed (stale status refreshed to completed) → 404 for the
    # nonexistent strategy, not 429.
    assert resp.status_code == 404
    assert _single_slot._active_jobs["job_a"]["status"] == "completed"
