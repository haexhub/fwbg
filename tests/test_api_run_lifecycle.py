"""Lifecycle contracts for the process-local run service."""

from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

from fwbg.api.run_service import RunCapacityError, RunService


class _FakeProcess:
    pid = 1234

    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def test_slot_reservation_is_atomic(tmp_path):
    service = RunService()
    jobs = {}
    lock = threading.Lock()

    def reserve():
        return service.reserve(jobs, lock, tmp_path, 1, "demo")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _reserve_result(reserve), range(2)))

    assert sorted(result[0] for result in results) == ["capacity", "ok"]
    assert len(jobs) == 1


def _reserve_result(reserve):
    try:
        return "ok", reserve()
    except RunCapacityError:
        return "capacity", None


def test_same_timestamp_reservations_have_unique_ids(tmp_path, monkeypatch):
    service = RunService()
    jobs = {}
    lock = threading.Lock()

    class _FrozenDateTime:
        @classmethod
        def now(cls):
            from datetime import datetime

            return datetime(2026, 1, 2, 3, 4, 5, 6)

    monkeypatch.setattr("fwbg.api.run_service.datetime", _FrozenDateTime)
    first = service.reserve(jobs, lock, tmp_path, 2, "demo")
    second = service.reserve(jobs, lock, tmp_path, 2, "demo")

    assert first["job_id"] != second["job_id"]


def test_failed_and_cancelled_states_are_terminal_and_persisted(tmp_path, monkeypatch):
    service = RunService()
    jobs = {}
    lock = threading.Lock()

    failed = service.reserve(jobs, lock, tmp_path, 2, "failed")
    service.fail_spawn(jobs, lock, tmp_path, failed["job_id"], "spawn failed")
    assert service.disk_state(tmp_path, failed["job_id"])["status"] == "failed"

    cancelled = service.reserve(jobs, lock, tmp_path, 2, "cancelled")
    process = _FakeProcess()
    service.attach(jobs, lock, tmp_path, cancelled["job_id"], process, Path("out"), Path("err"), [])
    killed = []
    monkeypatch.setattr("fwbg.api.run_service.os.kill", lambda pid, sig: killed.append((pid, sig)))
    result = service.cancel(jobs, lock, tmp_path, cancelled["job_id"])

    assert result["status"] == "cancelled"
    assert service.disk_state(tmp_path, cancelled["job_id"])["status"] == "cancelled"
    assert killed


def test_orphaned_running_state_becomes_failed_after_restart(tmp_path):
    service = RunService()
    jobs = {}
    lock = threading.Lock()
    job = service.reserve(jobs, lock, tmp_path, 1, "crashed")

    restarted = RunService().disk_state(tmp_path, job["job_id"])

    assert restarted["status"] == "failed"
    assert "restart" in restarted["error_message"]
