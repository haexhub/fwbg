"""Thread-safe lifecycle state for API-launched optimizer runs.

The API is intentionally process-local.  Deployments using multiple worker
processes must route run management to one worker or provide an external
queue/state service; this module only coordinates threads in one process.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, MutableMapping


ACTIVE_STATUSES = frozenset({"starting", "running"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
STATE_FILENAME = "run_state.json"


class RunCapacityError(RuntimeError):
    """Raised when all configured run slots are reserved."""


class RunService:
    """Lifecycle operations shared by the run endpoints.

    ``jobs`` and ``lock`` are supplied by ``runs.py`` so existing tests and
    integrations can continue to inspect the process-local registry.
    """

    @staticmethod
    def _state_path(results_dir: Path, job_id: str) -> Path:
        return Path(results_dir) / job_id / STATE_FILENAME

    @staticmethod
    def _write_state(results_dir: Path, job: MutableMapping[str, Any]) -> None:
        path = RunService._state_path(Path(results_dir), str(job["job_id"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: value
            for key, value in job.items()
            if key not in {"process", "cmd"}
        }
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temp, path)

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def _new_id(results_dir: Path, jobs: MutableMapping[str, Any]) -> str:
        # Microseconds plus UUID make same-timestamp requests distinct while
        # retaining the sortable timestamp prefix used by existing clients.
        prefix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        while True:
            job_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
            if job_id not in jobs and not (results_dir / job_id).exists():
                return job_id

    @staticmethod
    def _reap_locked(jobs: MutableMapping[str, Any], results_dir: Path) -> None:
        for job in list(jobs.values()):
            if job.get("status") not in ACTIVE_STATUSES:
                continue
            process = job.get("process")
            if process is None:
                continue
            returncode = process.poll()
            if returncode is None:
                continue
            status = "completed" if returncode == 0 else "failed"
            RunService._finish_locked(
                jobs,
                results_dir,
                str(job["job_id"]),
                status,
                None if returncode == 0 else f"Process exited with code {returncode}",
            )

    @staticmethod
    def _finish_locked(
        jobs: MutableMapping[str, Any],
        results_dir: Path,
        job_id: str,
        status: str,
        error_message: str | None,
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"invalid terminal run status: {status}")
        job = jobs.get(job_id)
        if job is None:
            return
        # Cancellation is terminal even if the child exits later.
        if job.get("status") == "cancelled" and status != "cancelled":
            return
        job["status"] = status
        job["finished_at"] = RunService._now()
        if error_message:
            job["error_message"] = error_message
        RunService._write_state(results_dir, job)

    def reserve(
        self,
        jobs: MutableMapping[str, Any],
        lock: threading.Lock,
        results_dir: Path,
        max_concurrent: int,
        strategy_name: str,
    ) -> dict[str, Any]:
        """Atomically reserve a slot before any subprocess spawn occurs."""
        results_dir = Path(results_dir)
        with lock:
            self._reap_locked(jobs, results_dir)
            running = sum(1 for job in jobs.values() if job.get("status") in ACTIVE_STATUSES)
            if running >= max_concurrent:
                raise RunCapacityError(f"Too many active runs (limit {max_concurrent})")
            job_id = self._new_id(results_dir, jobs)
            job = {
                "job_id": job_id,
                "strategy_name": strategy_name,
                "status": "starting",
                "started_at": self._now(),
                "process": None,
            }
            jobs[job_id] = job
            try:
                self._write_state(results_dir, job)
            except Exception:
                jobs.pop(job_id, None)
                raise
            return job

    def release(
        self,
        jobs: MutableMapping[str, Any],
        lock: threading.Lock,
        results_dir: Path,
        job_id: str,
    ) -> None:
        """Release a reservation that failed request validation before spawn."""
        with lock:
            jobs.pop(job_id, None)
        state_dir = Path(results_dir) / job_id
        if state_dir.exists():
            for path in state_dir.iterdir():
                path.unlink()
            state_dir.rmdir()

    def attach(
        self,
        jobs: MutableMapping[str, Any],
        lock: threading.Lock,
        results_dir: Path,
        job_id: str,
        process: Any,
        stdout_path: Path,
        stderr_path: Path,
        cmd: list[str],
    ) -> None:
        with lock:
            job = jobs[job_id]
            job.update(
                process=process,
                pid=process.pid,
                status="running",
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                cmd=cmd,
            )
            self._write_state(results_dir, job)

    def fail_spawn(
        self,
        jobs: MutableMapping[str, Any],
        lock: threading.Lock,
        results_dir: Path,
        job_id: str,
        message: str,
    ) -> None:
        with lock:
            self._finish_locked(jobs, Path(results_dir), job_id, "failed", message)

    def reconcile(
        self,
        jobs: MutableMapping[str, Any],
        lock: threading.Lock,
        results_dir: Path,
    ) -> None:
        with lock:
            self._reap_locked(jobs, Path(results_dir))

    def cancel(
        self,
        jobs: MutableMapping[str, Any],
        lock: threading.Lock,
        results_dir: Path,
        job_id: str,
    ) -> dict[str, Any] | None:
        with lock:
            self._reap_locked(jobs, Path(results_dir))
            job = jobs.get(job_id)
            if job is None:
                return None
            if job.get("status") in TERMINAL_STATUSES:
                return dict(job)
            process = job.get("process")
            self._finish_locked(jobs, Path(results_dir), job_id, "cancelled", None)
        if process is not None and process.poll() is None:
            try:
                os.kill(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return dict(job)

    def snapshot(
        self,
        jobs: MutableMapping[str, Any],
        lock: threading.Lock,
        results_dir: Path,
    ) -> dict[str, dict[str, Any]]:
        self.reconcile(jobs, lock, results_dir)
        with lock:
            return {job_id: dict(job) for job_id, job in jobs.items()}

    def disk_state(self, results_dir: Path, job_id: str) -> dict[str, Any] | None:
        path = self._state_path(Path(results_dir), job_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if data.get("status") in ACTIVE_STATUSES:
            data["status"] = "failed"
            data["error_message"] = "Run process state unavailable after API restart"
            data["finished_at"] = self._now()
            self._write_state(Path(results_dir), data)
        return data
