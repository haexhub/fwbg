"""Structured JSON logging for optimization runs.

Writes machine-readable log entries to a JSONL file per run.
Accessible from CLI (cat, jq) and web dashboard (API endpoint).

Each line is a self-contained JSON object:
{"timestamp": "...", "level": "info", "symbol": "EURUSD",
 "stage": "model_training", "message": "Trained on 50000 samples",
 "data": {"n_samples": 50000, "duration_seconds": 3.2}}
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class RunLogger:
    """Thread-safe structured logger writing to JSONL file."""

    def __init__(self, run_directory: Path):
        self._file_path = run_directory / "logs.jsonl"
        self._lock = threading.Lock()
        run_directory.mkdir(parents=True, exist_ok=True)
        self._file = open(self._file_path, "a", encoding="utf-8")

    def info(self, symbol: str, stage: str, message: str, **data: Any) -> None:
        self._write_entry("info", symbol, stage, message, data)

    def warning(self, symbol: str, stage: str, message: str, **data: Any) -> None:
        self._write_entry("warning", symbol, stage, message, data)

    def error(self, symbol: str, stage: str, message: str, **data: Any) -> None:
        self._write_entry("error", symbol, stage, message, data)

    def debug(self, symbol: str, stage: str, message: str, **data: Any) -> None:
        self._write_entry("debug", symbol, stage, message, data)

    def _write_entry(self, level: str, symbol: str, stage: str,
                     message: str, data: Dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level,
            "symbol": symbol,
            "stage": stage,
            "message": message,
        }
        if data:
            entry["data"] = data
        line = json.dumps(entry, default=str) + "\n"
        with self._lock:
            try:
                self._file.write(line)
                self._file.flush()
            except Exception:
                pass  # Best-effort

    def write_raw(self, entry: dict) -> None:
        """Write a pre-built log entry dict (from queue messages)."""
        line = json.dumps(entry, default=str) + "\n"
        with self._lock:
            try:
                self._file.write(line)
                self._file.flush()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            try:
                self._file.close()
            except Exception:
                pass
