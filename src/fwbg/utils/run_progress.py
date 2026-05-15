"""Persistent run progress tracking with JSON file output.

Progress is written to disk on every state change, making it:
- Survivable across crashes (last state preserved)
- Accessible from web dashboard (API reads the file)
- Accessible from CLI (cat/jq the file)
"""
import json
import time
import threading
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List
from enum import Enum


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AssetStageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageProgress:
    """Progress of a single processing stage for one asset."""
    stage_name: str
    status: AssetStageStatus = AssetStageStatus.PENDING
    description: str = ""
    progress_fraction: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    details: Dict = field(default_factory=dict)


@dataclass
class AssetProgress:
    """Progress tracking for a single asset."""
    symbol: str
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: float = 0.0
    result_summary: str = ""
    stages: List[StageProgress] = field(default_factory=list)


@dataclass
class RunProgress:
    """Complete progress state for an optimization run."""
    run_id: str
    status: str = "initializing"
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    overall_progress_fraction: float = 0.0
    total_assets: int = 0
    completed_assets: int = 0
    failed_assets: int = 0
    active_assets: int = 0
    pending_assets: int = 0
    assets: Dict[str, AssetProgress] = field(default_factory=dict)
    strategy_name: str = ""
    error_message: Optional[str] = None


# Stage weights for ETA estimation (fraction of total asset time)
_STAGE_WEIGHTS = {
    "data_loading": 0.02,
    "indicators": 0.05,
    "grid_search": 0.85,
    "model_training": 0.03,
    "evaluation": 0.05,
}


class RunProgressWriter:
    """Thread-safe writer that persists RunProgress to a JSON file.

    Debounces disk writes to avoid excessive I/O.
    """
    WRITE_INTERVAL_SECONDS = 1.0

    def __init__(self, run_directory: Path, run_id: str, asset_symbols: List[str],
                 strategy_name: str = ""):
        self._run_dir = run_directory
        self._file_path = run_directory / "progress.json"
        self._lock = threading.Lock()
        self._dirty = False
        self._last_write_time = 0.0
        self._progress = RunProgress(
            run_id=run_id,
            total_assets=len(asset_symbols),
            pending_assets=len(asset_symbols),
            started_at=_iso_now(),
            strategy_name=strategy_name,
        )
        for symbol in asset_symbols:
            self._progress.assets[symbol] = AssetProgress(symbol=symbol)
        self._flush_to_disk()

    def begin_asset(self, symbol: str) -> None:
        with self._lock:
            asset = self._progress.assets.get(symbol)
            if asset:
                asset.status = "running"
                asset.started_at = _iso_now()
            self._progress.status = "running"
            self._recalculate_totals()
            self._maybe_flush()

    # Canonical stage ordering for complete_earlier_stages
    _STAGE_ORDER = ["data_loading", "indicators", "evaluation", "grid_search", "model_training"]

    def complete_earlier_stages(self, symbol: str, new_stage_name: str) -> None:
        """Mark all stages that come before new_stage_name as completed."""
        with self._lock:
            asset = self._progress.assets.get(symbol)
            if not asset:
                return
            try:
                new_idx = self._STAGE_ORDER.index(new_stage_name)
            except ValueError:
                return  # Unknown stage, skip
            for stage in asset.stages:
                try:
                    stage_idx = self._STAGE_ORDER.index(stage.stage_name)
                except ValueError:
                    continue
                if stage_idx < new_idx and stage.status == AssetStageStatus.RUNNING:
                    stage.status = AssetStageStatus.COMPLETED
                    stage.progress_fraction = 1.0
                    if not stage.completed_at:
                        stage.completed_at = _iso_now()

    def update_asset_stage(self, symbol: str, stage_name: str,
                           status: str = "running", description: str = "",
                           progress_fraction: Optional[float] = None,
                           details: Optional[Dict] = None) -> None:
        with self._lock:
            asset = self._progress.assets.get(symbol)
            if not asset:
                return

            # Find or create stage
            stage = None
            for s in asset.stages:
                if s.stage_name == stage_name:
                    stage = s
                    break
            if stage is None:
                stage = StageProgress(stage_name=stage_name)
                asset.stages.append(stage)

            stage.status = AssetStageStatus(status)
            stage.description = description
            if progress_fraction is not None:
                stage.progress_fraction = progress_fraction
            if details:
                stage.details.update(details)

            if status == "running" and not stage.started_at:
                stage.started_at = _iso_now()
            elif status == "completed":
                stage.completed_at = _iso_now()
                stage.progress_fraction = 1.0
                if stage.started_at:
                    start = datetime.fromisoformat(stage.started_at)
                    end = datetime.fromisoformat(stage.completed_at)
                    stage.duration_seconds = (end - start).total_seconds()

            self._recalculate_totals()
            self._maybe_flush()

    def complete_asset(self, symbol: str, result_summary: str = "") -> None:
        with self._lock:
            asset = self._progress.assets.get(symbol)
            if asset:
                asset.status = "completed"
                asset.completed_at = _iso_now()
                asset.result_summary = result_summary
                if asset.started_at:
                    start = datetime.fromisoformat(asset.started_at)
                    end = datetime.fromisoformat(asset.completed_at)
                    asset.duration_seconds = (end - start).total_seconds()
                self._write_asset_progress(asset)
            self._recalculate_totals()
            self._estimate_remaining_time()
            self._flush_to_disk()

    def fail_asset(self, symbol: str, error: str = "") -> None:
        with self._lock:
            asset = self._progress.assets.get(symbol)
            if asset:
                asset.status = "failed"
                asset.completed_at = _iso_now()
                asset.result_summary = f"FAILED: {error}"
                self._write_asset_progress(asset)
            self._recalculate_totals()
            self._flush_to_disk()

    def complete_run(self) -> None:
        with self._lock:
            self._progress.status = "completed"
            self._progress.updated_at = _iso_now()
            self._progress.estimated_remaining_seconds = 0
            self._progress.overall_progress_fraction = 1.0
            self._recalculate_totals()
            self._flush_to_disk()

    def fail_run(self, error: str) -> None:
        with self._lock:
            self._progress.status = "failed"
            self._progress.error_message = error
            self._progress.updated_at = _iso_now()
            self._flush_to_disk()

    def _recalculate_totals(self) -> None:
        completed = 0
        failed = 0
        active = 0
        pending = 0
        for asset in self._progress.assets.values():
            if asset.status == "completed":
                completed += 1
            elif asset.status == "failed":
                failed += 1
            elif asset.status == "running":
                active += 1
            else:
                pending += 1

        self._progress.completed_assets = completed
        self._progress.failed_assets = failed
        self._progress.active_assets = active
        self._progress.pending_assets = pending

        total = self._progress.total_assets
        if total > 0:
            # Include partial progress from active assets
            partial = 0.0
            for asset in self._progress.assets.values():
                if asset.status == "running":
                    partial += self._estimate_asset_progress(asset)
            self._progress.overall_progress_fraction = (completed + failed + partial) / total

        if self._progress.started_at:
            start = datetime.fromisoformat(self._progress.started_at)
            now = datetime.now(timezone.utc)
            self._progress.elapsed_seconds = (now - start).total_seconds()

        self._progress.updated_at = _iso_now()

    def _estimate_asset_progress(self, asset: AssetProgress) -> float:
        """Estimate fraction complete for an active asset based on stage progress."""
        if not asset.stages:
            return 0.0

        progress = 0.0
        for stage in asset.stages:
            weight = _STAGE_WEIGHTS.get(stage.stage_name, 0.05)
            if stage.status == AssetStageStatus.COMPLETED:
                progress += weight
            elif stage.status == AssetStageStatus.RUNNING:
                progress += weight * stage.progress_fraction
        return min(progress, 1.0)

    def _estimate_remaining_time(self) -> None:
        completed_assets = [
            a for a in self._progress.assets.values()
            if a.status == "completed" and a.duration_seconds > 0
        ]
        if not completed_assets:
            return

        avg_duration = sum(a.duration_seconds for a in completed_assets) / len(completed_assets)
        remaining = self._progress.pending_assets + self._progress.active_assets
        # Assume max_concurrent_assets is at least 1
        active = max(1, self._progress.active_assets)
        self._progress.estimated_remaining_seconds = (avg_duration * remaining) / active

    def _write_asset_progress(self, asset: AssetProgress) -> None:
        """Write per-asset progress to grid_details/{symbol}/progress.json."""
        sym_dir = self._run_dir / "grid_details" / asset.symbol
        if not sym_dir.exists():
            return  # grid_details dir not created yet — on_result_ready handles it
        stages = []
        for s in asset.stages:
            stages.append({
                "stage_name": s.stage_name,
                "status": s.status.value if isinstance(s.status, Enum) else s.status,
                "description": s.description,
                "progress_fraction": round(s.progress_fraction, 3),
                "started_at": s.started_at,
                "completed_at": s.completed_at,
                "duration_seconds": round(s.duration_seconds, 2),
                "details": s.details,
            })
        data = {
            "symbol": asset.symbol,
            "status": asset.status,
            "started_at": asset.started_at,
            "completed_at": asset.completed_at,
            "duration_seconds": round(asset.duration_seconds, 2),
            "result_summary": asset.result_summary,
            "stages": stages,
        }
        try:
            progress_file = sym_dir / "progress.json"
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass  # Best-effort

    def _maybe_flush(self) -> None:
        now = time.monotonic()
        if now - self._last_write_time >= self.WRITE_INTERVAL_SECONDS:
            self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        """Atomic write via tmp + rename."""
        self._last_write_time = time.monotonic()
        data = self._serialize()
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self._file_path.parent, suffix=".tmp"
            )
            try:
                with open(tmp_fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                Path(tmp_path).replace(self._file_path)
            except Exception:
                Path(tmp_path).unlink(missing_ok=True)
                raise
        except Exception:
            pass  # Best-effort — don't crash the run for progress I/O

    def _serialize(self) -> dict:
        """Convert RunProgress to a JSON-safe dict.

        Per-asset detail (stages) lives in grid_details/{symbol}/progress.json.
        The global file only keeps lightweight status per asset.
        """
        p = self._progress
        assets = {}
        for sym, asset in p.assets.items():
            asset_data: dict = {
                "symbol": asset.symbol,
                "status": asset.status,
                "started_at": asset.started_at,
                "completed_at": asset.completed_at,
                "duration_seconds": round(asset.duration_seconds, 2),
                "result_summary": asset.result_summary,
            }
            # Include stages only for running/pending assets (live monitoring).
            # Completed/failed assets have their stages in grid_details/{sym}/progress.json.
            if asset.status in ("running", "pending"):
                stages = []
                for s in asset.stages:
                    stages.append({
                        "stage_name": s.stage_name,
                        "status": s.status.value if isinstance(s.status, Enum) else s.status,
                        "description": s.description,
                        "progress_fraction": round(s.progress_fraction, 3),
                        "started_at": s.started_at,
                        "completed_at": s.completed_at,
                        "duration_seconds": round(s.duration_seconds, 2),
                        "details": s.details,
                    })
                asset_data["stages"] = stages

            assets[sym] = asset_data

        return {
            "run_id": p.run_id,
            "status": p.status,
            "started_at": p.started_at,
            "updated_at": p.updated_at,
            "elapsed_seconds": round(p.elapsed_seconds, 1),
            "estimated_remaining_seconds": (
                round(p.estimated_remaining_seconds, 1)
                if p.estimated_remaining_seconds is not None else None
            ),
            "overall_progress_fraction": round(p.overall_progress_fraction, 4),
            "total_assets": p.total_assets,
            "completed_assets": p.completed_assets,
            "failed_assets": p.failed_assets,
            "active_assets": p.active_assets,
            "pending_assets": p.pending_assets,
            "strategy_name": p.strategy_name,
            "error_message": p.error_message,
            "assets": assets,
        }
