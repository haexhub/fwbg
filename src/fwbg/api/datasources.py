"""REST API for data source management."""
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from fwbg.core.data_sources import (
    _DATA_SOURCES,
    CSVSourceConfig,
    source_from_dict,
    save_source_config,
    delete_data_source,
    get_data_root,
)

router = APIRouter()

# In-memory store for async prepare tasks
_prepare_tasks: Dict[str, dict] = {}

# Max bytes accepted per uploaded file (100 MB)
MAX_UPLOAD_SIZE = 100 * 1024 * 1024

# Datasource names must be safe filesystem identifiers.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


def _safe_filename(filename: str) -> str:
    """Return the basename of *filename* and reject path traversal attempts."""
    if not filename:
        raise HTTPException(status_code=400, detail="Empty filename")
    base = os.path.basename(filename)
    if base in ("", ".", "..") or base != filename:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {filename!r}")
    if "/" in base or "\\" in base or "\x00" in base:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {filename!r}")
    return base


def _validate_source_name(name: str) -> None:
    if not _SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail=f"Invalid source name: {name!r}")


def _safe_child_path(parent: Path, child_name: str) -> Path:
    """Resolve *child_name* under *parent* and ensure it stays inside."""
    parent_resolved = parent.resolve()
    candidate = (parent / child_name).resolve()
    try:
        candidate.relative_to(parent_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal detected")
    return candidate


def _raw_dir(name: str) -> Path:
    return get_data_root() / name / "raw"


def _datasource_dir(name: str) -> Path:
    """Datasource directory — uses the path from config for CSV, or data/{name}/datasource for others."""
    source = _DATA_SOURCES.get(name)
    if isinstance(source, CSVSourceConfig):
        return source.path
    return get_data_root() / name / "datasource"


def _file_info(path: Path) -> dict:
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "modified": stat.st_mtime}


def _source_to_response(name: str) -> dict:
    source = _DATA_SOURCES[name]
    d = source.to_dict()

    if isinstance(source, CSVSourceConfig):
        ds_path = source.path
        raw_path = _raw_dir(name)

        files = []
        if ds_path.exists():
            files = [_file_info(f) for f in sorted(ds_path.glob("*.csv"))]

        raw_files = []
        if raw_path.exists():
            raw_files = [_file_info(f) for f in sorted(raw_path.iterdir()) if f.is_file()]

        d["files"] = files
        d["file_count"] = len(files)
        d["raw_files"] = raw_files
        d["raw_file_count"] = len(raw_files)

    return d


# ── CRUD ──────────────────────────────────────────────────────────────────────

@router.get("/datasources")
def list_sources():
    return [_source_to_response(name) for name in _DATA_SOURCES]


@router.post("/datasources", status_code=201)
def create_source(body: dict):
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    _validate_source_name(name)
    if name in _DATA_SOURCES:
        raise HTTPException(status_code=409, detail=f"Source already exists: {name}")

    source_type = body.get("type")

    if source_type == "csv":
        # Auto-set path to data/{name}/datasource
        root = get_data_root()
        datasource_dir = root / name / "datasource"
        raw_dir = root / name / "raw"
        datasource_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        body["path"] = str(datasource_dir)
    else:
        # Non-CSV: just ensure the directory exists
        (get_data_root() / name).mkdir(parents=True, exist_ok=True)

    try:
        source = source_from_dict(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    _DATA_SOURCES[name] = source
    save_source_config(source)
    return _source_to_response(name)


@router.delete("/datasources/{name}", status_code=204)
def remove_source(name: str):
    _validate_source_name(name)
    if name not in _DATA_SOURCES:
        raise HTTPException(status_code=404, detail=f"Source not found: {name}")

    # Remove config.json (data files stay on disk)
    config_path = get_data_root() / name / "config.json"
    if config_path.exists():
        config_path.unlink()

    delete_data_source(name)


@router.put("/datasources/{name}")
def update_source(name: str, body: dict):
    """Merge-update an existing source config."""
    _validate_source_name(name)
    if name not in _DATA_SOURCES:
        raise HTTPException(status_code=404, detail=f"Source not found: {name}")

    current = _DATA_SOURCES[name].to_dict()
    # Merge: body overwrites current, but keep name/type immutable
    body.pop("name", None)
    body.pop("type", None)
    current.update(body)
    current["name"] = name

    try:
        source = source_from_dict(current)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    _DATA_SOURCES[name] = source
    save_source_config(source)
    return _source_to_response(name)


# ── Raw files ─────────────────────────────────────────────────────────────────

@router.get("/datasources/{name}/raw")
def list_raw_files(name: str):
    if name not in _DATA_SOURCES:
        raise HTTPException(status_code=404, detail=f"Source not found: {name}")
    raw_path = _raw_dir(name)
    if not raw_path.exists():
        return []
    return [_file_info(f) for f in sorted(raw_path.iterdir()) if f.is_file()]


@router.post("/datasources/{name}/raw", status_code=201)
async def upload_raw_files(name: str, files: List[UploadFile] = File(...)):
    _validate_source_name(name)
    if name not in _DATA_SOURCES:
        raise HTTPException(status_code=404, detail=f"Source not found: {name}")
    raw_path = _raw_dir(name)
    raw_path.mkdir(parents=True, exist_ok=True)
    saved = []
    for file in files:
        safe_name = _safe_filename(file.filename or "")
        data = await file.read()
        if len(data) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large: {safe_name}")
        dest = _safe_child_path(raw_path, safe_name)
        dest.write_bytes(data)
        saved.append(safe_name)
    return {"saved": saved}


@router.delete("/datasources/{name}/raw/{filename}", status_code=204)
def delete_raw_file(name: str, filename: str):
    _validate_source_name(name)
    if name not in _DATA_SOURCES:
        raise HTTPException(status_code=404, detail=f"Source not found: {name}")
    safe_name = _safe_filename(filename)
    f = _safe_child_path(_raw_dir(name), safe_name)
    if not f.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {safe_name}")
    f.unlink()


# ── Datasource files ──────────────────────────────────────────────────────────

@router.get("/datasources/{name}/datasource")
def list_datasource_files(name: str):
    if name not in _DATA_SOURCES:
        raise HTTPException(status_code=404, detail=f"Source not found: {name}")
    ds_path = _datasource_dir(name)
    if not ds_path.exists():
        return []
    return [_file_info(f) for f in sorted(ds_path.glob("*.csv"))]


@router.delete("/datasources/{name}/datasource/{filename}", status_code=204)
def delete_datasource_file(name: str, filename: str):
    _validate_source_name(name)
    if name not in _DATA_SOURCES:
        raise HTTPException(status_code=404, detail=f"Source not found: {name}")
    safe_name = _safe_filename(filename)
    f = _safe_child_path(_datasource_dir(name), safe_name)
    if not f.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {safe_name}")
    f.unlink()


# ── ETL ───────────────────────────────────────────────────────────────────────

@router.get("/datasources/{name}/raw/{filename}/preview")
def preview_raw_file(name: str, filename: str, rows: int = 5):
    if name not in _DATA_SOURCES:
        raise HTTPException(status_code=404, detail=f"Source not found: {name}")
    f = _raw_dir(name) / filename
    if not f.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    try:
        import pandas as pd
        df = pd.read_csv(f, nrows=rows + 1)
        return {
            "columns": list(df.columns),
            "rows": df.head(rows).fillna("").values.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")


class ProcessRequest(BaseModel):
    filename: str
    symbol: str
    timeframe: str
    date_col: str
    open_col: str
    high_col: str
    low_col: str
    close_col: str
    volume_col: Optional[str] = None
    timestamp_format: str = ""  # "", "unix_s", "unix_ms"
    timezone: str = ""          # IANA timezone


@router.post("/datasources/{name}/process")
def process_raw_file(name: str, req: ProcessRequest):
    """ETL: read raw file with column mapping, write to datasource/SYMBOL_TIMEFRAME.csv."""
    if name not in _DATA_SOURCES:
        raise HTTPException(status_code=404, detail=f"Source not found: {name}")

    raw_file = _raw_dir(name) / req.filename
    if not raw_file.exists():
        raise HTTPException(status_code=404, detail=f"Raw file not found: {req.filename}")

    try:
        import pandas as pd
        df = pd.read_csv(raw_file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    col_map = {
        req.date_col: "DATE",
        req.open_col: "OPEN",
        req.high_col: "HIGH",
        req.low_col: "LOW",
        req.close_col: "CLOSE",
    }
    if req.volume_col:
        col_map[req.volume_col] = "VOLUME"

    for col in col_map:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column not found in file: {col!r}")

    df = df[list(col_map.keys())].rename(columns=col_map)

    # Timestamp conversion
    if req.timestamp_format in ("unix_s", "unix_ms"):
        unit = "s" if req.timestamp_format == "unix_s" else "ms"
        ts = pd.to_datetime(df["DATE"], unit=unit, utc=True)
        if req.timezone:
            ts = ts.dt.tz_convert(req.timezone)
        df["DATE"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        df["DATE"] = pd.to_datetime(df["DATE"])

    df = df.set_index("DATE").sort_index()

    ds_path = _datasource_dir(name)
    ds_path.mkdir(parents=True, exist_ok=True)

    out_filename = f"{req.symbol}_{req.timeframe}.csv"
    df.to_csv(ds_path / out_filename)

    return {"output": out_filename, "rows": len(df)}


# ── Batch prepare (async) ────────────────────────────────────────────────────

class PrepareRequest(BaseModel):
    glob_pattern: str = ""       # Optional override for raw_pattern
    excludes: List[str] = []     # Filenames to exclude from processing


@router.post("/datasources/{name}/prepare", status_code=202)
def start_prepare(name: str, req: PrepareRequest):
    """Launch batch prepare as a background task."""
    if name not in _DATA_SOURCES:
        raise HTTPException(status_code=404, detail=f"Source not found: {name}")
    source = _DATA_SOURCES[name]
    if not isinstance(source, CSVSourceConfig):
        raise HTTPException(status_code=400, detail="prepare is only supported for CSV sources")

    task_id = uuid.uuid4().hex[:12]
    _prepare_tasks[task_id] = {"status": "running", "result": None, "error": None}

    def run():
        try:
            converted = source.prepare(glob_override=req.glob_pattern, excludes=req.excludes)
            _prepare_tasks[task_id]["result"] = converted
            _prepare_tasks[task_id]["status"] = "done"
        except Exception as e:
            _prepare_tasks[task_id]["error"] = str(e)
            _prepare_tasks[task_id]["status"] = "error"

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    return {"task_id": task_id}


@router.get("/datasources/{name}/prepare/{task_id}")
def prepare_status(name: str, task_id: str):
    """Poll status of a prepare task."""
    if task_id not in _prepare_tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    return _prepare_tasks[task_id]
