"""Shared filesystem + JSON helpers for API endpoints.

Centralises common patterns to keep route handlers slim:
- Safe JSON loading with a configurable size cap (DoS protection)
- Validated path construction under the results directory
- Run-id / symbol identifier validation
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from fwbg.api.deps import get_test_results_dir


# Path-safe identifier regex — used for run_id, symbol, strategy_name.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")

# Per-file cap for JSON inputs read from disk. Configs and results in this app
# are KB-scale; multi-MB files are almost certainly malformed or hostile.
DEFAULT_JSON_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


def validate_id(value: str, field: str) -> str:
    if not _SAFE_ID_RE.match(value or ""):
        raise HTTPException(400, f"Invalid {field}: {value!r}")
    return value


def safe_results_path(*parts: str) -> Path:
    """Build a path under the configured results directory.

    Each component is validated as a safe identifier and the resolved path is
    checked to ensure it cannot escape the results root via ``..`` or symlinks.
    """
    base = get_test_results_dir().resolve()
    for part in parts:
        validate_id(part, "path component")
    candidate = base.joinpath(*parts).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise HTTPException(400, "Path traversal detected")
    return candidate


def safe_load_json(
    path: Path,
    default: Any = None,
    max_bytes: int = DEFAULT_JSON_MAX_BYTES,
) -> Any:
    """Load JSON from *path* with a hard size cap.

    Returns *default* on any failure (missing file, invalid JSON, IO error,
    oversize). Raises HTTPException(413) only when the file is present but too
    large — callers that want silent fallback there should pass a smaller
    *max_bytes* and catch the exception.
    """
    try:
        if not path.exists():
            return default
        if path.stat().st_size > max_bytes:
            raise HTTPException(413, f"File too large: {path.name}")
        with open(path, "rb") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"File too large: {path.name}")
        return json.loads(data.decode("utf-8"))
    except HTTPException:
        raise
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default
