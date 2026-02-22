"""Preset CRUD endpoints for reusable strategy sub-configurations.

Each preset file embeds metadata:
    { "_meta": { "name": "Display Name", "description": "...", "version": 1 }, ...content }

File naming convention: {slug}_v{version}.json
"""
import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from fwbg.api.deps import get_strategies_dir
from fwbg.api.strategies import SECTION_FIELD_DIRS

router = APIRouter(prefix="/presets", tags=["presets"])

SECTION_DIRS = {
    "pipelines": "pipelines",
    "exit_params": "exit_params",
    "models": "models",
    "validations": "validations",
    "filters": "filters",
    "resources": "resources",
    "grids": "grids",
    "regime_filters": "regime_filters",
    "risk_params": "risk_params",
}


# ── Models ────────────────────────────────────────────────────────────────────

class PresetMeta(BaseModel):
    name: str
    description: str = ""
    version: int = 1


class PresetCreateBody(BaseModel):
    name: str
    description: str = ""
    content: dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """Convert display name to a safe filename slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug.strip("_")


def _get_preset_dir(section: str) -> Path:
    """Resolve the filesystem path for a preset section directory."""
    if section not in SECTION_DIRS:
        raise HTTPException(404, f"Unknown preset section: {section}")
    strategies_dir = get_strategies_dir()
    path = strategies_dir.parent / SECTION_DIRS[section]
    path.mkdir(parents=True, exist_ok=True)
    return path


_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_name(name: str) -> None:
    """Reject preset names that could enable path traversal or injection.

    Only alphanumeric characters, underscores, and hyphens are allowed.
    This is checked before any filesystem access.
    """
    if not name or not _SAFE_NAME_RE.match(name):
        raise HTTPException(400, "Invalid preset name: only letters, digits, _ and - are allowed")


def _guard_path(filepath: Path, preset_dir: Path) -> None:
    """Raise 400 if filepath escapes preset_dir (path traversal guard)."""
    if not str(filepath.resolve()).startswith(str(preset_dir.resolve())):
        raise HTTPException(400, "Invalid preset name")


def _resolve_preset_path(name: str, preset_dir: Path) -> Path:
    """Return the Path for a preset ID, falling back to versioned filenames.

    Tries ``{name}.json`` first; if absent, picks the highest-versioned
    ``{name}_v*.json`` (supports old refs before migration).
    Raises HTTPException 400 for unsafe names, 404 when nothing is found.
    """
    _validate_name(name)
    filepath = preset_dir / f"{name}.json"
    _guard_path(filepath, preset_dir)
    if filepath.exists():
        return filepath

    matches = sorted(preset_dir.glob(f"{name}_v*.json"))
    if not matches:
        raise HTTPException(404, f"Preset not found: {name}")
    resolved = matches[-1]  # highest lexicographic = highest version
    _guard_path(resolved, preset_dir)
    return resolved


def _read_preset_file(filepath: Path) -> dict:
    """Parse a preset file → { id, meta, content }."""
    file_id = filepath.stem
    raw = json.loads(filepath.read_text())
    meta_raw = raw.pop("_meta", None)
    if meta_raw:
        meta = PresetMeta(
            name=meta_raw.get("name", file_id),
            description=meta_raw.get("description", ""),
            version=meta_raw.get("version", 1),
        )
    else:
        meta = PresetMeta(name=file_id, description="", version=1)
    return {"id": file_id, "meta": meta.model_dump(), "content": raw}


def _write_preset_file(filepath: Path, meta: PresetMeta, content: dict) -> None:
    """Write { _meta, ...content } to a preset file."""
    data = {"_meta": meta.model_dump(), **content}
    filepath.write_text(json.dumps(data, indent=2))


def _find_versions_by_name(preset_dir: Path, display_name: str) -> list[Path]:
    """Find all preset files whose _meta.name matches display_name."""
    result = []
    for f in preset_dir.glob("*.json"):
        try:
            raw = json.loads(f.read_text())
            if raw.get("_meta", {}).get("name") == display_name:
                result.append(f)
        except (json.JSONDecodeError, IOError):
            continue
    return result


def _next_version(preset_dir: Path, display_name: str) -> int:
    """Compute next available version number for a given display name."""
    files = _find_versions_by_name(preset_dir, display_name)
    versions = []
    for f in files:
        try:
            raw = json.loads(f.read_text())
            versions.append(raw.get("_meta", {}).get("version", 1))
        except (json.JSONDecodeError, IOError):
            continue
    return max(versions, default=0) + 1


# ── Startup Migration ─────────────────────────────────────────────────────────

def migrate_presets() -> None:
    """Migrate legacy preset files (without _meta) to the new versioned format.

    Called once at app startup. Each file gets a _meta block added and is
    renamed to {slug}_v1.json if not already versioned.
    """
    try:
        strategies_dir = get_strategies_dir()
    except Exception:
        return

    base = strategies_dir.parent
    for section_dir in SECTION_DIRS.values():
        path = base / section_dir
        if not path.exists():
            continue
        for f in list(path.glob("*.json")):
            try:
                raw = json.loads(f.read_text())
            except (json.JSONDecodeError, IOError):
                continue

            if "_meta" in raw:
                continue  # Already migrated

            stem = f.stem
            # All legacy files (no _meta) start at v1 regardless of their filename
            display_name = re.sub(r"_v\d+$", "", stem)  # strip any existing _vN suffix
            new_name = f"{_slugify(display_name)}_v1.json"

            meta = PresetMeta(name=display_name, description="", version=1)
            new_path = path / new_name

            if new_path.exists() and new_path != f:
                continue  # Don't overwrite existing versioned file

            _write_preset_file(new_path, meta, raw)
            if new_path != f:
                f.unlink()


def _versioned_id(name: str, preset_dir: Path) -> str | None:
    """Return the versioned file stem for an old-style preset name.

    Returns None if the exact ``{name}.json`` already exists (already correct)
    or if no versioned file is found.
    """
    if (preset_dir / f"{name}.json").exists():
        return None
    matches = sorted(preset_dir.glob(f"{name}_v*.json"))
    return matches[-1].stem if matches else None


def migrate_strategy_refs() -> None:
    """Update strategy JSON files to use versioned preset IDs.

    Replaces bare preset names (e.g. ``"orb_simple"``) with the versioned ID
    (e.g. ``"orb_simple_v1"``) in all strategy files.  Called at app startup
    after migrate_presets().
    """
    try:
        strategies_dir = get_strategies_dir()
    except Exception:
        return

    base = strategies_dir.parent

    for strategy_file in sorted(strategies_dir.glob("*.json")):
        try:
            data = json.loads(strategy_file.read_text())
        except (json.JSONDecodeError, IOError):
            continue

        modified = False

        # Simple section refs (pipeline, model, validation, …)
        for field, section_dir in SECTION_FIELD_DIRS.items():
            value = data.get(field)
            if isinstance(value, str):
                new_id = _versioned_id(value, base / section_dir)
                if new_id:
                    data[field] = new_id
                    modified = True

        # Grid assignments
        grids_data = data.get("grids", {})
        if isinstance(grids_data, dict) and "assignments" in grids_data:
            grids_dir = base / "grids"
            regime_filters_dir = base / "regime_filters"

            rfg = grids_data.get("regime_filter_grid")
            if isinstance(rfg, str):
                new_id = _versioned_id(rfg, regime_filters_dir)
                if new_id:
                    grids_data["regime_filter_grid"] = new_id
                    modified = True

            for asset_class, assignment in grids_data["assignments"].items():
                if isinstance(assignment, str):
                    new_id = _versioned_id(assignment, grids_dir)
                    if new_id:
                        grids_data["assignments"][asset_class] = new_id
                        modified = True
                elif isinstance(assignment, dict) and "preset" in assignment:
                    new_id = _versioned_id(assignment["preset"], grids_dir)
                    if new_id:
                        assignment["preset"] = new_id
                        modified = True

        if modified:
            strategy_file.write_text(json.dumps(data, indent=2))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/{section}")
def list_presets(section: str) -> list[dict]:
    """List all preset files for a section, sorted by name then version."""
    preset_dir = _get_preset_dir(section)
    presets = []
    for f in sorted(preset_dir.glob("*.json"), key=lambda p: p.name):
        try:
            presets.append(_read_preset_file(f))
        except (json.JSONDecodeError, IOError, ValueError):
            continue
    return presets


@router.post("/{section}")
def create_preset(section: str, body: PresetCreateBody) -> dict:
    """Create a new preset (always v1). Returns { id, meta, content }."""
    preset_dir = _get_preset_dir(section)
    slug = _slugify(body.name)
    if not slug:
        raise HTTPException(400, "Invalid preset name")
    _validate_name(slug)  # ensure slug only contains safe characters

    file_id = f"{slug}_v1"
    filepath = preset_dir / f"{file_id}.json"
    _guard_path(filepath, preset_dir)

    if filepath.exists():
        raise HTTPException(409, f"Konfiguration existiert bereits: {file_id}")

    meta = PresetMeta(name=body.name, description=body.description, version=1)
    _write_preset_file(filepath, meta, body.content)
    return _read_preset_file(filepath)


@router.get("/{section}/{name}")
def get_preset(section: str, name: str) -> dict:
    """Load a single preset by file ID."""
    preset_dir = _get_preset_dir(section)
    filepath = _resolve_preset_path(name, preset_dir)
    return _read_preset_file(filepath)


@router.put("/{section}/{name}")
def update_preset(section: str, name: str, body: dict) -> dict:
    """Overwrite preset content; preserve _meta (version unchanged)."""
    preset_dir = _get_preset_dir(section)
    filepath = _resolve_preset_path(name, preset_dir)

    raw = json.loads(filepath.read_text())
    meta_raw = raw.get("_meta", {})
    meta = PresetMeta(
        name=meta_raw.get("name", name),
        description=meta_raw.get("description", ""),
        version=meta_raw.get("version", 1),
    )
    _write_preset_file(filepath, meta, body)
    return _read_preset_file(filepath)


@router.post("/{section}/{name}/version")
def create_version(section: str, name: str, body: dict) -> dict:
    """Create a new version of an existing preset with the given content."""
    preset_dir = _get_preset_dir(section)
    filepath = _resolve_preset_path(name, preset_dir)

    raw = json.loads(filepath.read_text())
    meta_raw = raw.get("_meta", {})
    display_name = meta_raw.get("name", name)
    description = meta_raw.get("description", "")
    slug = _slugify(display_name)

    next_v = _next_version(preset_dir, display_name)
    new_id = f"{slug}_v{next_v}"
    new_filepath = preset_dir / f"{new_id}.json"

    if new_filepath.exists():
        raise HTTPException(409, f"Version already exists: {new_id}")

    meta = PresetMeta(name=display_name, description=description, version=next_v)
    _write_preset_file(new_filepath, meta, body)
    return _read_preset_file(new_filepath)


class PresetMetaUpdateBody(BaseModel):
    name: str
    description: str = ""


@router.patch("/{section}/{name}")
def update_preset_meta(section: str, name: str, body: PresetMetaUpdateBody) -> dict:
    """Update preset metadata (name and description only; version unchanged)."""
    preset_dir = _get_preset_dir(section)
    filepath = _resolve_preset_path(name, preset_dir)

    raw = json.loads(filepath.read_text())
    meta_raw = raw.get("_meta", {})
    meta = PresetMeta(
        name=body.name,
        description=body.description,
        version=meta_raw.get("version", 1),
    )
    content = {k: v for k, v in raw.items() if k != "_meta"}
    _write_preset_file(filepath, meta, content)
    return _read_preset_file(filepath)


@router.delete("/{section}/{name}")
def delete_preset(
    section: str,
    name: str,
    scope: str = Query(default="one"),
) -> dict:
    """Delete a preset. scope='all' deletes every version with the same display name."""
    preset_dir = _get_preset_dir(section)
    filepath = _resolve_preset_path(name, preset_dir)

    if scope == "all":
        raw = json.loads(filepath.read_text())
        display_name = raw.get("_meta", {}).get("name", name)
        files_to_delete = _find_versions_by_name(preset_dir, display_name)
        deleted = []
        for f in files_to_delete:
            deleted.append(f.stem)
            f.unlink()
        return {"deleted": deleted, "status": "deleted"}

    filepath.unlink()
    return {"deleted": [name], "status": "deleted"}
