"""Preset CRUD endpoints for reusable strategy sub-configurations."""
import json
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fwbg.api.deps import get_strategies_dir
from fwbg.core.config import _load_json_preset

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


class PresetCreate(BaseModel):
    name: str
    content: dict


def _get_preset_dir(section: str) -> str:
    """Resolve the filesystem path for a preset section directory."""
    if section not in SECTION_DIRS:
        raise HTTPException(404, f"Unknown preset section: {section}")
    strategies_dir = get_strategies_dir()
    # Configs live in strategies/configs/, presets live in strategies/<section>/
    base = strategies_dir.parent
    path = base / SECTION_DIRS[section]
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


@router.get("/{section}")
def list_presets(section: str) -> list[dict]:
    """List all available presets for a section."""
    preset_dir = _get_preset_dir(section)
    presets = []
    for f in sorted(os.scandir(preset_dir), key=lambda e: e.name):
        if not f.name.endswith(".json"):
            continue
        try:
            content = json.loads(open(f.path).read())
            presets.append({"name": f.name[:-5], "content": content})
        except (json.JSONDecodeError, IOError):
            continue
    return presets


@router.post("/{section}")
def create_preset(section: str, body: PresetCreate) -> dict:
    """Create a new preset file."""
    preset_dir = _get_preset_dir(section)
    filename = body.name.replace(" ", "_").lower()
    filepath = os.path.join(preset_dir, f"{filename}.json")

    # Path traversal guard
    if not os.path.realpath(filepath).startswith(os.path.realpath(preset_dir)):
        raise HTTPException(400, "Invalid preset name")

    if os.path.exists(filepath):
        raise HTTPException(409, f"Preset already exists: {filename}")

    with open(filepath, "w") as f:
        json.dump(body.content, f, indent=2)

    return {"name": filename, "status": "created"}


@router.get("/{section}/{name}")
def get_preset(section: str, name: str) -> dict:
    """Load a single preset by name."""
    preset_dir = _get_preset_dir(section)
    try:
        content = _load_json_preset(name, preset_dir)
    except FileNotFoundError:
        raise HTTPException(404, f"Preset not found: {section}/{name}")
    return {"name": name, "content": content}


@router.put("/{section}/{name}")
def update_preset(section: str, name: str, body: dict) -> dict:
    """Update an existing preset file."""
    preset_dir = _get_preset_dir(section)
    filepath = os.path.join(preset_dir, f"{name}.json")

    if not os.path.realpath(filepath).startswith(os.path.realpath(preset_dir)):
        raise HTTPException(400, "Invalid preset name")
    if not os.path.exists(filepath):
        raise HTTPException(404, f"Preset not found: {section}/{name}")

    with open(filepath, "w") as f:
        json.dump(body, f, indent=2)

    return {"name": name, "status": "updated"}


@router.delete("/{section}/{name}")
def delete_preset(section: str, name: str) -> dict:
    """Delete a preset file."""
    preset_dir = _get_preset_dir(section)
    filepath = os.path.join(preset_dir, f"{name}.json")

    if not os.path.realpath(filepath).startswith(os.path.realpath(preset_dir)):
        raise HTTPException(400, "Invalid preset name")
    if not os.path.exists(filepath):
        raise HTTPException(404, f"Preset not found: {section}/{name}")

    os.unlink(filepath)
    return {"name": name, "status": "deleted"}
