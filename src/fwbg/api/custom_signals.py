"""CRUD API for custom signal definitions."""
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fwbg.api.deps import get_plugin_registry
from fwbg.pipeline.registry import get_core_plugins_dir
import fwbg.pipeline.registry as _registry_mod

router = APIRouter(prefix="/custom-signals", tags=["custom-signals"])


def _get_definitions_dir() -> Path:
    """Get the custom signal definitions directory."""
    d = get_core_plugins_dir() / "custom" / "indicators" / "signals" / "definitions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slugify(name: str) -> str:
    """Convert a signal name to a safe filename slug."""
    slug = re.sub(r"[^a-z0-9_-]", "_", name.lower().strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def _invalidate_registry() -> None:
    """Invalidate the cached plugin registry so new definitions are picked up."""
    # Reset global singleton so it's recreated fresh on next access
    _registry_mod._global_registry = None
    get_plugin_registry.cache_clear()


class ConditionSchema(BaseModel):
    column: str
    op: str
    value: Any


class RuleSchema(BaseModel):
    output: str
    logic: str = "AND"
    conditions: List[ConditionSchema]


class SignalDefinition(BaseModel):
    name: str
    description: str = ""
    version: str = "1.0.0"
    dependencies: List[str] = []
    rules: List[RuleSchema]


@router.get("")
def list_signals() -> List[Dict[str, Any]]:
    """List all custom signal definitions."""
    defs_dir = _get_definitions_dir()
    signals = []
    for f in sorted(defs_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            signals.append({
                "name": data.get("name", f.stem),
                "description": data.get("description", ""),
                "version": data.get("version", "1.0.0"),
                "filename": f.stem,
                "rules_count": len(data.get("rules", [])),
                "dependencies": data.get("dependencies", []),
            })
        except (json.JSONDecodeError, IOError):
            continue
    return signals


@router.get("/{name}")
def get_signal(name: str) -> Dict[str, Any]:
    """Load a single signal definition."""
    defs_dir = _get_definitions_dir()
    path = defs_dir / f"{_slugify(name)}.json"
    if not path.exists():
        raise HTTPException(404, f"Signal '{name}' not found")
    return json.loads(path.read_text())


@router.post("")
def create_signal(body: SignalDefinition) -> Dict[str, str]:
    """Create a new custom signal definition."""
    defs_dir = _get_definitions_dir()
    slug = _slugify(body.name)
    if not slug:
        raise HTTPException(400, "Invalid signal name")

    path = defs_dir / f"{slug}.json"
    if path.exists():
        raise HTTPException(409, f"Signal '{body.name}' already exists")

    path.write_text(json.dumps(body.model_dump(), indent=2))
    _invalidate_registry()
    return {"name": body.name, "filename": slug}


@router.put("/{name}")
def update_signal(name: str, body: SignalDefinition) -> Dict[str, str]:
    """Update an existing signal definition."""
    defs_dir = _get_definitions_dir()
    old_path = defs_dir / f"{_slugify(name)}.json"
    if not old_path.exists():
        raise HTTPException(404, f"Signal '{name}' not found")

    new_slug = _slugify(body.name)
    new_path = defs_dir / f"{new_slug}.json"

    # If name changed, remove old file
    if old_path != new_path:
        old_path.unlink()

    new_path.write_text(json.dumps(body.model_dump(), indent=2))
    _invalidate_registry()
    return {"name": body.name, "filename": new_slug}


@router.delete("/{name}")
def delete_signal(name: str) -> Dict[str, str]:
    """Delete a custom signal definition."""
    defs_dir = _get_definitions_dir()
    path = defs_dir / f"{_slugify(name)}.json"
    if not path.exists():
        raise HTTPException(404, f"Signal '{name}' not found")

    path.unlink()
    _invalidate_registry()
    return {"status": "deleted", "name": name}
