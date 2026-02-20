"""Strategy file endpoints."""
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fwbg.api.deps import get_strategies_dir
from fwbg.core.config import _parse_grids, _resolve_section

router = APIRouter(prefix="/strategies", tags=["strategies"])


def _serialize_grid(grid) -> dict:
    """Serialize a GridConfig object to a plain dict for API responses."""
    result: dict = {
        "tp": grid.tp,
        "sl": grid.sl,
        "ct": grid.ct,
        "timeout_bars": grid.timeout_bars,
    }
    if grid.regime_filter_grid.condition_grids:
        result["regime_filter_grid"] = {
            "condition_grids": grid.regime_filter_grid.condition_grids
        }
    if grid.long_tp is not None:
        result["long_tp"] = grid.long_tp
    if grid.long_sl is not None:
        result["long_sl"] = grid.long_sl
    if grid.long_ct is not None:
        result["long_ct"] = grid.long_ct
    if grid.short_tp is not None:
        result["short_tp"] = grid.short_tp
    if grid.short_sl is not None:
        result["short_sl"] = grid.short_sl
    if grid.short_ct is not None:
        result["short_ct"] = grid.short_ct
    return result


class StrategyCreate(BaseModel):
    """Request body for creating/updating a strategy."""
    name: str
    data: dict


@router.get("")
def list_strategies() -> list[dict]:
    """List all strategy files."""
    strategies_dir = get_strategies_dir()
    strategies = []

    for f in sorted(strategies_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            strategies.append({
                "filename": f.stem,
                "name": data.get("name", f.stem),
                "description": data.get("description", ""),
                "tags": data.get("tags", []),
            })
        except (json.JSONDecodeError, IOError):
            continue

    return strategies


@router.get("/{name}")
def get_strategy(name: str) -> dict:
    """Load a strategy JSON file, resolving preset references."""
    strategies_dir = get_strategies_dir()
    filepath = strategies_dir / f"{name}.json"

    if not filepath.exists():
        raise HTTPException(404, f"Strategy not found: {name}")

    try:
        data = json.loads(filepath.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Invalid JSON in strategy file: {e}")

    # Extract preset refs BEFORE resolving (to preserve string identifiers)
    refs: dict = {}
    _simple_sections = ["pipeline", "exit_params", "model", "validation", "filters", "resources", "risk_params"]
    for key in _simple_sections:
        if key in data and isinstance(data[key], str):
            refs[key] = data[key]

    grids_raw = data.get("grids", {})
    if isinstance(grids_raw, dict) and "assignments" in grids_raw:
        refs["grids_regime_filter"] = grids_raw.get("regime_filter_grid")
        grids_refs: dict = {}
        for asset_class, assignment in grids_raw["assignments"].items():
            if isinstance(assignment, str):
                grids_refs[asset_class] = {"name": assignment, "overrides": {}}
            elif isinstance(assignment, dict) and "preset" in assignment:
                overrides = {k: v for k, v in assignment.items() if k != "preset"}
                grids_refs[asset_class] = {"name": assignment["preset"], "overrides": overrides}
            else:
                grids_refs[asset_class] = None
        refs["grids"] = grids_refs

    # Resolve string preset references to inline dicts
    strategy_dir = str(filepath.parent.resolve())
    _section_dirs = {
        "pipeline": "pipelines",
        "exit_params": "exit_params",
        "model": "models",
        "validation": "validations",
        "filters": "filters",
        "resources": "resources",
        "risk_params": "risk_params",
    }
    for key, section_dir in _section_dirs.items():
        if key in data:
            data[key] = _resolve_section(data[key], section_dir, strategy_dir)

    # Resolve grids: convert preset-based format to flat {asset_class: grid_dict}
    if "grids" in data:
        try:
            resolved_grids = _parse_grids(data["grids"], strategy_dir)
            data["grids"] = {k: _serialize_grid(v) for k, v in resolved_grids.items()}
        except (FileNotFoundError, ValueError, KeyError):
            pass  # Leave grids as-is if resolution fails

    data["_refs"] = refs
    return data


@router.post("")
def create_strategy(body: StrategyCreate) -> dict:
    """Create a new strategy file."""
    strategies_dir = get_strategies_dir()
    filename = body.name.replace(" ", "_").lower()
    filepath = strategies_dir / f"{filename}.json"

    if filepath.exists():
        raise HTTPException(409, f"Strategy already exists: {filename}")

    body.data["name"] = body.name
    filepath.write_text(json.dumps(body.data, indent=2))

    return {"filename": filename, "name": body.name, "status": "created"}


@router.put("/{name}")
def update_strategy(name: str, body: dict) -> dict:
    """Update an existing strategy file."""
    strategies_dir = get_strategies_dir()
    filepath = strategies_dir / f"{name}.json"

    if not filepath.exists():
        raise HTTPException(404, f"Strategy not found: {name}")

    filepath.write_text(json.dumps(body, indent=2))
    return {"filename": name, "status": "updated"}


@router.delete("/{name}")
def delete_strategy(name: str) -> dict:
    """Delete a strategy file."""
    strategies_dir = get_strategies_dir()
    filepath = strategies_dir / f"{name}.json"

    if not filepath.exists():
        raise HTTPException(404, f"Strategy not found: {name}")

    filepath.unlink()
    return {"filename": name, "status": "deleted"}
