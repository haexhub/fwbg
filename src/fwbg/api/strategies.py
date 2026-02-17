"""Strategy file endpoints."""
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fwbg.api.deps import get_strategies_dir

router = APIRouter(prefix="/strategies", tags=["strategies"])


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
    """Load a strategy JSON file."""
    strategies_dir = get_strategies_dir()
    filepath = strategies_dir / f"{name}.json"

    if not filepath.exists():
        raise HTTPException(404, f"Strategy not found: {name}")

    try:
        return json.loads(filepath.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Invalid JSON in strategy file: {e}")


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
