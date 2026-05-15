"""Strategy file endpoints."""
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from fwbg.api.deps import get_strategies_dir
from fwbg.api.git_utils import (
    commit_file, ensure_git_repo, file_at_commit, file_history,
    get_git_identity, is_git_repo, set_git_identity,
)
from fwbg.core.config import _resolve_section

router = APIRouter(prefix="/strategies", tags=["strategies"])

# Strategy JSON field name → relative preset subdirectory
SECTION_FIELD_DIRS: dict[str, str] = {
    "pipeline": "pipelines",
    "exit_params": "exit_params",
    "model": "models",
    "validation": "validations",
    "filters": "filters",
    "resources": "resources",
    "risk_params": "risk_params",
}


class StrategyCreate(BaseModel):
    """Request body for creating/updating a strategy."""
    name: str
    data: dict


class CommitRequest(BaseModel):
    """Request body for committing a strategy change."""
    message: str = ""


class GitIdentityRequest(BaseModel):
    """Request body for setting git author identity."""
    name: str
    email: str


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


# ── Git identity (must be registered before /{name} catch-all) ──


@router.get("/git/identity")
def get_identity() -> dict:
    """Return the current git identity for the strategies repo."""
    strategies_dir = get_strategies_dir()
    ensure_git_repo(strategies_dir)
    return get_git_identity(strategies_dir)


@router.put("/git/identity")
def set_identity(body: GitIdentityRequest) -> dict:
    """Set git user.name and user.email for the strategies repo."""
    if not body.name.strip() or not body.email.strip():
        raise HTTPException(422, "Both name and email are required")
    strategies_dir = get_strategies_dir()
    ensure_git_repo(strategies_dir)
    set_git_identity(strategies_dir, body.name.strip(), body.email.strip())
    return {"status": "ok", "name": body.name.strip(), "email": body.email.strip()}


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

    # Resolve string preset references to inline dicts
    strategy_dir = str(filepath.parent.resolve())
    for key, section_dir in SECTION_FIELD_DIRS.items():
        if key in data:
            data[key] = _resolve_section(data[key], section_dir, strategy_dir)

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

    if is_git_repo(strategies_dir):
        try:
            commit_file(strategies_dir, f"{filename}.json", f"create: {filename}")
        except RuntimeError:
            pass  # git not critical

    return {"filename": filename, "name": body.name, "status": "created"}


@router.put("/{name}")
def update_strategy(name: str, body: dict) -> dict:
    """Update an existing strategy file (write only, no git commit)."""
    strategies_dir = get_strategies_dir()
    filepath = strategies_dir / f"{name}.json"

    if not filepath.exists():
        raise HTTPException(404, f"Strategy not found: {name}")

    filepath.write_text(json.dumps(body, indent=2))
    return {"filename": name, "status": "updated"}


@router.post("/{name}/commit")
def commit_strategy(name: str, body: CommitRequest) -> dict:
    """Commit the current state of a strategy file to git.

    Auto-initializes the git repo on first use.  Returns a 428 error
    with ``code: "identity_required"`` when user.name / user.email are
    not configured yet — the client should prompt the user and call
    ``PUT /strategies/git/identity`` before retrying.
    """
    strategies_dir = get_strategies_dir()
    filepath = strategies_dir / f"{name}.json"

    if not filepath.exists():
        raise HTTPException(404, f"Strategy not found: {name}")

    ensure_git_repo(strategies_dir)

    identity = get_git_identity(strategies_dir)
    if not identity["name"] or not identity["email"]:
        raise HTTPException(428, detail={
            "code": "identity_required",
            "message": "Git author identity not configured",
            "current": identity,
        })

    message = body.message.strip() or f"update: {name}"
    try:
        commit_hash = commit_file(strategies_dir, f"{name}.json", message)
    except RuntimeError as e:
        raise HTTPException(500, f"Git commit failed: {e}")

    return {"filename": name, "hash": commit_hash, "status": "committed"}


@router.get("/{name}/history")
def strategy_history(name: str) -> list[dict]:
    """Return git commit history for a strategy file."""
    strategies_dir = get_strategies_dir()
    filepath = strategies_dir / f"{name}.json"

    if not filepath.exists():
        raise HTTPException(404, f"Strategy not found: {name}")

    if not is_git_repo(strategies_dir):
        return []

    return file_history(strategies_dir, f"{name}.json")


@router.get("/{name}/version/{ref}")
def strategy_version(name: str, ref: str) -> dict:
    """Load a specific git version of a strategy."""
    strategies_dir = get_strategies_dir()

    if not is_git_repo(strategies_dir):
        raise HTTPException(501, "Strategies directory is not a git repository")

    try:
        raw = file_at_commit(strategies_dir, f"{name}.json", ref)
    except RuntimeError as e:
        raise HTTPException(404, f"Version not found: {e}")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"Invalid JSON at ref {ref}: {e}")


@router.delete("/{name}")
def delete_strategy(name: str) -> dict:
    """Delete a strategy file."""
    strategies_dir = get_strategies_dir()
    filepath = strategies_dir / f"{name}.json"

    if not filepath.exists():
        raise HTTPException(404, f"Strategy not found: {name}")

    filepath.unlink()
    return {"filename": name, "status": "deleted"}
