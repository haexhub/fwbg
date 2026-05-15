"""Git-based versioning utilities for strategy configs."""
import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _run(args: list[str], cwd: Path) -> str:
    """Run a git command and return stdout. Raises on error."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def is_git_repo(path: Path) -> bool:
    """Check if the given path is inside a git repository."""
    try:
        _run(["rev-parse", "--git-dir"], cwd=path)
        return True
    except (RuntimeError, FileNotFoundError):
        return False


def ensure_git_repo(path: Path) -> None:
    """Initialize a git repo at *path* if one doesn't exist yet."""
    if is_git_repo(path):
        return
    path.mkdir(parents=True, exist_ok=True)
    _run(["init"], cwd=path)
    log.info("Initialized git repository at %s", path)


def get_git_identity(repo_dir: Path) -> dict[str, str]:
    """Return the local git user.name and user.email (empty strings if unset)."""
    name = ""
    email = ""
    try:
        name = _run(["config", "user.name"], cwd=repo_dir)
    except RuntimeError:
        pass
    try:
        email = _run(["config", "user.email"], cwd=repo_dir)
    except RuntimeError:
        pass
    return {"name": name, "email": email}


def set_git_identity(repo_dir: Path, name: str, email: str) -> None:
    """Set local git user.name and user.email for the repo."""
    _run(["config", "user.name", name], cwd=repo_dir)
    _run(["config", "user.email", email], cwd=repo_dir)
    log.info("Set git identity for %s: %s <%s>", repo_dir, name, email)


def commit_file(repo_dir: Path, filename: str, message: str) -> str:
    """Stage a single file and commit it. Returns the new commit hash."""
    _run(["add", filename], cwd=repo_dir)
    # Check if there's actually anything to commit
    status = _run(["status", "--porcelain", filename], cwd=repo_dir)
    if not status:
        # Nothing changed — return current HEAD
        return _run(["rev-parse", "HEAD"], cwd=repo_dir)
    _run(["commit", "-m", message, "--", filename], cwd=repo_dir)
    return _run(["rev-parse", "HEAD"], cwd=repo_dir)


def file_history(repo_dir: Path, filename: str, limit: int = 50) -> list[dict]:
    """Return git log for a single file as a list of dicts."""
    fmt = "%H\x1f%h\x1f%ai\x1f%s"
    try:
        out = _run(
            ["log", f"--max-count={limit}", f"--format={fmt}", "--", filename],
            cwd=repo_dir,
        )
    except RuntimeError:
        return []
    if not out:
        return []
    entries = []
    for line in out.splitlines():
        parts = line.split("\x1f", 3)
        if len(parts) == 4:
            entries.append({
                "hash": parts[0],
                "short_hash": parts[1],
                "date": parts[2],
                "message": parts[3],
            })
    return entries


def file_at_commit(repo_dir: Path, filename: str, ref: str) -> str:
    """Return the raw file content at a given git ref."""
    return _run(["show", f"{ref}:{filename}"], cwd=repo_dir)
