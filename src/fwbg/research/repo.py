"""Layout and git repository of the research register."""

from __future__ import annotations

import fcntl
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fwbg.core.git_env import isolated_git_env
from fwbg.research.fingerprint import DIGEST_RE, canonical_json, digest
from fwbg.research.ledger import LEDGER_FILENAME

CRITERIA_TEMPLATE = Path("templates") / "criteria-default.yaml"
_DIRECTORIES = ("strategies", "configs", "templates", "reports")
_GITIGNORE = ".gitignore"
REGISTER_LOCK = ".register.lock"
_GITIGNORE_TEXT = f".staging/\n{REGISTER_LOCK}\n"

DEFAULT_CRITERIA_YAML = """\
# Starting values for the success criteria of new research cards.
# Hand-curated; never derive these values from past results.
# Each card sets its own values at registration — max_drawdown in particular
# depends on the strategy and its position sizing.
min_sharpe: 1.0
min_trades: 100
max_drawdown: 0.40
min_fold_stability: 0.6
max_p_value: 0.05
"""


class ResearchRepoError(Exception):
    """The research register could not be initialised."""


def init_research_repo(research_dir: Path) -> bool:
    """Create the register layout and its git repository.

    Idempotent and recoverable: existing files are never overwritten, and a
    failed first commit is retried on the next call. Returns ``True`` if this
    call created the initial commit. Filesystem errors (``OSError``) pass
    through.

    Raises:
        ResearchRepoError: git failed, or the ledger is missing from an
            initialised register (it is never recreated, so the trial history
            cannot be reset by a re-run).
    """
    research_dir.mkdir(parents=True, exist_ok=True)
    # Empty directories are not tracked, so a clone of the register gets them here.
    for name in _DIRECTORIES:
        (research_dir / name).mkdir(exist_ok=True)
    _write_if_missing(research_dir / CRITERIA_TEMPLATE, DEFAULT_CRITERIA_YAML)
    if not (research_dir / ".git").exists():
        _git(research_dir, "init", "--quiet", "--initial-branch=main")
    ledger = research_dir / LEDGER_FILENAME
    if _has_commit(research_dir):
        if not ledger.exists():
            raise ResearchRepoError(
                f"{LEDGER_FILENAME} is missing from an initialised research register at "
                f"{research_dir}; restore it from git (git checkout -- {LEDGER_FILENAME})"
            )
        return False
    _write_if_missing(research_dir / _GITIGNORE, _GITIGNORE_TEXT)
    ledger.touch()
    # --force: a user's global excludes must not keep register files out; named paths only.
    _git(
        research_dir,
        "add",
        "--force",
        "--",
        CRITERIA_TEMPLATE.as_posix(),
        LEDGER_FILENAME,
        _GITIGNORE,
    )
    _git(research_dir, "commit", "--quiet", "--message", "chore: initialise research register")
    return True


@contextmanager
def register_lock(research_dir: Path) -> Iterator[None]:
    """Serialise register writes (ledger appends plus their commit) across processes.

    Separate from the ``flock`` that ``append_trial`` takes on the ledger itself:
    locking ``trials.jsonl`` again from the same process would deadlock.
    """
    with (research_dir / REGISTER_LOCK).open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def commit_register(research_dir: Path, paths: Sequence[str], message: str) -> bool:
    """Commit exactly *paths* (relative to the register); ``False`` if nothing changed."""
    # --force: global excludes must not hide register files; only the named paths are added.
    _git(research_dir, "add", "--force", "--", *paths)
    staged = _git(research_dir, "diff", "--cached", "--quiet", "--", *paths, ok_codes=(0, 1))
    if staged.returncode == 0:
        return False
    _git(research_dir, "commit", "--quiet", "--message", message, "--", *paths)
    return True


def store_config(research_dir: Path, fingerprint: str, config: Mapping[str, Any]) -> Path:
    """Store the fingerprinted *config* as ``configs/<hex>.json`` (content-addressed).

    Raises:
        ValueError: *fingerprint* is malformed or is not the digest of *config*.
        ResearchRepoError: a different file already exists under that name.
    """
    if not DIGEST_RE.fullmatch(fingerprint) or digest(config) != fingerprint:
        raise ValueError(f"{fingerprint!r} is not the fingerprint of the given config")
    path = research_dir / "configs" / f"{fingerprint.removeprefix('sha256:')}.json"
    text = canonical_json(config) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ResearchRepoError(f"{path.name} exists with different content")
        return path
    path.write_text(text, encoding="utf-8")
    return path


def _write_if_missing(path: Path, text: str) -> None:
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def _has_commit(research_dir: Path) -> bool:
    """Exit code 1 means no commit yet; any other failure (a broken repository) raises."""
    result = _git(research_dir, "rev-parse", "--verify", "--quiet", "HEAD", ok_codes=(0, 1))
    return result.returncode == 0


def _git(
    cwd: Path, *args: str, ok_codes: tuple[int, ...] = (0,)
) -> subprocess.CompletedProcess[str]:
    env = isolated_git_env()
    # Never search above the register: a damaged .git must not fall back to an enclosing repo.
    env["GIT_CEILING_DIRECTORIES"] = str(cwd.resolve().parent)
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ResearchRepoError(f"git {args[0]} failed: {exc}") from exc
    if result.returncode not in ok_codes:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ResearchRepoError(f"git {args[0]} failed: {detail}")
    return result
