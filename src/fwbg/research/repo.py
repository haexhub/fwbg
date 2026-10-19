"""Layout and git repository of the research register."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fwbg.research.ledger import LEDGER_FILENAME

CRITERIA_TEMPLATE = Path("templates") / "criteria-default.yaml"
_DIRECTORIES = ("strategies", "configs", "templates", "reports")
_GITIGNORE = ".gitignore"
_GITIGNORE_TEXT = ".staging/\n.register.lock\n"

# Git hooks export GIT_DIR and GIT_INDEX_FILE; inheriting them would write private
# research data into another repository (list: `git rev-parse --local-env-vars`).
GIT_LOCAL_ENV_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_PREFIX",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_GRAFT_FILE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
)

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
    env = {name: value for name, value in os.environ.items() if name not in GIT_LOCAL_ENV_VARS}
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
