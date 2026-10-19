"""Layout and git repository of the research register."""

from __future__ import annotations

import subprocess
from pathlib import Path

from fwbg.research.ledger import LEDGER_FILENAME

CRITERIA_TEMPLATE = Path("templates") / "criteria-default.yaml"
_DIRECTORIES = ("strategies", "configs", "templates", "reports")

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
    """The research register's git repository could not be created."""


def init_research_repo(research_dir: Path) -> bool:
    """Create the register layout and its git repository.

    Idempotent: existing files are never overwritten. Returns ``True`` if this
    call created the git repository.
    """
    research_dir.mkdir(parents=True, exist_ok=True)
    for name in _DIRECTORIES:
        (research_dir / name).mkdir(exist_ok=True)
    template = research_dir / CRITERIA_TEMPLATE
    if not template.exists():
        template.write_text(DEFAULT_CRITERIA_YAML, encoding="utf-8")
    (research_dir / LEDGER_FILENAME).touch()
    if (research_dir / ".git").exists():
        return False
    _git(research_dir, "init", "--quiet")
    _git(research_dir, "add", "--all")
    _git(research_dir, "commit", "--quiet", "--message", "chore: initialise research register")
    return True


def _git(cwd: Path, *args: str) -> None:
    try:
        subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=30
        )
    except subprocess.CalledProcessError as exc:
        raise ResearchRepoError(f"git {args[0]} failed: {exc.stderr.strip()}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ResearchRepoError(f"git {args[0]} failed: {exc}") from exc
