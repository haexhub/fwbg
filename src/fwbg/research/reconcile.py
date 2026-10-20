"""``fwbg research reconcile``: repair the register after crashes and report its gaps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fwbg.api.run_service import ACTIVE_STATUSES
from fwbg.api.workspace import get_strategies_dir
from fwbg.research.backfill import ungated_runs
from fwbg.research.card import load_card
from fwbg.research.ledger import LEDGER_FILENAME, Trial, read_trials
from fwbg.research.legacy import scan_runs
from fwbg.research.pending import STAGING, recover_stale
from fwbg.research.repo import REGISTER_PATHS, commit_register, register_lock
from fwbg.research.views import card_ids


@dataclass(frozen=True)
class ReconcileResult:
    recovered: tuple[str, ...]  # trial ids recorded from stale markers
    committed: bool  # a recovery commit was made
    unindexed: tuple[str, ...]  # run ids in the workspace that are neither in the ledger nor tagged


def reconcile(research_dir: Path, results_dir: Path) -> ReconcileResult:
    """Validate the register, record stale runs, commit what is uncommitted, report gaps.

    Idempotent. Raises (and changes nothing) when the ledger or a card does not
    parse; a failed commit raises ResearchRepoError with the data left in place.
    A run is assigned to a card only by the operator, through ``research backfill``.
    """
    recovered, committed = repair(research_dir)
    return ReconcileResult(
        recovered=tuple(str(trial.trial_id) for trial in recovered),
        committed=committed,
        unindexed=unindexed_runs(research_dir, results_dir),
    )


def repair(research_dir: Path) -> tuple[list[Trial], bool]:
    """Record the stale runs and commit the register; the recorded trials and whether
    a commit was made. Raises, without writing, if the ledger or a card does not parse."""
    # Before the lock: a missing register fails with the init hint and gets no lock file.
    read_trials(research_dir / LEDGER_FILENAME)
    with register_lock(research_dir):
        # Validated under the lock, so nothing changes between the check and the writes.
        read_trials(research_dir / LEDGER_FILENAME)
        for card_id in card_ids(research_dir):
            load_card(research_dir, card_id)
        recovered = recover_stale(research_dir)
        committed = commit_register(
            research_dir, (*REGISTER_PATHS, "reports"), "chore: reconcile register"
        )
    return recovered, committed


def unindexed_runs(research_dir: Path, results_dir: Path) -> tuple[str, ...]:
    """Run ids under *results_dir* that are neither in the ledger nor tagged.

    Runs that may still be running (a pending marker, or an active or unreadable
    ``progress.json``) are not read: their files may be half-written.
    """
    pending = {path.stem for path in (research_dir / STAGING).glob("*.json")}
    skip = pending | set(runs_in_progress(results_dir))
    runs = scan_runs(results_dir, get_strategies_dir().parent, skip)
    # ungated_runs keeps legacy trials (backfill re-applies them); here any ledger entry counts.
    recorded = {trial.run_id for trial in read_trials(research_dir / LEDGER_FILENAME)}
    return tuple(
        run.run_id for run in ungated_runs(runs, research_dir) if run.run_id not in recorded
    )


def runs_in_progress(results_dir: Path) -> list[str]:
    """Run ids whose ``progress.json`` reports an active run or cannot be read."""
    active = []
    for progress in sorted(results_dir.glob("*/progress.json")):
        try:
            state = json.loads(progress.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = None
        # Stale "running" files exist: the backfill only warns about these runs and
        # reconcile leaves them out of its report. An unreadable file counts too.
        if not isinstance(state, dict) or str(state.get("status")) in ACTIVE_STATUSES:
            active.append(progress.parent.name)
    return active
