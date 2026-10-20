"""Record what study scripts evaluate outside the gated run path.

A script that evaluates strategy variants itself (not through
``python -m fwbg.cli``) records every variant with :func:`register_trial`, so
the register's trial count stays honest. Import it from here, not from
``fwbg.research``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from fwbg.api.workspace import get_research_dir
from fwbg.core.git_env import source_dir
from fwbg.research.card import CARD_ID_RE, CardError, ResearchCard, load_card
from fwbg.research.fingerprint import FINGERPRINT_VERSION, canonical_config
from fwbg.research.gate import Denial, DenialCode, RunRequest, check
from fwbg.research.ledger import (
    LEDGER_FILENAME,
    LedgerError,
    Trial,
    TrialKind,
    TrialStatus,
    append_trial,
    read_trials,
)
from fwbg.research.legacy import run_code, run_window
from fwbg.research.pending import PendingError
from fwbg.research.repo import (
    REGISTER_PATHS,
    ResearchRepoError,
    commit_register,
    register_lock,
    store_config,
)
from fwbg.research.runs import GateDenied, _warn_uncommitted, running_duplicates
from fwbg.results.storage import get_git_info


def register_trial(
    research_id: str,
    config: Mapping[str, Any],
    symbols: Sequence[str],
    metrics: Mapping[str, Mapping[str, float | int | None]],
    note: str,
    *,
    reason: str | None = None,
    failed: bool = False,
    run_id: str | None = None,
    research_dir: Path | None = None,
) -> Trial:
    """Record one evaluated variant of a study script as an iteration trial.

    Call it for every variant the script evaluated, discarded and failed ones too.
    *config* is what the variant evaluated, including ``end_date`` (and
    ``start_date``) of its data; it is fingerprinted like a run's metadata.
    The register's gate applies (active card, frozen criteria and cutoff, end on
    or before the cutoff, duplicate configuration needs *reason*).

    Raises:
        GateDenied: the register refused the trial; nothing was written.
    """
    if research_dir is None:
        research_dir = get_research_dir(create=False)
    end = config.get("end_date")
    if not isinstance(end, str) or not end.strip():
        # A study has already run: the gate cannot clamp its window to the cutoff.
        message = "a study trial must state the end_date of the data it evaluated"
        raise GateDenied([Denial(DenialCode.INVALID_WINDOW, message)])
    canonical = tuple(sorted({symbol.upper() for symbol in symbols}))
    request = RunRequest(research_id, config, canonical, note, reason)
    status = TrialStatus.FAILED if failed else TrialStatus.OK
    try:
        with register_lock(research_dir):
            return _record(research_dir, request, status, metrics, run_id)
    except GateDenied:
        raise
    except (OSError, CardError, LedgerError, PendingError, ResearchRepoError) as exc:
        # Only reached before the append: a failed commit after it only warns.
        denial = Denial(DenialCode.REGISTER_UNREADABLE, f"research register: {exc}")
        raise GateDenied([denial]) from exc


def _record(
    research_dir: Path,
    request: RunRequest,
    status: TrialStatus,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    run_id: str | None,
) -> Trial:
    card: ResearchCard | None = None
    if request.research_id is not None and CARD_ID_RE.fullmatch(request.research_id):
        card = load_card(research_dir, request.research_id)
    trials = read_trials(research_dir / LEDGER_FILENAME)
    decision = check(card, request, trials)
    denials = list(decision.denials)
    denials += running_duplicates(research_dir, decision.fingerprint, request.reason)
    if denials:
        raise GateDenied(denials)
    assert card is not None and decision.fingerprint is not None  # allowed decisions have both
    metadata = copy.deepcopy(dict(decision.effective_metadata))
    trial = Trial(
        strategy=card.id,
        run_id=run_id,
        kind=TrialKind.ITERATION,
        status=status,
        ts=datetime.now().isoformat(timespec="seconds"),
        fingerprint=decision.fingerprint,
        fingerprint_version=FINGERPRINT_VERSION,
        criteria_hash=card.criteria_hash,
        holdout_cutoff=card.holdout_cutoff,
        code=run_code({"git": get_git_info(source_dir())}),
        window=run_window(metadata),
        note=request.note,
        reason=request.reason if request.reason and request.reason.strip() else None,
        metrics={symbol: dict(values) for symbol, values in metrics.items()},
    )
    # Config first: the ledger must never name a fingerprint without its config.
    store_config(research_dir, decision.fingerprint, canonical_config(metadata, request.symbols))
    stored = append_trial(research_dir / LEDGER_FILENAME, trial)
    try:
        commit_register(research_dir, REGISTER_PATHS, f"trial {stored.trial_id}: {card.id} study")
    except ResearchRepoError as exc:
        _warn_uncommitted(str(stored.trial_id), exc)
    return stored
