"""Record what study scripts evaluate outside the gated run path.

A script that evaluates strategy variants itself (not through
``python -m fwbg.cli``) records every variant with :func:`register_trial`, so
the register's trial count stays honest. Import it from here, not from
``fwbg.research``.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from fwbg_sdk.enums import Timeframe

from fwbg.api.workspace import get_research_dir
from fwbg.core.git_env import source_dir
from fwbg.core.presets import resolve_presets
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
    readable_line,
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
from fwbg.research.runs import GateDenied, record_stale, running_duplicates, warn_uncommitted
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
        GateDenied: the register refused the trial (nothing was written for it) or
            could not be read or written.
    """
    if research_dir is None:
        research_dir = get_research_dir(create=False)
    request = _request(research_id, config, symbols, note, reason)
    upper_metrics = {symbol.upper(): dict(values) for symbol, values in metrics.items()}
    if len(upper_metrics) != len(metrics):
        counts = Counter(symbol.upper() for symbol in metrics)
        repeated = sorted(symbol for symbol, count in counts.items() if count > 1)
        message = f"metrics name a symbol more than once: {', '.join(repeated)}"
        raise GateDenied([Denial(DenialCode.INVALID_CONFIG, message)])
    status = TrialStatus.FAILED if failed else TrialStatus.OK
    code = run_code({"git": get_git_info(source_dir())})
    try:
        with register_lock(research_dir):
            return _record(research_dir, request, status, upper_metrics, run_id, code)
    except (OSError, CardError, LedgerError, PendingError, ResearchRepoError) as exc:
        # ponytail: an OSError while append_trial flushes its line reports the register
        # unreadable although the trial may be stored; check `research show` before retrying.
        denial = Denial(DenialCode.REGISTER_UNREADABLE, f"research register: {exc}")
        raise GateDenied([denial]) from exc


def _request(
    research_id: str,
    config: Mapping[str, Any],
    symbols: Sequence[str],
    note: str,
    reason: str | None,
) -> RunRequest:
    """The gate request for *config*, comparable with a CLI run's; raises :class:`GateDenied`."""
    end = config.get("end_date")
    if not isinstance(end, str) or not end.strip():
        # A study has already run: the gate cannot clamp its window to the cutoff.
        message = (
            "a study trial must state the end_date of the data it evaluated "
            "as a YYYY-MM-DD string"
        )
        raise GateDenied([Denial(DenialCode.INVALID_WINDOW, message)])
    if isinstance(symbols, str):
        message = f"symbols must be a list of symbols, not the string {symbols!r}"
        raise GateDenied([Denial(DenialCode.INVALID_CONFIG, message)])
    try:
        # Like prepare_metadata, without a default timeframe: a study states its own.
        metadata = resolve_presets(config)
        if metadata.get("timeframe") is not None:
            metadata["timeframe"] = Timeframe.from_str(metadata["timeframe"]).canonical
    except (OSError, ValueError) as exc:
        message = f"configuration cannot be prepared: {exc}"
        raise GateDenied([Denial(DenialCode.INVALID_CONFIG, message)]) from exc
    canonical = tuple(sorted({symbol.upper() for symbol in symbols}))
    return RunRequest(research_id, metadata, canonical, note, reason)


def _record(
    research_dir: Path,
    request: RunRequest,
    status: TrialStatus,
    metrics: Mapping[str, Mapping[str, float | int | None]],
    run_id: str | None,
    code: Mapping[str, Any],
) -> Trial:
    record_stale(research_dir)  # a killed run's trial must count before this one is gated
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
        code=code,
        window=run_window(metadata),
        note=request.note,
        reason=request.reason if request.reason and request.reason.strip() else None,
        metrics=metrics,
    )
    try:
        # The id is a stand-in: append_trial assigns the real one.
        readable_line(replace(trial, trial_id="T-000001"))
    except ValueError as exc:
        message = f"trial cannot be stored: {exc}"
        raise GateDenied([Denial(DenialCode.INVALID_CONFIG, message)]) from exc
    # Config first: the ledger must never name a fingerprint without its config.
    store_config(research_dir, decision.fingerprint, canonical_config(metadata, request.symbols))
    stored = append_trial(research_dir / LEDGER_FILENAME, trial)
    try:
        commit_register(research_dir, REGISTER_PATHS, f"trial {stored.trial_id}: {card.id} study")
    except ResearchRepoError as exc:
        warn_uncommitted(str(stored.trial_id), exc)
    return stored
