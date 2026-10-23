"""Apply a backfill mapping: write legacy runs and documented studies into the register.

:func:`apply_mapping` writes the cards, legacy and study trials and stored
configs of a mapping validated by :mod:`fwbg.research.mapping`. The backfill
is one-shot: it runs on a fresh register before the step-3 gate goes live.
Applying the same mapping again only adds what is missing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fwbg.research.card import (
    CardError,
    ResearchCard,
    criteria_hash,
    load_card,
    render_card,
    write_card,
)
from fwbg.research.fingerprint import FINGERPRINT_VERSION, digest
from fwbg.research.ledger import (
    LEDGER_FILENAME,
    Trial,
    TrialKind,
    TrialStatus,
    append_trial,
    read_trials,
    readable_line,
)
from fwbg.research.legacy import LegacyRun
from fwbg.research.mapping import BackfillError, CardPlan, StudyTrial, parse_criteria
from fwbg.research.pending import STAGING
from fwbg.research.repo import commit_register, register_lock, store_config

COMMIT_MESSAGE = "chore: backfill legacy research"
_BACKFILL_KINDS = frozenset({TrialKind.LEGACY, TrialKind.SMOKE})
_GATED_KINDS = frozenset({TrialKind.ITERATION, TrialKind.HOLDOUT})
# Run ids of backfilled study trials; register_trial refuses them.
STUDY_PREFIX = "study:"


@dataclass(frozen=True)
class BackfillResult:
    """What one :func:`apply_mapping` call wrote, and whether it committed."""

    cards_written: int
    trials_appended: int
    committed: bool


@dataclass(frozen=True)
class _Writes:
    """Everything one apply writes, built and checked before the register is touched."""

    cards: tuple[tuple[ResearchCard, str], ...]  # new card and its slug
    configs: Mapping[str, Mapping[str, Any]]  # fingerprint -> canonical config
    trials: tuple[Trial, ...]


def apply_mapping(
    plans: Sequence[CardPlan],
    runs: Sequence[LegacyRun],
    research_dir: Path,
    default_criteria: Mapping[str, float],
    today: date,
) -> BackfillResult:
    """Write the mapped cards, their legacy and study trials and the runs' configs.

    Every scanned run outside the gate's record (see :func:`ungated_runs`)
    must be mapped exactly once, and no run or study may be dated after *today*. A card's criteria are *default_criteria* with the
    card's own criteria applied on top. Every card and trial is built and
    checked before the register is touched, so a defect raises
    :class:`BackfillError` without writing anything. Cards that already exist
    are kept and must agree with the mapping; trials already in the ledger
    are skipped.

    If a write fails part-way (a full disk, a failing commit), apply the same
    mapping again without editing it: the recorded trials are skipped and the
    rest is written and committed.
    """
    template = parse_criteria(default_criteria, "criteria template")
    _prepare(plans, runs, research_dir, template, today)
    with register_lock(research_dir):
        # Again under the lock: another writer may have changed the register meanwhile.
        writes = _prepare(plans, runs, research_dir, template, today)
        for card, slug in writes.cards:
            write_card(research_dir, card, slug)
        for fingerprint, config in writes.configs.items():
            store_config(research_dir, fingerprint, config)
        ledger = research_dir / LEDGER_FILENAME
        for trial in writes.trials:
            append_trial(ledger, trial)
        committed = commit_register(
            research_dir, ["strategies", "configs", LEDGER_FILENAME], COMMIT_MESSAGE
        )
    return BackfillResult(len(writes.cards), len(writes.trials), committed)


def ungated_runs(runs: Sequence[LegacyRun], research_dir: Path) -> list[LegacyRun]:
    """*runs* without the gated ones, which are on record already.

    A gated run is in the ledger as an iteration or holdout, or still has its
    pending marker, even when tagging its ``config.json`` failed.
    """
    return _ungated(runs, research_dir, read_trials(research_dir / LEDGER_FILENAME))


def _ungated(
    runs: Sequence[LegacyRun], research_dir: Path, trials: Sequence[Trial]
) -> list[LegacyRun]:
    gated = {trial.run_id for trial in trials if trial.kind in _GATED_KINDS}
    return [
        run
        for run in runs
        if run.run_id not in gated and not (research_dir / STAGING / f"{run.run_id}.json").exists()
    ]


def _check_coverage(plans: Sequence[CardPlan], runs_by_id: Mapping[str, LegacyRun]) -> None:
    """Reject unknown, repeated or unmapped runs before preparing any writes."""
    mapped = [run_id for plan in plans for run_id in plan.runs]
    unknown = sorted(set(mapped) - runs_by_id.keys())
    twice = sorted(run_id for run_id, count in Counter(mapped).items() if count > 1)
    missing = sorted(runs_by_id.keys() - set(mapped))
    problems = []
    if unknown:
        problems.append(f"unknown runs: {', '.join(unknown)}")
    if twice:
        problems.append(f"runs mapped more than once: {', '.join(twice)}")
    if missing:
        problems.append(f"unmapped runs: {', '.join(missing)}")
    if problems:
        raise BackfillError("; ".join(problems))


def _prepare(
    plans: Sequence[CardPlan],
    runs: Sequence[LegacyRun],
    research_dir: Path,
    template: Mapping[str, float],
    today: date,
) -> _Writes:
    """Build and check every card, config and trial of the apply; write nothing."""
    ledger = read_trials(research_dir / LEDGER_FILENAME)
    runs_by_id = {run.run_id: run for run in _ungated(runs, research_dir, ledger)}
    _check_coverage(plans, runs_by_id)
    recorded = _recorded(ledger)
    _check_ledger(plans, recorded)
    done = set(recorded)
    cards: list[tuple[ResearchCard, str]] = []
    configs: dict[str, Mapping[str, Any]] = {}
    trials: list[Trial] = []
    for plan in plans:
        cutoff = _cutoff(plan, runs_by_id, today)
        criteria = {**template, **(plan.criteria or {})}
        card = load_card(research_dir, plan.id)
        if card is None:
            card = _new_card(plan, cutoff, criteria, today)
            cards.append((card, plan.slug))
        else:
            _check_existing(card, plan, cutoff, criteria, research_dir)
        for run_id in plan.runs:
            run = runs_by_id[run_id]
            # Recorded runs too: the register never names a fingerprint without its config.
            if run.config is not None:
                configs[run.fingerprint] = run.config
            if run_id not in done:
                trials.append(_storable(_legacy_trial(card, run)))
                done.add(run_id)
        for study in plan.studies:
            study_run_id = _study_run_id(card.id, study.variant)
            if study_run_id not in done:
                trials.append(_storable(_study_trial(card, study, study_run_id)))
                done.add(study_run_id)
    return _Writes(tuple(cards), configs, tuple(trials))


def _recorded(trials: Sequence[Trial]) -> dict[str, str]:
    """Run id -> card of every trial an earlier backfill wrote."""
    return {
        trial.run_id: trial.strategy
        for trial in trials
        if trial.run_id is not None
        and (trial.kind in _BACKFILL_KINDS or trial.run_id.startswith(STUDY_PREFIX))
    }


def _check_ledger(plans: Sequence[CardPlan], recorded: Mapping[str, str]) -> None:
    """Every trial an earlier apply recorded must stay in the mapping, on the same card."""
    mapped = {run_id: plan.id for plan in plans for run_id in plan.runs}
    mapped.update(
        {_study_run_id(plan.id, study.variant): plan.id for plan in plans for study in plan.studies}
    )
    problems = []
    for run_id, card_id in sorted(recorded.items()):
        target = mapped.get(run_id)
        if target is None:
            problems.append(f"{run_id} is recorded on {card_id} but missing from the mapping")
        elif target != card_id:
            problems.append(f"{run_id} is recorded on {card_id} but mapped to {target}")
    if problems:
        raise BackfillError(
            "; ".join(problems) + "; the ledger keeps the grouping of the first apply: "
            "reset the register (it is not pushed yet) if the grouping really changes"
        )


def _check_existing(
    card: ResearchCard,
    plan: CardPlan,
    cutoff: date,
    criteria: Mapping[str, float],
    research_dir: Path,
) -> None:
    """An existing card is kept as written, so the mapping must agree with it."""
    if card.holdout_cutoff != cutoff or card.criteria_hash != criteria_hash(criteria):
        raise BackfillError(
            f"{plan.id} exists with a different cutoff or criteria; "
            "legacy cards are frozen once written"
        )
    # load_card found exactly one file for the id, so this path exists only if the slug agrees.
    slug_kept = (research_dir / "strategies" / f"{plan.id}-{plan.slug}.md").exists()
    agreement = {
        "title": card.title == plan.title,
        "slug": slug_kept,
        "status": card.status is plan.status,
        "tags": card.tags == plan.tags,
        "body": card.body == plan.body,
    }
    changed = [name for name, agrees in agreement.items() if not agrees]
    if changed:
        raise BackfillError(
            f"{plan.id}: the mapping changes the {', '.join(changed)} of the existing card; "
            "card files, not the mapping, hold these after the first apply"
        )


def _cutoff(plan: CardPlan, runs_by_id: Mapping[str, LegacyRun], today: date) -> date:
    """The day after the card's last run or study.

    ``end_date`` is exclusive, so the last day a legacy run saw stays out of the holdout.
    """
    dates = {run_id: runs_by_id[run_id].run_date for run_id in plan.runs}
    dates.update({_study_run_id(plan.id, study.variant): study.date for study in plan.studies})
    future = sorted(name for name, day in dates.items() if day > today)
    if future:
        raise BackfillError(f"{plan.id}: dated after today ({today}): {', '.join(future)}")
    return max(dates.values()) + timedelta(days=1)


def _new_card(
    plan: CardPlan, cutoff: date, criteria: Mapping[str, float], today: date
) -> ResearchCard:
    """Build a legacy card and verify that it can be rendered before writing."""
    card = ResearchCard(
        id=plan.id,
        title=plan.title,
        status=plan.status,
        tags=plan.tags,
        created=today,
        holdout_cutoff=cutoff,
        holdout_contaminated=False,
        success_criteria=criteria,
        registered_criteria_hash=criteria_hash(criteria),
        criteria_amendments=(),
        body=plan.body,
    )
    try:
        render_card(card)
    except CardError as exc:
        # render_card's errors name the card already.
        raise BackfillError(str(exc)) from exc
    return card


def _study_run_id(card_id: str, variant: str) -> str:
    """Identify a study trial by its card and variant for repeatable imports."""
    return f"{STUDY_PREFIX}{card_id}:{variant}"


def _storable(trial: Trial) -> Trial:
    """*trial*, refused here if :func:`append_trial` could not store it."""
    try:
        # The id is a stand-in: append_trial assigns the real one.
        readable_line(replace(trial, trial_id="T-000001"))
    except ValueError as exc:
        raise BackfillError(f"{trial.strategy}: {trial.run_id} cannot be stored: {exc}") from exc
    return trial


def _legacy_trial(card: ResearchCard, run: LegacyRun) -> Trial:
    """Build a trial from a scanned run, preserving provenance and resolution notes."""
    note = f"legacy run: {run.name}"
    if run.unresolved:
        note += f" ({run.unresolved})"
    if run.presets_from_workspace:
        note += " (presets resolved from the workspace)"
    return Trial(
        strategy=card.id,
        run_id=run.run_id,
        kind=run.kind,
        status=run.status,
        ts=run.ts,
        fingerprint=run.fingerprint,
        fingerprint_version=FINGERPRINT_VERSION,
        criteria_hash=card.criteria_hash,
        holdout_cutoff=card.holdout_cutoff,
        code=run.code,
        window=run.window,
        note=note.replace("\n", " "),
        metrics=run.metrics,
    )


def _study_trial(card: ResearchCard, study: StudyTrial, run_id: str) -> Trial:
    """Build a legacy trial with a synthetic fingerprint for a documented study."""
    return Trial(
        strategy=card.id,
        run_id=run_id,
        kind=TrialKind.LEGACY,
        status=TrialStatus.OK,
        ts=study.date.isoformat(),
        fingerprint=digest({"study": card.id, "variant": study.variant, "symbols": study.symbols}),
        fingerprint_version=FINGERPRINT_VERSION,
        criteria_hash=card.criteria_hash,
        holdout_cutoff=card.holdout_cutoff,
        code={},
        window={"start": None, "end": None},
        note=study.note,
        metrics={symbol: dict(values) for symbol, values in study.metrics.items()},
    )
