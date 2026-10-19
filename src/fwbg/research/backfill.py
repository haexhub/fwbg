"""Backfill legacy runs and documented studies into the research register.

``draft_mapping`` proposes one card per strategy name; the operator regroups
the cards, adds verdict prose and study trials, and ``apply_mapping`` writes
cards, legacy trials and stored configs. The backfill is one-shot: it runs on
a fresh register before the step-3 gate goes live. Applying the same mapping
again only adds what is missing.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from fwbg.research.card import (
    CARD_ID_RE,
    SLUG_RE,
    CardError,
    ResearchCard,
    Status,
    UniqueKeyLoader,
    criteria_hash,
    load_card,
    render_card,
    slugify,
    write_card,
)
from fwbg.research.fingerprint import FINGERPRINT_VERSION, digest
from fwbg.research.ledger import (
    LEDGER_FILENAME,
    LedgerError,
    Metrics,
    Trial,
    TrialKind,
    TrialStatus,
    append_trial,
    read_trials,
)
from fwbg.research.legacy import LegacyRun
from fwbg.research.repo import commit_register, register_lock, store_config

COMMIT_MESSAGE = "chore: backfill legacy research"
_LEGACY_STATUSES = (Status.PARKED, Status.REJECTED)
_CARD_KEYS = frozenset(
    {"id", "title", "slug", "status", "tags", "criteria", "body", "runs", "studies"}
)
_STUDY_KEYS = frozenset({"variant", "symbols", "date", "note", "metrics"})


class BackfillError(Exception):
    """The mapping is invalid or does not match the scanned runs or the register."""


@dataclass(frozen=True)
class StudyTrial:
    """One documented evaluation of a study that ran outside the run system."""

    variant: str
    symbols: tuple[str, ...]
    date: date
    note: str
    metrics: Metrics


@dataclass(frozen=True)
class CardPlan:
    """One card of the mapping, validated."""

    id: str
    title: str
    slug: str
    status: Status
    tags: tuple[str, ...]
    criteria: Mapping[str, float] | None
    body: str
    runs: tuple[str, ...]
    studies: tuple[StudyTrial, ...]


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


def draft_mapping(runs: Sequence[LegacyRun]) -> dict[str, Any]:
    """Propose one parked card per strategy name, in order of first run.

    Ids start at S-0001: the draft is meant for a fresh register.
    """
    by_name: dict[str, list[LegacyRun]] = {}
    for run in sorted(runs, key=lambda r: (r.run_date, r.run_id)):
        by_name.setdefault(run.name, []).append(run)
    cards = []
    for number, (name, group) in enumerate(by_name.items(), start=1):
        cards.append(
            {
                "id": f"S-{number:04d}",
                "title": name,
                "slug": slugify(name),
                "status": Status.PARKED.value,
                "tags": [],
                "body": "## Hypothesis\n\n## Verdict\n\nNo conclusion documented.\n",
                "runs": [run.run_id for run in group],
            }
        )
    return {"cards": cards}


def load_mapping(text: str) -> list[CardPlan]:
    """Parse a mapping file's YAML (a repeated key is an error) into validated card plans."""
    try:
        data = yaml.load(text, Loader=UniqueKeyLoader)
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise BackfillError(f"mapping is not valid YAML{where}: {exc.problem}") from exc
    except (yaml.YAMLError, ValueError) as exc:
        # PyYAML raises a plain ValueError for impossible dates such as 2026-02-30.
        raise BackfillError(f"mapping is not valid YAML: {exc}") from exc
    return parse_mapping(data)


def parse_mapping(data: object) -> list[CardPlan]:
    """Validate a mapping document; raise :class:`BackfillError` on any defect."""
    if not isinstance(data, Mapping) or not isinstance(data.get("cards"), list):
        raise BackfillError("mapping must be a mapping with a 'cards' list")
    plans = [_card_plan(item, index) for index, item in enumerate(data["cards"], start=1)]
    duplicates = _repeated(plan.id for plan in plans)
    if duplicates:
        raise BackfillError(f"duplicate card ids: {', '.join(duplicates)}")
    return plans


def apply_mapping(
    plans: Sequence[CardPlan],
    runs: Sequence[LegacyRun],
    research_dir: Path,
    default_criteria: Mapping[str, float],
    today: date,
) -> BackfillResult:
    """Write the mapped cards, their legacy and study trials and the runs' configs.

    Every scanned run must be mapped exactly once, and no run or study may be
    dated after *today*. A card's criteria are *default_criteria* with the
    card's own criteria applied on top. Every card and trial is built and
    checked before the register is touched, so a defect raises
    :class:`BackfillError` without writing anything. Cards that already exist
    are kept and must agree with the mapping; trials already in the ledger
    are skipped.

    If a write fails part-way (a full disk, a failing commit), apply the same
    mapping again without editing it: the recorded trials are skipped and the
    rest is written and committed.
    """
    template = _criteria(default_criteria, "criteria template")
    runs_by_id = {run.run_id: run for run in runs}
    _check_coverage(plans, runs_by_id)
    _prepare(plans, runs_by_id, research_dir, template, today)
    with register_lock(research_dir):
        # Again under the lock: another writer may have changed the register meanwhile.
        writes = _prepare(plans, runs_by_id, research_dir, template, today)
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


def _card_plan(item: object, index: int) -> CardPlan:
    where = f"cards[{index}]"
    if not isinstance(item, Mapping):
        raise BackfillError(f"{where}: expected a mapping")
    unknown = sorted(str(key) for key in item.keys() - _CARD_KEYS)
    if unknown:
        raise BackfillError(f"{where}: unknown keys: {', '.join(unknown)}")
    card_id = item.get("id")
    if not isinstance(card_id, str) or not CARD_ID_RE.fullmatch(card_id):
        raise BackfillError(f"{where}: invalid id {card_id!r}")
    where = card_id
    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        raise BackfillError(f"{where}: title must be non-empty text")
    slug = item.get("slug") or slugify(title)
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
        raise BackfillError(f"{where}: invalid slug {slug!r}")
    try:
        status = Status(item.get("status", Status.PARKED.value))
    except ValueError as exc:
        raise BackfillError(f"{where}: {exc}") from exc
    if status not in _LEGACY_STATUSES:
        raise BackfillError(f"{where}: legacy cards are parked or rejected, not {status.value}")
    runs = _text_list(item.get("runs"), f"{where}.runs")
    studies = tuple(
        _study(study, f"{where}.studies[{number}]")
        for number, study in enumerate(item.get("studies") or [], start=1)
    )
    if not runs and not studies:
        raise BackfillError(f"{where}: a card needs at least one run or study trial")
    body = item.get("body", "")
    if not isinstance(body, str):
        raise BackfillError(f"{where}: body must be text")
    criteria = item.get("criteria")
    return CardPlan(
        id=card_id,
        title=title,
        slug=slug,
        status=status,
        tags=_text_list(item.get("tags"), f"{where}.tags"),
        criteria=None if criteria is None else _criteria(criteria, f"{where}.criteria"),
        body=body,
        runs=runs,
        studies=studies,
    )


def _study(item: object, where: str) -> StudyTrial:
    if not isinstance(item, Mapping):
        raise BackfillError(f"{where}: expected a mapping")
    unknown = sorted(str(key) for key in item.keys() - _STUDY_KEYS)
    if unknown:
        raise BackfillError(f"{where}: unknown keys: {', '.join(unknown)}")
    variant = item.get("variant")
    note = item.get("note")
    when = item.get("date")
    if not isinstance(variant, str) or not variant.strip():
        raise BackfillError(f"{where}.variant: expected non-empty text, got {variant!r}")
    # The gate's rule for run notes: non-blank, no line break of any kind.
    if not isinstance(note, str) or not note.strip() or note.splitlines() != [note]:
        raise BackfillError(f"{where}.note: must be one line of text, got {note!r}")
    # A datetime is a date too, but it cannot be compared with one.
    if type(when) is not date:
        raise BackfillError(f"{where}.date: expected a YYYY-MM-DD date, got {when!r}")
    symbols = tuple(s.upper() for s in _text_list(item.get("symbols"), f"{where}.symbols"))
    if not symbols:
        raise BackfillError(f"{where}: a study trial names at least one symbol")
    return StudyTrial(variant, symbols, when, note, _study_metrics(item.get("metrics"), where))


def _study_metrics(value: object, where: str) -> dict[str, dict[str, float | int | None]]:
    """Symbol -> metric -> number or null, as the ledger stores it."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BackfillError(f"{where}: metrics must be a mapping")
    metrics: dict[str, dict[str, float | int | None]] = {}
    for symbol, values in value.items():
        if not isinstance(symbol, str) or not isinstance(values, Mapping):
            raise BackfillError(f"{where}: metrics.{symbol} must map names to numbers")
        if symbol.upper() in metrics:
            raise BackfillError(f"{where}: metrics name {symbol} twice, ignoring case")
        checked: dict[str, float | int | None] = {}
        for name, number in values.items():
            if not isinstance(name, str) or not _ledger_number(number):
                raise BackfillError(f"{where}: metrics.{symbol}.{name} must be a finite number")
            checked[name] = number
        metrics[symbol.upper()] = checked
    return metrics


def _ledger_number(value: object) -> bool:
    """Whether the ledger stores *value* as a metric: null, an integer or a finite float."""
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _text_list(value: object, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise BackfillError(f"{where}: expected a list of non-empty text, got {value!r}")
    return tuple(value)


def _criteria(value: object, where: str) -> dict[str, float]:
    """Thresholds by name, under the rules of a card's ``success_criteria``."""
    if not isinstance(value, Mapping) or not value:
        raise BackfillError(f"{where}: criteria must be a non-empty mapping")
    criteria: dict[str, float] = {}
    for key, threshold in value.items():
        if not isinstance(key, str):
            raise BackfillError(f"{where}: invalid criterion {key!r}")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise BackfillError(f"{where}: criterion {key} must be a number, got {threshold!r}")
        try:
            number = float(threshold)
        except OverflowError as exc:
            raise BackfillError(f"{where}: criterion {key} is out of range") from exc
        if not math.isfinite(number):
            raise BackfillError(
                f"{where}: criterion {key} must be a finite number, got {threshold!r}"
            )
        criteria[key] = number
    return criteria


def _repeated(values: Iterable[str]) -> list[str]:
    """The values that occur more than once, sorted."""
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _check_coverage(plans: Sequence[CardPlan], runs_by_id: Mapping[str, LegacyRun]) -> None:
    mapped = [run_id for plan in plans for run_id in plan.runs]
    unknown = sorted(set(mapped) - runs_by_id.keys())
    twice = _repeated(mapped)
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
    runs_by_id: Mapping[str, LegacyRun],
    research_dir: Path,
    template: Mapping[str, float],
    today: date,
) -> _Writes:
    """Build and check every card, config and trial of the apply; write nothing."""
    recorded = {trial.run_id for trial in read_trials(research_dir / LEDGER_FILENAME)}
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
        elif card.holdout_cutoff != cutoff or card.criteria_hash != criteria_hash(criteria):
            raise BackfillError(
                f"{plan.id} exists with a different cutoff or criteria; "
                "legacy cards are frozen once written"
            )
        for run_id in plan.runs:
            if run_id in recorded:
                continue
            run = runs_by_id[run_id]
            if run.config is not None:
                configs[run.fingerprint] = run.config
            trials.append(_storable(_legacy_trial(card, run)))
        for study in plan.studies:
            study_run_id = _study_run_id(card.id, study.variant)
            if study_run_id not in recorded:
                trials.append(_storable(_study_trial(card, study, study_run_id)))
    return _Writes(tuple(cards), configs, tuple(trials))


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
        raise BackfillError(f"{plan.id}: {exc}") from exc
    return card


def _study_run_id(card_id: str, variant: str) -> str:
    return f"study:{card_id}:{variant}"


def _storable(trial: Trial) -> Trial:
    """*trial*, refused here if the ledger could not store it as one line that reads back."""
    numbered = replace(trial, trial_id="T-000001")
    try:
        stored = Trial.from_json(numbered.to_json())
    except (TypeError, ValueError, LedgerError) as exc:
        raise BackfillError(f"{trial.strategy}: {trial.run_id} cannot be stored: {exc}") from exc
    if stored != numbered:
        raise BackfillError(f"{trial.strategy}: {trial.run_id} would not read back unchanged")
    return trial


def _legacy_trial(card: ResearchCard, run: LegacyRun) -> Trial:
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
