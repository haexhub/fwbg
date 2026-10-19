"""Backfill legacy runs and documented studies into the research register.

``draft_mapping`` proposes one card per strategy name; the operator regroups
the cards, adds verdict prose and study trials, and ``apply_mapping`` writes
cards, legacy trials and stored configs. Applying again adds only what is new.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fwbg.research.card import (
    CARD_ID_RE,
    SLUG_RE,
    CardError,
    ResearchCard,
    Status,
    criteria_hash,
    load_card,
    slugify,
    write_card,
)
from fwbg.research.fingerprint import FINGERPRINT_VERSION, digest
from fwbg.research.legacy import LegacyRun
from fwbg.research.ledger import (
    LEDGER_FILENAME,
    Metrics,
    Trial,
    TrialKind,
    TrialStatus,
    append_trial,
    read_trials,
)
from fwbg.research.repo import commit_register, register_lock, store_config

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
    cards_written: int
    trials_appended: int
    committed: bool


def draft_mapping(runs: Sequence[LegacyRun]) -> dict[str, Any]:
    """Propose one parked card per strategy name, in order of first run."""
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


def parse_mapping(data: object) -> list[CardPlan]:
    """Validate a mapping document; raise :class:`BackfillError` on any defect."""
    if not isinstance(data, Mapping) or not isinstance(data.get("cards"), list):
        raise BackfillError("mapping must be a mapping with a 'cards' list")
    plans = [_card_plan(item, index) for index, item in enumerate(data["cards"], start=1)]
    ids = [plan.id for plan in plans]
    duplicates = sorted({card_id for card_id in ids if ids.count(card_id) > 1})
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
    """Write the mapped cards and their legacy trials into the register.

    Every scanned run must be mapped exactly once. Cards that already exist
    are kept and must agree with the mapping; runs and study variants already
    in the ledger are skipped, so applying again only adds what is new.
    """
    template = _criteria(default_criteria, "criteria template")
    if template is None:
        raise BackfillError("criteria template is empty")
    runs_by_id = {run.run_id: run for run in runs}
    _check_coverage(plans, runs_by_id)
    ledger = research_dir / LEDGER_FILENAME
    cards_written = 0
    trials_appended = 0
    with register_lock(research_dir):
        recorded = {trial.run_id for trial in read_trials(ledger)}
        for plan in plans:
            card, written = _ensure_card(plan, runs_by_id, research_dir, template, today)
            cards_written += written
            for run_id in plan.runs:
                if run_id in recorded:
                    continue
                run = runs_by_id[run_id]
                if run.config is not None:
                    store_config(research_dir, run.fingerprint, run.config)
                append_trial(ledger, _legacy_trial(card, run))
                trials_appended += 1
            for study in plan.studies:
                study_run_id = f"study:{card.id}:{study.variant}"
                if study_run_id in recorded:
                    continue
                append_trial(ledger, _study_trial(card, study, study_run_id))
                trials_appended += 1
        committed = commit_register(
            research_dir,
            ["strategies", "configs", LEDGER_FILENAME],
            f"chore: backfill legacy research ({cards_written} cards, {trials_appended} trials)",
        )
    return BackfillResult(cards_written, trials_appended, committed)


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
    return CardPlan(
        id=card_id,
        title=title,
        slug=slug,
        status=status,
        tags=_text_list(item.get("tags"), f"{where}.tags"),
        criteria=_criteria(item.get("criteria"), where),
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
        raise BackfillError(f"{where}: variant must be non-empty text")
    if not isinstance(note, str) or len(note.splitlines()) != 1:
        raise BackfillError(f"{where}: note must be one line of text")
    if not isinstance(when, date):
        raise BackfillError(f"{where}: date must be a YYYY-MM-DD date")
    symbols = tuple(symbol.upper() for symbol in _text_list(item.get("symbols"), where))
    if not symbols:
        raise BackfillError(f"{where}: a study trial names at least one symbol")
    return StudyTrial(variant, symbols, when, note, _study_metrics(item.get("metrics"), where))


def _study_metrics(value: object, where: str) -> dict[str, dict[str, float | int | None]]:
    """Symbol -> metric -> finite number or null, as the ledger stores it."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BackfillError(f"{where}: metrics must be a mapping")
    metrics: dict[str, dict[str, float | int | None]] = {}
    for symbol, values in value.items():
        if not isinstance(symbol, str) or not isinstance(values, Mapping):
            raise BackfillError(f"{where}: metrics.{symbol} must map names to numbers")
        checked: dict[str, float | int | None] = {}
        for name, number in values.items():
            valid = number is None or (
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                and math.isfinite(number)
            )
            if not isinstance(name, str) or not valid:
                raise BackfillError(f"{where}: metrics.{symbol}.{name} must be a finite number")
            checked[name] = number
        metrics[symbol.upper()] = checked
    return metrics


def _text_list(value: object, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise BackfillError(f"{where}: expected a list of text")
    return tuple(value)


def _criteria(value: object, where: str) -> Mapping[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise BackfillError(f"{where}: criteria must be a non-empty mapping")
    criteria: dict[str, float] = {}
    for key, threshold in value.items():
        if not isinstance(key, str) or isinstance(threshold, bool):
            raise BackfillError(f"{where}: invalid criterion {key!r}")
        if not isinstance(threshold, (int, float)):
            raise BackfillError(f"{where}: criterion {key} must be a number")
        criteria[key] = float(threshold)
    return criteria


def _check_coverage(plans: Sequence[CardPlan], runs_by_id: Mapping[str, LegacyRun]) -> None:
    mapped = [run_id for plan in plans for run_id in plan.runs]
    unknown = sorted(set(mapped) - runs_by_id.keys())
    twice = sorted({run_id for run_id in mapped if mapped.count(run_id) > 1})
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


def _ensure_card(
    plan: CardPlan,
    runs_by_id: Mapping[str, LegacyRun],
    research_dir: Path,
    default_criteria: Mapping[str, float],
    today: date,
) -> tuple[ResearchCard, int]:
    """The plan's card: the existing one if it agrees, else a newly written one."""
    criteria = dict(plan.criteria or default_criteria)
    dates = [runs_by_id[run_id].run_date for run_id in plan.runs]
    dates += [study.date for study in plan.studies]
    # end_date is exclusive, so the last day a legacy run saw stays out of the holdout.
    cutoff = max(dates) + timedelta(days=1)
    existing = load_card(research_dir, plan.id)
    if existing is not None:
        if existing.holdout_cutoff != cutoff or existing.criteria_hash != criteria_hash(criteria):
            raise BackfillError(
                f"{plan.id} exists with a different cutoff or criteria; "
                "legacy cards are frozen once written"
            )
        return existing, 0
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
        write_card(research_dir, card, plan.slug)
    except CardError as exc:
        raise BackfillError(str(exc)) from exc
    return card, 1


def _legacy_trial(card: ResearchCard, run: LegacyRun) -> Trial:
    note = f"legacy run: {run.name}"
    if run.unresolved:
        note += f" ({run.unresolved})"
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
