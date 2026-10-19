"""The backfill mapping: which legacy runs and studies belong to which research card.

``draft_mapping`` proposes one card per strategy name; the operator regroups
the cards and adds verdict prose and study trials. ``load_mapping`` and
``parse_mapping`` validate the reviewed mapping into :class:`CardPlan` objects
for :func:`fwbg.research.backfill.apply_mapping`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import yaml

from fwbg.research.card import (
    CARD_ID_RE,
    SLUG_RE,
    CardError,
    Status,
    UniqueKeyLoader,
    slugify,
)
from fwbg.research.ledger import Metrics
from fwbg.research.legacy import LegacyRun

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
    slug = item.get("slug")
    if not slug:
        try:
            slug = slugify(title)
        except CardError as exc:
            raise BackfillError(f"{where}: {exc}") from exc
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
    variants = _repeated(study.variant for study in studies)
    if variants:
        raise BackfillError(f"{where}: duplicate study variants: {', '.join(variants)}")
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
        criteria=None if criteria is None else parse_criteria(criteria, f"{where}.criteria"),
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


def parse_criteria(value: object, where: str) -> dict[str, float]:
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
