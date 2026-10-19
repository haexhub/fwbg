"""Research cards: one Markdown file with YAML front matter per research strategy."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path

import yaml

from fwbg.research.fingerprint import digest

CARD_ID_RE = re.compile(r"S-\d{4}")
_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---[ \t]*\n?(.*)\Z", re.DOTALL)


class CardError(Exception):
    """A research card is malformed, ambiguous or does not match its file."""


class Status(str, Enum):
    ACTIVE = "active"
    PARKED = "parked"
    REJECTED = "rejected"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class CriteriaAmendment:
    date: date
    reason: str
    criteria_hash: str


@dataclass(frozen=True)
class ResearchCard:
    id: str
    title: str
    status: Status
    tags: tuple[str, ...]
    created: date
    holdout_cutoff: date
    holdout_contaminated: bool
    success_criteria: Mapping[str, float]
    registered_criteria_hash: str
    criteria_amendments: tuple[CriteriaAmendment, ...]
    body: str

    @property
    def criteria_hash(self) -> str:
        return criteria_hash(self.success_criteria)


def criteria_hash(criteria: Mapping[str, float]) -> str:
    """Hash of a set of success criteria, independent of key order."""
    return digest(dict(criteria))


def parse_card(text: str) -> ResearchCard:
    """Parse a card file; raise :class:`CardError` on any defect."""
    match = _FRONT_MATTER_RE.match(text)
    if match is None:
        raise CardError("card has no YAML front matter")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise CardError(f"invalid YAML front matter: {exc}") from exc
    if not isinstance(meta, Mapping):
        raise CardError("front matter must be a mapping")
    try:
        return ResearchCard(
            id=_card_id(meta["id"]),
            title=str(meta["title"]),
            status=Status(meta["status"]),
            tags=_tags(meta.get("tags")),
            created=_as_date(meta["created"], "created"),
            holdout_cutoff=_as_date(meta["holdout_cutoff"], "holdout_cutoff"),
            holdout_contaminated=bool(meta.get("holdout_contaminated", False)),
            success_criteria=_criteria(meta["success_criteria"]),
            registered_criteria_hash=str(meta["registered_criteria_hash"]),
            criteria_amendments=tuple(
                _amendment(item) for item in meta.get("criteria_amendments") or ()
            ),
            body=match.group(2),
        )
    except KeyError as exc:
        raise CardError(f"missing field: {exc.args[0]}") from exc
    except ValueError as exc:
        raise CardError(str(exc)) from exc


def load_card(research_dir: Path, card_id: str) -> ResearchCard | None:
    """Load card *card_id* from ``strategies/``; ``None`` if no file exists for it."""
    _card_id(card_id)
    folder = research_dir / "strategies"
    paths = sorted({*folder.glob(f"{card_id}-*.md"), *folder.glob(f"{card_id}.md")})
    if not paths:
        return None
    if len(paths) > 1:
        raise CardError(f"ambiguous research id {card_id}: {[p.name for p in paths]}")
    card = parse_card(paths[0].read_text(encoding="utf-8"))
    if card.id != card_id:
        raise CardError(f"{paths[0].name} declares id {card.id}, expected {card_id}")
    return card


def _card_id(value: object) -> str:
    if not isinstance(value, str) or not CARD_ID_RE.fullmatch(value):
        raise CardError(f"invalid research id {value!r} (expected S-NNNN)")
    return value


def _tags(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CardError(f"tags must be a list, got {value!r}")
    return tuple(str(tag) for tag in value)


def _as_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise CardError(f"{field_name}: not an ISO date: {value!r}") from exc
    raise CardError(f"{field_name}: expected a date, got {value!r}")


def _criteria(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise CardError("success_criteria must be a non-empty mapping")
    criteria: dict[str, float] = {}
    for key, threshold in value.items():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise CardError(f"success_criteria.{key}: expected a number, got {threshold!r}")
        criteria[str(key)] = float(threshold)
    return criteria


def _amendment(value: object) -> CriteriaAmendment:
    if not isinstance(value, Mapping):
        raise CardError(f"criteria_amendments: expected a mapping, got {value!r}")
    try:
        return CriteriaAmendment(
            date=_as_date(value["date"], "criteria_amendments.date"),
            reason=str(value["reason"]),
            criteria_hash=str(value["criteria_hash"]),
        )
    except KeyError as exc:
        raise CardError(f"criteria_amendments: missing {exc.args[0]}") from exc
