"""Research cards: one Markdown file with YAML front matter per research strategy."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from fwbg.research.fingerprint import DIGEST_RE, digest

CARD_ID_RE = re.compile(r"S-[0-9]{4}")
_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n?(.*)\Z", re.DOTALL)
_FIELDS = frozenset(
    {
        "id",
        "title",
        "status",
        "tags",
        "created",
        "holdout_cutoff",
        "holdout_contaminated",
        "success_criteria",
        "registered_criteria_hash",
        "criteria_amendments",
    }
)
_AMENDMENT_FIELDS = frozenset({"date", "reason", "criteria_hash"})


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
        meta = yaml.load(match.group(1), Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, ValueError) as exc:
        # PyYAML raises a plain ValueError for impossible dates such as 2025-04-31.
        raise CardError(f"invalid YAML front matter: {exc}") from exc
    if not isinstance(meta, Mapping):
        raise CardError("front matter must be a mapping")
    unknown = sorted(str(key) for key in meta.keys() - _FIELDS)
    if unknown:
        raise CardError(f"unknown front matter keys: {', '.join(unknown)}")
    try:
        return ResearchCard(
            id=_card_id(meta["id"]),
            title=_text(meta["title"], "title"),
            status=_status(meta["status"]),
            tags=_tags(meta.get("tags")),
            created=_as_date(meta["created"], "created"),
            holdout_cutoff=_as_date(meta["holdout_cutoff"], "holdout_cutoff"),
            holdout_contaminated=_flag(meta["holdout_contaminated"], "holdout_contaminated"),
            success_criteria=_criteria(meta["success_criteria"]),
            registered_criteria_hash=_hash(
                meta["registered_criteria_hash"], "registered_criteria_hash"
            ),
            criteria_amendments=_amendments(meta.get("criteria_amendments")),
            body=match.group(2),
        )
    except KeyError as exc:
        raise CardError(f"missing field: {exc.args[0]}") from exc


def load_card(research_dir: Path, card_id: str) -> ResearchCard | None:
    """Load card *card_id* from ``strategies/``; ``None`` if no file exists for it."""
    _card_id(card_id)
    folder = research_dir / "strategies"
    paths = sorted({*folder.glob(f"{card_id}-*.md"), *folder.glob(f"{card_id}.md")})
    if not paths:
        return None
    if len(paths) > 1:
        raise CardError(f"ambiguous research id {card_id}: {[p.name for p in paths]}")
    try:
        card = parse_card(paths[0].read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise CardError(
            f"{paths[0].name}: not valid UTF-8 ({exc.reason} at byte {exc.start})"
        ) from exc
    except CardError as exc:
        raise CardError(f"{paths[0].name}: {exc}") from exc
    if card.id != card_id:
        raise CardError(f"{paths[0].name} declares id {card.id}, expected {card_id}")
    return card


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe loader that refuses duplicate mapping keys instead of keeping the last one."""

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        mapping: dict[Any, Any] = super().construct_mapping(node, deep=deep)
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return mapping


def _card_id(value: object) -> str:
    if not isinstance(value, str) or not CARD_ID_RE.fullmatch(value):
        raise CardError(f"invalid research id {value!r} (expected S-NNNN)")
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CardError(f"{field_name}: expected non-empty text, got {value!r}")
    return value


def _hash(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not DIGEST_RE.fullmatch(value):
        raise CardError(f"{field_name}: expected sha256:<64 hex digits>, got {value!r}")
    return value


def _flag(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise CardError(f"{field_name}: expected true or false, got {value!r}")
    return value


def _status(value: object) -> Status:
    try:
        return Status(value)
    except ValueError as exc:
        valid = ", ".join(status.value for status in Status)
        raise CardError(f"status: invalid value {value!r} (expected one of {valid})") from exc


def _tags(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(tag, str) for tag in value):
        raise CardError(f"tags must be a list of strings, got {value!r}")
    return tuple(value)


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


def _criteria(value: object) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise CardError("success_criteria must be a non-empty mapping")
    criteria: dict[str, float] = {}
    for key, threshold in value.items():
        if not isinstance(key, str):
            raise CardError(f"success_criteria: keys must be text, got {key!r}")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise CardError(f"success_criteria.{key}: expected a number, got {threshold!r}")
        try:
            number = float(threshold)
        except OverflowError as exc:
            raise CardError(f"success_criteria.{key}: number out of range") from exc
        if not math.isfinite(number):
            raise CardError(f"success_criteria.{key}: expected a finite number, got {threshold!r}")
        criteria[key] = number
    return MappingProxyType(criteria)


def _amendments(value: object) -> tuple[CriteriaAmendment, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CardError(f"criteria_amendments must be a list, got {value!r}")
    return tuple(_amendment(item) for item in value)


def _amendment(value: object) -> CriteriaAmendment:
    if not isinstance(value, Mapping):
        raise CardError(f"criteria_amendments: expected a mapping, got {value!r}")
    unknown = sorted(str(key) for key in value.keys() - _AMENDMENT_FIELDS)
    if unknown:
        raise CardError(f"criteria_amendments: unknown keys: {', '.join(unknown)}")
    try:
        return CriteriaAmendment(
            date=_as_date(value["date"], "criteria_amendments.date"),
            reason=_text(value["reason"], "criteria_amendments.reason"),
            criteria_hash=_hash(value["criteria_hash"], "criteria_amendments.criteria_hash"),
        )
    except KeyError as exc:
        raise CardError(f"criteria_amendments: missing {exc.args[0]}") from exc
