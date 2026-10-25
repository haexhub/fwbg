"""Research cards: one Markdown file with YAML front matter per research strategy."""

from __future__ import annotations

import contextlib
import math
import os
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

from fwbg.research.fingerprint import DIGEST_RE, digest

if TYPE_CHECKING:
    # ledger imports CARD_ID_RE from here.
    from fwbg.research.ledger import Trial

CARD_ID_RE = re.compile(r"S-[0-9]{4}")
CARD_FILE_RE = re.compile(r"(S-[0-9]{4})(?:-.*)?\.md")
CRITERIA_NAMES = frozenset(
    {"min_sharpe", "min_trades", "max_drawdown", "min_fold_stability", "max_p_value"}
)
# At most 60 characters, the length slugify cuts a title to.
SLUG_RE = re.compile(r"(?=.{1,60}\Z)[a-z0-9]+(?:-[a-z0-9]+)*")
_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n?(.*)\Z", re.DOTALL)
_HEADING_RE = re.compile(r"^## (.+?)[ \t]*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})", re.MULTILINE)
_FENCE_CLOSE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*$")
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
        meta = yaml.load(match.group(1), Loader=UniqueKeyLoader)
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
            success_criteria=validate_criteria(meta["success_criteria"]),
            registered_criteria_hash=_hash(
                meta["registered_criteria_hash"], "registered_criteria_hash"
            ),
            criteria_amendments=_amendments(meta.get("criteria_amendments")),
            body=match.group(2),
        )
    except KeyError as exc:
        raise CardError(f"missing field: {exc.args[0]}") from exc


def card_path(research_dir: Path, card_id: str) -> Path | None:
    """The file of card *card_id*; ``None`` if there is none."""
    _card_id(card_id)
    folder = research_dir / "strategies"
    paths = sorted({*folder.glob(f"{card_id}-*.md"), *folder.glob(f"{card_id}.md")})
    if len(paths) > 1:
        raise CardError(f"ambiguous research id {card_id}: {[p.name for p in paths]}")
    return paths[0] if paths else None


def load_card(research_dir: Path, card_id: str) -> ResearchCard | None:
    """Load card *card_id* from ``strategies/``; ``None`` if no file exists for it."""
    path = card_path(research_dir, card_id)
    if path is None:
        return None
    try:
        card = parse_card(path.read_text(encoding="utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise CardError(f"{path.name}: not valid UTF-8 ({exc.reason} at byte {exc.start})") from exc
    except CardError as exc:
        raise CardError(f"{path.name}: {exc}") from exc
    if card.id != card_id:
        raise CardError(f"{path.name} declares id {card.id}, expected {card_id}")
    return card


class UniqueKeyLoader(yaml.SafeLoader):
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


def validate_criteria(value: object) -> Mapping[str, float]:
    """Success criteria as a read-only mapping; raise :class:`CardError` on any defect."""
    if not isinstance(value, Mapping) or not value:
        raise CardError("success_criteria must be a non-empty mapping")
    criteria: dict[str, float] = {}
    for key, threshold in value.items():
        if not isinstance(key, str):
            raise CardError(f"success_criteria: keys must be text, got {key!r}")
        if key not in CRITERIA_NAMES:
            raise CardError(
                f"success_criteria: unknown criterion {key!r} "
                f"(known: {', '.join(sorted(CRITERIA_NAMES))})"
            )
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


def render_card(card: ResearchCard) -> str:
    """Serialise *card* so that :func:`parse_card` reads it back unchanged."""
    # Reading a card file translates newlines, so a carriage return would not survive.
    if "\r" in card.body:
        raise CardError(f"{card.id}: body contains a carriage return")
    meta: dict[str, Any] = {
        "id": card.id,
        "title": card.title,
        "status": card.status.value,
        "tags": list(card.tags),
        "created": card.created,
        "holdout_cutoff": card.holdout_cutoff,
        "holdout_contaminated": card.holdout_contaminated,
        "success_criteria": dict(card.success_criteria),
        "registered_criteria_hash": card.registered_criteria_hash,
    }
    if card.criteria_amendments:
        meta["criteria_amendments"] = [
            {"date": item.date, "reason": item.reason, "criteria_hash": item.criteria_hash}
            for item in card.criteria_amendments
        ]
    front = yaml.safe_dump(
        meta, sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000
    )
    text = f"---\n{front}---\n{card.body}"
    try:
        read_back = parse_card(text)
    except CardError as exc:
        raise CardError(f"{card.id} would not read back: {exc}") from exc
    if read_back != card:
        raise CardError(f"{card.id} would not read back unchanged")
    return text


def slugify(title: str) -> str:
    """File-name slug for a card title: lowercase ASCII words joined by hyphens."""
    words = re.findall(r"[a-z0-9]+", title.lower())
    slug = "-".join(words)[:60].strip("-")
    if not slug:
        raise CardError(f"title {title!r} gives an empty slug")
    return slug


def write_card(research_dir: Path, card: ResearchCard, slug: str) -> Path:
    """Write *card* to ``strategies/<id>-<slug>.md``; an existing card is never overwritten.

    The register must be initialised (``strategies/`` exists); filesystem errors
    (``OSError``) pass through. Callers that may run concurrently must hold
    :func:`fwbg.research.repo.register_lock`: the exclusive create only stops a
    second write of the same file, not the same id under another slug.

    Raises:
        CardError: the slug is invalid, the card cannot be rendered, or a card
            with this id already exists.
    """
    if not SLUG_RE.fullmatch(slug):
        raise CardError(f"invalid slug {slug!r}")
    if load_card(research_dir, card.id) is not None:
        raise CardError(f"{card.id} already exists; cards are never overwritten")
    text = render_card(card)
    path = research_dir / "strategies" / f"{card.id}-{slug}.md"
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise CardError(f"{card.id} already exists; cards are never overwritten") from exc
    try:
        with handle:
            handle.write(text)
    except OSError:
        # A truncated card would make every later load of this id fail.
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        raise
    return path


def replace_card(research_dir: Path, card: ResearchCard) -> Path:
    """Rewrite the existing file of ``card.id`` atomically; call under ``register_lock``."""
    path = card_path(research_dir, card.id)
    if path is None:
        raise CardError(f"no research card {card.id}")
    text = render_card(card)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def next_card_id(research_dir: Path, trials: Iterable[Trial]) -> str:
    """The id after the highest card file or ledger card; call under ``register_lock``."""
    numbers = [
        int(match.group(1)[2:])
        for path in (research_dir / "strategies").glob("S-*.md")
        if (match := CARD_FILE_RE.fullmatch(path.name))
    ]
    numbers += [int(trial.strategy[2:]) for trial in trials]
    number = max(numbers, default=0) + 1
    if number > 9999:
        raise CardError("research ids are exhausted (S-9999)")
    return f"S-{number:04d}"


def _headings(body: str) -> list[re.Match[str]]:
    """The ``## `` heading matches of *body*, leaving out those inside code fences."""
    fences: list[tuple[int, int]] = []
    opener: tuple[str, int, int] | None = None
    offset = 0
    for line in body.splitlines(keepends=True):
        if opener is None:
            match = _FENCE_RE.match(line)
            if match is not None:
                marker = match.group(1)
                opener = (marker[0], len(marker), offset)
        else:
            match = _FENCE_CLOSE_RE.match(line)
            if match is not None:
                marker = match.group(1)
                if marker[0] == opener[0] and len(marker) >= opener[1]:
                    fences.append((opener[2], offset + len(line)))
                    opener = None
        offset += len(line)
    if opener is not None:
        fences.append((opener[2], len(body)))
    return [
        m
        for m in _HEADING_RE.finditer(body)
        if not any(start <= m.start() < end for start, end in fences)
    ]


def section_text(body: str, heading: str) -> str | None:
    """The text under ``## <heading>`` in a card body, stripped; ``None`` if absent."""
    heads = _headings(body)
    own = next((m for m in heads if m.group(1) == heading), None)
    if own is None:
        return None
    end = next((m.start() for m in heads if m.start() > own.start()), len(body))
    return body[own.end() : end].strip()


def add_section_entry(body: str, heading: str, entry: str, before: Sequence[str] = ()) -> str:
    """Append *entry* to the ``## <heading>`` section of a card body.

    A missing section is created before the first *before* section the body
    has, else at the end. Text outside the section stays as it was; ``## ``
    lines inside code fences are not headings.

    Raises:
        CardError: the section appears more than once.
    """
    entry = entry.strip("\n")
    heads = _headings(body)
    owns = [m for m in heads if m.group(1) == heading]
    if len(owns) > 1:
        raise CardError(f"section {heading!r} appears {len(owns)} times; fix the card by hand")
    if owns:
        own = owns[0]
        end = next((m.start() for m in heads if m.start() > own.start()), len(body))
        section, tail = body[own.start() : end].rstrip("\n"), body[end:]
        return body[: own.start()] + section + "\n\n" + entry + "\n" + ("\n" + tail if tail else "")
    new = f"## {heading}\n\n{entry}\n"
    anchor = next((m.start() for m in heads if m.group(1) in before), None)
    if anchor is not None:
        return body[:anchor] + new + "\n" + body[anchor:]
    head = body.rstrip("\n")
    return (head + "\n\n" if head else "") + new
