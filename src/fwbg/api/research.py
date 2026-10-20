"""Research register endpoints: list, show, register cards and change their status."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

from fwbg.api.workspace import get_research_dir
from fwbg.research.card import (
    CARD_FILE_RE,
    CARD_ID_RE,
    CardError,
    ResearchCard,
    Status,
    load_card,
    slugify,
    validate_criteria,
)
from fwbg.research.commands import (
    CommandError,
    CommandInputError,
    UncommittedError,
    new_card,
    reopen,
    write_verdict,
)
from fwbg.research.gate import criteria_denials
from fwbg.research.ledger import LEDGER_FILENAME, LedgerError, read_trials
from fwbg.research.pending import PendingError, live_runs, read_pending
from fwbg.research.repo import ResearchRepoError
from fwbg.research.views import card_data, card_summary, coverage_section, iter_cards

router = APIRouter(prefix="/research", tags=["research"])

CardId = Annotated[str, PathParam(pattern=f"^{CARD_ID_RE.pattern}$")]


class NewCard(BaseModel):
    """Request body for registering a card; hypothesis and mechanism are required here."""

    title: str = Field(min_length=1)
    tags: list[str]
    hypothesis: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    cutoff: date | None = None
    criteria: dict[str, float] | None = None


class StatusChange(BaseModel):
    """Request body for a verdict (parked, rejected, candidate) or a reopen (active)."""

    status: Status
    reason: str
    actor: str = Field(min_length=1)
    reopen_if: str | None = None


# ponytail: the GET endpoints read without the register lock, so a write running
# at the same moment can show a card as unreadable for that one request.


@router.get("")
def list_cards(status: Status | None = None, tag: str | None = None) -> dict[str, Any]:
    """Cards in id order; a card that does not load is listed with its error.

    If the pending markers cannot be read, ``running`` is ``null`` and
    ``pending_error`` says why.
    """
    research_dir = get_research_dir(create=False)
    with _reading():
        trials = read_trials(research_dir / LEDGER_FILENAME)
        try:
            running: Counter[str] | None = Counter(
                run.trial.strategy for run in live_runs(research_dir)
            )
            pending_error = None
        except PendingError as exc:
            running, pending_error = None, str(exc)
        entries: list[dict[str, Any]] = []
        for card_id, card in iter_cards(research_dir, status, tag):
            if isinstance(card, CardError):
                entries.append({"id": card_id, "error": str(card)})
                continue
            own = [trial for trial in trials if trial.strategy == card.id]
            entries.append(card_summary(card, own, None if running is None else running[card.id]))
    return {"cards": entries, "pending_error": pending_error}


@router.get("/{card_id}")
def show_card(card_id: CardId) -> dict[str, Any]:
    """A card with its trials, pending runs, criteria problems and config coverage."""
    research_dir = get_research_dir(create=False)
    with _reading():
        card = _require(research_dir, card_id)
        own = [t for t in read_trials(research_dir / LEDGER_FILENAME) if t.strategy == card.id]
        try:
            pending: list[str] | None = [
                run.run_id for run in read_pending(research_dir) if run.trial.strategy == card.id
            ]
            pending_error = None
        except PendingError as exc:
            pending, pending_error = None, str(exc)
        coverage = coverage_section(research_dir, own)
    return {
        "card": card_data(card),
        "trials": [json.loads(trial.to_json()) for trial in own],
        "pending": pending,
        "pending_error": pending_error,
        "criteria_problems": [denial.message for denial in criteria_denials(card, own)],
        "coverage": coverage,
    }


@router.post("", status_code=201)
def register_card(body: NewCard) -> dict[str, Any]:
    """Register a new active card and commit it.

    A card written but not committed is still registered: the response carries
    a ``warning`` instead of an error, so a caller does not register it twice.
    """
    research_dir = get_research_dir(create=False)
    # Checked here, so a defect in the caller's input is a 422, not the register's 409.
    try:
        slugify(body.title)
        if body.criteria:
            validate_criteria(body.criteria)
    except CardError as exc:
        raise HTTPException(422, str(exc)) from exc
    warning = None
    with _writing():
        try:
            path = new_card(
                research_dir,
                body.title,
                body.tags,
                date.today(),
                cutoff=body.cutoff,
                criteria=body.criteria,
                hypothesis=body.hypothesis,
                mechanism=body.mechanism,
            )
        except UncommittedError as exc:
            warning, card_id = str(exc), exc.card_id
        else:
            match = CARD_FILE_RE.fullmatch(path.name)
            if match is None:
                raise HTTPException(500, f"registered card has an unexpected file name {path.name}")
            card_id = match.group(1)
    with _reading():
        card = _require(research_dir, card_id)
    return _written(card, warning)


@router.post("/{card_id}/status")
def change_status(card_id: CardId, body: StatusChange) -> dict[str, Any]:
    """Reopen a card (status ``active``) or close it with a verdict.

    As with registering, an uncommitted change succeeds with a ``warning``.
    """
    research_dir = get_research_dir(create=False)
    if body.status is Status.ACTIVE and body.reopen_if is not None:
        raise HTTPException(422, "a reopen takes no reopen_if")
    if body.status is not Status.ACTIVE and body.reopen_if is None:
        raise HTTPException(422, "a verdict needs reopen_if")
    with _reading():
        _require(research_dir, card_id)
    warning = None
    with _writing():
        try:
            if body.reopen_if is None:
                reopen(research_dir, card_id, body.reason, date.today(), actor=body.actor)
            else:
                write_verdict(
                    research_dir,
                    card_id,
                    body.status,
                    body.reason,
                    body.reopen_if,
                    date.today(),
                    actor=body.actor,
                )
        except UncommittedError as exc:
            warning = str(exc)
    with _reading():
        card = _require(research_dir, card_id)
    return _written(card, warning)


def _written(card: ResearchCard, warning: str | None) -> dict[str, Any]:
    return {"card": card_data(card)} | ({} if warning is None else {"warning": warning})


def _require(research_dir: Path, card_id: str) -> ResearchCard:
    card = load_card(research_dir, card_id)
    if card is None:
        raise HTTPException(404, f"no research card {card_id}")
    return card


# ponytail: an OSError (disk full, permissions) is mapped by neither context
# manager and surfaces as a 500.


@contextmanager
def _reading() -> Iterator[None]:
    """Map read failures: a defect in the register's files is a 409 'unreadable'."""
    try:
        yield
    except CommandError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (CardError, LedgerError, PendingError) as exc:
        raise HTTPException(409, f"research register unreadable: {exc}") from exc


@contextmanager
def _writing() -> Iterator[None]:
    """Map write failures: bad input 422, refusals 409, unreadable files 409 'unreadable'."""
    try:
        yield
    except CommandInputError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (CommandError, ResearchRepoError, CardError) as exc:
        # A CardError here comes from writing: a card that exists, ids exhausted, a duplicate section.
        raise HTTPException(409, str(exc)) from exc
    except (LedgerError, PendingError) as exc:
        raise HTTPException(409, f"research register unreadable: {exc}") from exc
