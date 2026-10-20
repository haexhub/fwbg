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
    validate_criteria,
)
from fwbg.research.commands import CommandError, new_card, reopen, write_verdict
from fwbg.research.ledger import LEDGER_FILENAME, LedgerError, read_trials
from fwbg.research.pending import PendingError, live_runs, read_pending
from fwbg.research.repo import ResearchRepoError
from fwbg.research.views import card_data, card_ids, card_summary, coverage_section

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


@router.get("")
def list_cards(status: Status | None = None, tag: str | None = None) -> list[dict[str, Any]]:
    """Every card in id order; a card that does not load is listed with its error."""
    research_dir = get_research_dir(create=False)
    with _register_errors():
        trials = read_trials(research_dir / LEDGER_FILENAME)
        running = Counter(run.trial.strategy for run in live_runs(research_dir))
        entries: list[dict[str, Any]] = []
        for card_id in card_ids(research_dir):
            try:
                card = load_card(research_dir, card_id)
            except CardError as exc:
                # Listed under every filter: it might match, and it needs fixing.
                entries.append({"id": card_id, "error": str(exc)})
                continue
            if card is None:
                continue
            if (status is not None and card.status is not status) or (
                tag is not None and tag not in card.tags
            ):
                continue
            own = [trial for trial in trials if trial.strategy == card.id]
            entries.append(card_summary(card, own, running[card.id]))
    return entries


@router.get("/{card_id}")
def show_card(card_id: CardId) -> dict[str, Any]:
    """A card with its trials, its pending runs and the coverage of its configs."""
    research_dir = get_research_dir(create=False)
    with _register_errors():
        card = _require(research_dir, card_id)
        own = [t for t in read_trials(research_dir / LEDGER_FILENAME) if t.strategy == card.id]
        pending = [
            run.run_id for run in read_pending(research_dir) if run.trial.strategy == card.id
        ]
        coverage = coverage_section(research_dir, own)
    return {
        "card": card_data(card),
        "trials": [json.loads(trial.to_json()) for trial in own],
        "pending": pending,
        "coverage": coverage,
    }


@router.post("", status_code=201)
def register_card(body: NewCard) -> dict[str, Any]:
    """Register a new active card and commit it."""
    research_dir = get_research_dir(create=False)
    if body.criteria:
        # Checked here, so a defect in the caller's criteria is a 422, not the register's 409.
        try:
            validate_criteria(body.criteria)
        except CardError as exc:
            raise HTTPException(422, str(exc)) from exc
    with _register_errors():
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
        match = CARD_FILE_RE.fullmatch(path.name)
        assert match is not None
        card = _require(research_dir, match.group(1))
    return {"card": card_data(card)}


@router.post("/{card_id}/status")
def change_status(card_id: CardId, body: StatusChange) -> dict[str, Any]:
    """Reopen a card (status ``active``) or close it with a verdict."""
    research_dir = get_research_dir(create=False)
    if body.status is not Status.ACTIVE and body.reopen_if is None:
        raise HTTPException(422, "a verdict needs reopen_if")
    with _register_errors():
        _require(research_dir, card_id)
        if body.status is Status.ACTIVE:
            card = reopen(research_dir, card_id, body.reason, date.today(), actor=body.actor)
        else:
            assert body.reopen_if is not None
            card = write_verdict(
                research_dir,
                card_id,
                body.status,
                body.reason,
                body.reopen_if,
                date.today(),
                actor=body.actor,
            )
    return {"card": card_data(card)}


def _require(research_dir: Path, card_id: str) -> ResearchCard:
    card = load_card(research_dir, card_id)
    if card is None:
        raise HTTPException(404, f"no research card {card_id}")
    return card


@contextmanager
def _register_errors() -> Iterator[None]:
    """Map register failures to HTTP errors: refusals 409, defects 409 'unreadable'."""
    try:
        yield
    except (CommandError, ResearchRepoError) as exc:
        raise HTTPException(409, str(exc)) from exc
    except (CardError, LedgerError, PendingError) as exc:
        raise HTTPException(409, f"research register unreadable: {exc}") from exc
