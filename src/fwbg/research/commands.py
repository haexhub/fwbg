"""Write commands of the research register: register, amend, verdict, reopen.

Each command re-reads the card under ``register_lock``, rewrites it and
commits all register paths. If the commit fails, the change stays on disk
and the next register commit includes it.
"""

from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

import yaml

from fwbg.research.card import (
    CriteriaAmendment,
    ResearchCard,
    Status,
    UniqueKeyLoader,
    add_section_entry,
    criteria_hash,
    load_card,
    next_card_id,
    replace_card,
    slugify,
    validate_criteria,
    write_card,
)
from fwbg.research.ledger import LEDGER_FILENAME, Trial, TrialKind, read_trials
from fwbg.research.pending import live_runs, recover_stale
from fwbg.research.repo import CRITERIA_TEMPLATE, REGISTER_PATHS, commit_register, register_lock
from fwbg.research.stats import best_result, criteria_check, evaluations

DEFAULT_HOLDOUT_MONTHS = 18
VERDICT_STATUSES = (Status.PARKED, Status.REJECTED, Status.CANDIDATE)
_PLACEHOLDER = "To be written before the first iteration."
_BODY = (
    "## Hypothesis\n\n{hypothesis}\n\n## Mechanism\n\n{mechanism}\n\n"
    "## History\n\n## Verdict\n\nOpen.\n"
)


class CommandError(Exception):
    """A research command was refused; nothing was changed."""


def months_before(day: date, months: int) -> date:
    """The same day *months* earlier, clamped to the end of a shorter month."""
    year, month = divmod(day.year * 12 + day.month - 1 - months, 12)
    return date(year, month + 1, min(day.day, calendar.monthrange(year, month + 1)[1]))


def new_card(
    research_dir: Path,
    title: str,
    tags: Sequence[str],
    today: date,
    *,
    cutoff: date | None = None,
    slug: str | None = None,
    criteria: Mapping[str, float] | None = None,
    hypothesis: str | None = None,
    mechanism: str | None = None,
) -> Path:
    """Register a new active card with criteria frozen from the template plus *criteria*."""
    _one_line(title, "title")
    holdout_cutoff = cutoff or months_before(today, DEFAULT_HOLDOUT_MONTHS)
    if holdout_cutoff > today:
        raise CommandError(f"holdout cutoff {holdout_cutoff} lies in the future")
    frozen = validate_criteria({**_template(research_dir), **(criteria or {})})
    body = _BODY.format(hypothesis=hypothesis or _PLACEHOLDER, mechanism=mechanism or _PLACEHOLDER)
    with register_lock(research_dir):
        card = ResearchCard(
            id=next_card_id(research_dir, read_trials(research_dir / LEDGER_FILENAME)),
            title=title,
            status=Status.ACTIVE,
            tags=_tags(tags),
            created=today,
            holdout_cutoff=holdout_cutoff,
            holdout_contaminated=False,
            success_criteria=frozen,
            registered_criteria_hash=criteria_hash(frozen),
            criteria_amendments=(),
            body=body,
        )
        path = write_card(research_dir, card, slug or slugify(title))
        commit_register(research_dir, REGISTER_PATHS, f"research: register {card.id} {title}")
    return path


def amend_criteria(
    research_dir: Path, card_id: str, changes: Mapping[str, float], reason: str, today: date
) -> ResearchCard:
    """Change success criteria and record the change as an amendment."""
    _one_line(reason, "reason")
    if not changes:
        raise CommandError("nothing to amend; pass --set name=value")
    with register_lock(research_dir):
        card = _require(research_dir, card_id)
        recorded = (
            card.criteria_amendments[-1].criteria_hash
            if card.criteria_amendments
            else card.registered_criteria_hash
        )
        if card.criteria_hash != recorded:
            raise CommandError(
                f"the criteria of {card_id} were edited by hand; restore them before amending"
            )
        criteria = validate_criteria({**card.success_criteria, **changes})
        new_hash = criteria_hash(criteria)
        if new_hash == card.criteria_hash:
            raise CommandError(f"the criteria of {card_id} are unchanged")
        amendment = CriteriaAmendment(date=today, reason=reason, criteria_hash=new_hash)
        updated = replace(
            card,
            success_criteria=criteria,
            criteria_amendments=(*card.criteria_amendments, amendment),
        )
        replace_card(research_dir, updated)
        commit_register(research_dir, REGISTER_PATHS, f"research: amend {card_id} criteria")
    return updated


def write_verdict(
    research_dir: Path, card_id: str, status: Status, reason: str, reopen_if: str, today: date
) -> ResearchCard:
    """Close an active card with a verdict and the statistics it rests on."""
    _one_line(reason, "reason")
    _one_line(reopen_if, "reopen condition")
    if status not in VERDICT_STATUSES:
        raise CommandError(f"a verdict sets parked, rejected or candidate, not {status.value}")
    with register_lock(research_dir):
        # Committed on its own, so a refused verdict leaves no recovered trial uncommitted.
        if recover_stale(research_dir):
            commit_register(research_dir, REGISTER_PATHS, "trial: record interrupted runs")
        card = _require(research_dir, card_id)
        if card.status is not Status.ACTIVE:
            raise CommandError(f"{card_id} is {card.status.value}; reopen it before a new verdict")
        running = live_runs(research_dir, card_id)
        if running:
            names = ", ".join(run.run_id for run in running)
            raise CommandError(f"{card_id} has runs in progress ({names}); wait for them")
        trials = read_trials(research_dir / LEDGER_FILENAME)
        entry = verdict_entry(card, trials, status, reason, reopen_if, today)
        updated = replace(card, status=status, body=add_section_entry(card.body, "Verdict", entry))
        replace_card(research_dir, updated)
        commit_register(research_dir, REGISTER_PATHS, f"research: verdict {card_id} {status.value}")
    return updated


def reopen(research_dir: Path, card_id: str, reason: str, today: date) -> ResearchCard:
    """Turn a parked or rejected card active again, recording why."""
    _one_line(reason, "reason")
    with register_lock(research_dir):
        card = _require(research_dir, card_id)
        if card.status not in (Status.PARKED, Status.REJECTED):
            raise CommandError(f"{card_id} is {card.status.value}; only parked or rejected reopen")
        entry = f"- {today.isoformat()}: reopened from {card.status.value}: {reason}"
        body = add_section_entry(card.body, "History", entry, before=("Verdict",))
        updated = replace(card, status=Status.ACTIVE, body=body)
        replace_card(research_dir, updated)
        commit_register(research_dir, REGISTER_PATHS, f"research: reopen {card_id}")
    return updated


def verdict_entry(
    card: ResearchCard,
    trials: Sequence[Trial],
    status: Status,
    reason: str,
    reopen_if: str,
    today: date,
) -> str:
    """The dated verdict text: reasoning plus the statistics it rests on."""
    own = [trial for trial in trials if trial.strategy == card.id]
    looks = [str(trial.trial_id) for trial in own if trial.kind is TrialKind.HOLDOUT]
    lines = [
        f"### {today.isoformat()}: {status.value}",
        "",
        f"Reason: {reason}",
        f"Reopen if: {reopen_if}",
        f"Trials: {sum(t.counted for t in own)} counted on {card.id} "
        f"({evaluations(own)} symbol evaluations); {sum(t.counted for t in trials)} counted "
        f"in the register ({evaluations(trials)} symbol evaluations).",
        f"Criteria versions used: {len({trial.criteria_hash for trial in own})}.",
        f"Holdout: {'used by ' + ', '.join(looks) if looks else 'unused'}.",
    ]
    best = best_result(own)
    if best is None:
        lines.append("Best development result: none with a Sharpe ratio.")
    else:
        check = "; ".join(criteria_check(card.success_criteria, best.metrics))
        lines.append(f"Best development result: {best.trial_id} {best.symbol}: {check}.")
    lines.append("Deflated Sharpe ratio: not computed yet (register step 5).")
    return "\n".join(lines)


def _require(research_dir: Path, card_id: str) -> ResearchCard:
    card = load_card(research_dir, card_id)
    if card is None:
        raise CommandError(f"no research card {card_id}")
    return card


def _template(research_dir: Path) -> dict[str, object]:
    path = research_dir / CRITERIA_TEMPLATE
    if not path.exists():
        raise CommandError(
            f"research register not initialised at {research_dir}; run 'fwbg research init'"
        )
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except (yaml.YAMLError, ValueError) as exc:
        raise CommandError(f"{CRITERIA_TEMPLATE}: {exc}") from exc
    if not isinstance(data, dict):
        raise CommandError(f"{CRITERIA_TEMPLATE} must be a mapping of criteria")
    return data


def _tags(tags: Sequence[str]) -> tuple[str, ...]:
    cleaned = tuple(tag.strip() for tag in tags if tag.strip())
    for tag in cleaned:
        _one_line(tag, "tag")
    return cleaned


def _one_line(value: str | None, what: str) -> None:
    # The gate's note rule: non-blank, no line break of any kind.
    if value is None or not value.strip() or value.splitlines() != [value]:
        raise CommandError(f"{what} must be one non-empty line, got {value!r}")
