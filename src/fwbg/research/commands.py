"""Write commands of the research register: register, amend, verdict, reopen.

Each command re-reads the card under ``register_lock``, rewrites it and
commits all register paths. If the commit fails, the change stays on disk,
the command raises :class:`CommandError` saying so, and the next register
commit includes it.
"""

from __future__ import annotations

import calendar
import logging
import re
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
from fwbg.research.gate import criteria_denials
from fwbg.research.ledger import LEDGER_FILENAME, Trial, TrialKind, TrialStatus, read_trials
from fwbg.research.pending import live_runs, read_pending, recover_stale
from fwbg.research.repo import (
    CRITERIA_TEMPLATE,
    REGISTER_PATHS,
    ResearchRepoError,
    commit_register,
    register_lock,
)
from fwbg.research.stats import (
    MIN_RETURN_SAMPLES,
    best_result,
    criteria_check,
    deflated_best,
    evaluations,
    return_samples,
)

DEFAULT_HOLDOUT_MONTHS = 18
VERDICT_STATUSES = (Status.PARKED, Status.REJECTED, Status.CANDIDATE)
_PLACEHOLDER = "To be written before the first iteration."
_BODY = (
    "## Hypothesis\n\n{hypothesis}\n\n## Mechanism\n\n{mechanism}\n\n"
    "## History\n\n## Verdict\n\nOpen.\n"
)
# A top-level or section heading inside section text would split the card's sections.
_HEADING_LINE = re.compile(r"^#{1,2}[ \t]", re.MULTILINE)

log = logging.getLogger(__name__)


class CommandError(Exception):
    """A research command was refused, or its commit failed; the message says which."""


class CommandInputError(CommandError):
    """A research command was refused for its arguments alone, not the register's state."""


class UncommittedError(CommandError):
    """The change was written to disk, but its commit failed; the next commit includes it."""

    def __init__(self, message: str, card_id: str) -> None:
        super().__init__(message)
        self.card_id = card_id


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
        raise CommandInputError(f"holdout cutoff {holdout_cutoff} lies in the future")
    for text, what in ((hypothesis, "hypothesis"), (mechanism, "mechanism")):
        if text is not None and _HEADING_LINE.search(text):
            raise CommandInputError(f"{what} must not contain '#' or '##' heading lines")
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
        _commit(research_dir, card.id, f"research: register {card.id} {title}")
    return path


def amend_criteria(
    research_dir: Path, card_id: str, changes: Mapping[str, float], reason: str, today: date
) -> ResearchCard:
    """Change success criteria and record the change as an amendment."""
    _one_line(reason, "reason")
    if not changes:
        raise CommandInputError("nothing to amend; pass --set name=value")
    with register_lock(research_dir):
        card = _require(research_dir, card_id)
        if card.status is not Status.ACTIVE:
            raise CommandError(f"{card_id} is {card.status.value}; reopen it first")
        own = [t for t in read_trials(research_dir / LEDGER_FILENAME) if t.strategy == card_id]
        if criteria_denials(card, own):
            raise CommandError(
                f"the criteria of {card_id} do not match its recorded registration and "
                "amendments; restore them by hand (see the register's git history) before amending"
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
        _commit(research_dir, card_id, f"research: amend {card_id} criteria")
    return updated


def write_verdict(
    research_dir: Path,
    card_id: str,
    status: Status,
    reason: str,
    reopen_if: str,
    today: date,
    *,
    actor: str | None = None,
) -> ResearchCard:
    """Close an active card with a verdict and the statistics it rests on.

    *actor* names who decided (an API caller); the CLI passes none.
    """
    _one_line(reason, "reason")
    _one_line(reopen_if, "reopen condition")
    if actor is not None:
        _one_line(actor, "actor")
    if status not in VERDICT_STATUSES:
        raise CommandInputError(f"a verdict sets parked, rejected or candidate, not {status.value}")
    with register_lock(research_dir):
        # Committed on its own, so a refused verdict leaves no recovered trial uncommitted.
        if recover_stale(research_dir):
            try:
                commit_register(research_dir, REGISTER_PATHS, "trial: record interrupted runs")
            except ResearchRepoError as exc:
                # The verdict's own commit stages the recovered trials again.
                log.warning("recovered runs not committed: %s", exc)
        card = _require(research_dir, card_id)
        if card.status is not Status.ACTIVE:
            raise CommandError(f"{card_id} is {card.status.value}; reopen it before a new verdict")
        live = {run.run_id for run in live_runs(research_dir, card_id)}
        pending = read_pending(research_dir)
        own = [run.run_id for run in pending if run.trial.strategy == card_id]
        if live:
            names = ", ".join(sorted(live))
            raise CommandError(f"{card_id} has runs in progress ({names}); wait for them")
        if own:
            names = ", ".join(own)
            raise CommandError(
                f"{card_id} has runs that could not be recorded ({names}); see the warning above"
            )
        trials = read_trials(research_dir / LEDGER_FILENAME)
        entry = verdict_entry(
            card, trials, status, reason, reopen_if, today, len(pending), actor=actor
        )
        updated = replace(card, status=status, body=add_section_entry(card.body, "Verdict", entry))
        replace_card(research_dir, updated)
        _commit(research_dir, card_id, f"research: verdict {card_id} {status.value}")
    return updated


def reopen(
    research_dir: Path, card_id: str, reason: str, today: date, *, actor: str | None = None
) -> ResearchCard:
    """Turn a parked or rejected card active again, recording why (and who, if given)."""
    _one_line(reason, "reason")
    if actor is not None:
        _one_line(actor, "actor")
    with register_lock(research_dir):
        card = _require(research_dir, card_id)
        if card.status not in (Status.PARKED, Status.REJECTED):
            raise CommandError(f"{card_id} is {card.status.value}; only parked or rejected reopen")
        by = f" by {actor}" if actor is not None else ""
        entry = f"- {today.isoformat()}: reopened from {card.status.value}{by}: {reason}"
        body = add_section_entry(card.body, "History", entry, before=("Verdict",))
        updated = replace(card, status=Status.ACTIVE, body=body)
        replace_card(research_dir, updated)
        _commit(research_dir, card_id, f"research: reopen {card_id}")
    return updated


def verdict_entry(
    card: ResearchCard,
    trials: Sequence[Trial],
    status: Status,
    reason: str,
    reopen_if: str,
    today: date,
    pending: int = 0,
    *,
    actor: str | None = None,
) -> str:
    """The dated verdict text: reasoning plus the statistics it rests on.

    *pending* is the number of pending runs (of other cards) not yet in *trials*;
    *actor*, if given, is recorded after the reopen condition.
    """
    own = [trial for trial in trials if trial.strategy == card.id]
    looks = [str(trial.trial_id) for trial in own if trial.kind is TrialKind.HOLDOUT]
    versions = [_short(h) for h in dict.fromkeys(trial.criteria_hash for trial in own)]
    excluded = f"; register N excludes {pending} pending runs" if pending else ""
    lines = [
        f"### {today.isoformat()}: {status.value}",
        "",
        f"Reason: {reason}",
        f"Reopen if: {reopen_if}",
        *([f"By: {actor}"] if actor is not None else []),
        f"Trials: {sum(t.counted for t in own)} counted on {card.id} "
        f"({evaluations(own)} symbol evaluations); {sum(t.counted for t in trials)} counted "
        f"in the register ({evaluations(trials)} symbol evaluations){excluded}.",
        f"Criteria versions used: {len(versions)}"
        + (f" ({', '.join(versions)})." if versions else "."),
        f"Holdout: {'used by ' + ', '.join(looks) if looks else 'unused'}.",
    ]
    best = best_result(own)
    if best is None:
        lines.append("Best development result: none with a Sharpe ratio.")
    else:
        check = "; ".join(criteria_check(card.success_criteria, best.metrics))
        amendments = len(card.criteria_amendments)
        version = f"amendment {amendments}" if amendments else "registration"
        lines.append(
            f"Best development result: {best.trial_id} {best.symbol}, checked against "
            f"the current criteria ({version}): {check}."
        )
    lines.append(_dsr_line(own))
    lines.extend(_holdout_lines(card, own))
    return "\n".join(lines)


def _dsr_line(own: Sequence[Trial]) -> str:
    result = deflated_best(own)
    if result is None:
        have = len(return_samples(own))
        return (
            "Deflated Sharpe ratio: too few trials with return statistics "
            f"({have} of {MIN_RETURN_SAMPLES} needed)."
        )
    best = result.best
    basis = (
        f"per-trade SR {best.sr:.4f} over {best.n_returns} returns, "
        f"skew {best.skew:.3g}, kurtosis {best.kurtosis:.3g}; "
        f"N={result.n_trials} symbol evaluations; V[SR]={result.sr_variance:.3g} from "
        f"{result.samples} evaluations with return statistics; "
        f"E[max SR]={result.expected_max_sr:.4f}"
    )
    if result.dsr is None:
        return (
            f"Deflated Sharpe ratio: not defined for {best.trial_id} {best.symbol} "
            f"(its skew and kurtosis leave no positive SR variance; {basis})."
        )
    return f"Deflated Sharpe ratio: {result.dsr:.3f} for {best.trial_id} {best.symbol} ({basis})."


def _holdout_lines(card: ResearchCard, own: Sequence[Trial]) -> list[str]:
    """One line per holdout look: its symbols checked against the current criteria."""
    lines = []
    for trial in own:
        if trial.kind is not TrialKind.HOLDOUT:
            continue
        window = f"{trial.window.get('start') or 'open'}..{trial.window.get('end') or 'open'}"
        head = f"Holdout result {trial.trial_id} (window {window}, cutoff {trial.holdout_cutoff})"
        if trial.status is not TrialStatus.OK:
            lines.append(f"{head}: failed, no symbol evaluated.")
            continue
        parts = [
            (
                f"{symbol} " + "; ".join(criteria_check(card.success_criteria, values))
                if any(isinstance(v, (int, float)) for v in values.values())
                else f"{symbol} no holdout metrics"
            )
            for symbol, values in trial.metrics.items()
        ]
        lines.append(f"{head}: {' | '.join(parts)}.")
    return lines


def _commit(research_dir: Path, card_id: str, message: str) -> None:
    try:
        commit_register(research_dir, REGISTER_PATHS, message)
    except ResearchRepoError as exc:
        raise UncommittedError(
            f"{card_id} was updated on disk but not committed: {exc}; "
            "the next register write commits it",
            card_id,
        ) from exc


def _short(hash_: str) -> str:
    return hash_.removeprefix("sha256:")[:12]


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
        raise CommandInputError(f"{what} must be one non-empty line, got {value!r}")
