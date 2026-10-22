"""Read-only views of the research register: ``fwbg research list``, ``show`` and API data."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from fwbg.research.card import CARD_FILE_RE, CardError, ResearchCard, Status, load_card
from fwbg.research.commands import CommandError
from fwbg.research.fingerprint import canonical_json
from fwbg.research.gate import criteria_denials
from fwbg.research.ledger import LEDGER_FILENAME, Trial, TrialKind, read_trials
from fwbg.research.pending import PendingError, live_runs
from fwbg.research.repo import config_path
from fwbg.research.stats import (
    best_result,
    evaluations,
    plural,
    unrankable_configs,
    unranked_evaluations,
    unranked_note,
)

WIDTH = 120
# Indented by two spaces in ``show``.
_COVERAGE_WIDTH = WIDTH - 2
# The holdout split is part of the window, not an explored parameter.
_ALWAYS = (
    "symbols",
    "strategy.timeframe",
    "strategy.start_date",
    "strategy.end_date",
    "strategy.holdout_cutoff",
)
_EVALUATED = "evaluated symbols"
# Free text that never changes a result.
_IGNORED_KEYS = frozenset({"_comment"})
_MAX_VALUES = 10
_MAX_PATH = 72
_MIN_ROOM = 24
_MISSING = "-"
_INDEX = re.compile(r"\[([0-9]+)\]")


def list_lines(
    research_dir: Path, status: Status | None = None, tag: str | None = None
) -> list[str]:
    """One line per card, in id order; a card that does not load shows its error."""
    trials = read_trials(research_dir / LEDGER_FILENAME)
    unrankable = unrankable_configs(research_dir, trials)
    rows: list[tuple[str, ...] | str] = []
    for card_id, card in iter_cards(research_dir, status, tag):
        if isinstance(card, CardError):
            rows.append(f"{card_id}  unreadable: {card}")
            continue
        own = [trial for trial in trials if trial.strategy == card.id]
        counted = plural(sum(trial.counted for trial in own), "trial")
        rows.append(
            (
                counted,
                _best(own, unrankable),
                _holdout(card, own),
                card.id,
                card.status.value,
                card.title,
            )
        )
    # Column widths over all rows, so a long symbol or a two-digit count does not shift the rest.
    widths = [
        max((len(row[i]) for row in rows if isinstance(row, tuple)), default=0) for i in range(3)
    ]
    lines = []
    for row in rows:
        if isinstance(row, str):
            lines.append(_clip(row))
            continue
        counted, best, holdout, card_id, card_status, title = row
        lines.append(
            _clip(
                f"{card_id}  {card_status:<9} {counted:<{widths[0]}}  {best:<{widths[1]}}  "
                f"{holdout:<{widths[2]}}  {title}"
            )
        )
    return lines


def iter_cards(
    research_dir: Path, status: Status | None = None, tag: str | None = None
) -> Iterator[tuple[str, ResearchCard | CardError]]:
    """The cards matching *status* and *tag* in id order, each with its id.

    A card that does not load comes as its :class:`CardError`, under every filter:
    it might match, and it needs fixing.
    """
    for card_id in card_ids(research_dir):
        try:
            card = load_card(research_dir, card_id)
        except CardError as exc:
            yield card_id, exc
            continue
        if card is None:
            continue
        if (status is None or card.status is status) and (tag is None or tag in card.tags):
            yield card_id, card


def show_text(research_dir: Path, card_id: str) -> str:
    """A card with its criteria state, body, live runs, trial timeline and coverage."""
    card = load_card(research_dir, card_id)
    if card is None:
        raise CommandError(f"no research card {card_id}")
    own = [t for t in read_trials(research_dir / LEDGER_FILENAME) if t.strategy == card.id]
    name_width = max(len(name) for name in card.success_criteria)
    counted = sum(trial.counted for trial in own)
    unranked = unranked_note(unranked_evaluations(own, unrankable_configs(research_dir, own)))
    lines = [
        _clip(f"{card.id}  {card.title}"),
        _clip(
            f"status: {card.status.value}  tags: {', '.join(card.tags) or '-'}  "
            f"created: {card.created}"
        ),
        f"holdout cutoff: {card.holdout_cutoff}  "
        f"contaminated: {'yes' if card.holdout_contaminated else 'no'}  {_holdout(card, own)}",
        "",
        f"Criteria ({_criteria_state(card, own)}):",
        *(f"  {name:<{name_width}}  {value:g}" for name, value in card.success_criteria.items()),
        "",
        card.body.strip("\n"),
        "",
        *_live_lines(research_dir, card.id),
        "",
        f"Trials: {counted} counted ({evaluations(own)} symbol evaluations), "
        f"{len(own) - counted} failed or smoke",
        *([unranked] if unranked is not None else []),
        *(_trial_row(trial) for trial in own),
        "",
        *coverage_section(research_dir, own),
    ]
    return "\n".join(lines)


def card_summary(
    card: ResearchCard,
    own: Sequence[Trial],
    running: int | None,
    *,
    unrankable: Mapping[str, str],
) -> dict[str, Any]:
    """One card's list entry as JSON-ready data; *own* are its trials.

    *running* is the number of its live runs, ``None`` if unknown; *unrankable*
    maps the fingerprints whose runs below RANKED_PROTOCOL are not ranked to
    their reason (:func:`~fwbg.research.stats.unrankable_configs`).
    """
    best = best_result(own, unrankable)
    return {
        "id": card.id,
        "title": card.title,
        "status": card.status.value,
        "tags": list(card.tags),
        "created": card.created.isoformat(),
        "holdout_cutoff": card.holdout_cutoff.isoformat(),
        "trials": sum(trial.counted for trial in own),
        "best": (
            None
            if best is None
            else {"trial": best.trial_id, "symbol": best.symbol, "metrics": dict(best.metrics)}
        ),
        "holdout_used": any(trial.kind is TrialKind.HOLDOUT for trial in own),
        "running": running,
    }


def card_data(card: ResearchCard) -> dict[str, Any]:
    """The whole card as JSON-ready data: front matter plus the Markdown body."""
    return {
        "id": card.id,
        "title": card.title,
        "status": card.status.value,
        "tags": list(card.tags),
        "created": card.created.isoformat(),
        "holdout_cutoff": card.holdout_cutoff.isoformat(),
        "holdout_contaminated": card.holdout_contaminated,
        "success_criteria": dict(card.success_criteria),
        "registered_criteria_hash": card.registered_criteria_hash,
        "criteria_amendments": [
            {
                "date": amendment.date.isoformat(),
                "reason": amendment.reason,
                "criteria_hash": amendment.criteria_hash,
            }
            for amendment in card.criteria_amendments
        ],
        "body": card.body,
    }


def coverage(configs: Sequence[Mapping[str, Any]], evaluated: Iterable[str] = ()) -> list[str]:
    """The paths whose values differ between *configs*, plus symbols and the window.

    List items are matched by their ``name`` (``[<name>]``), or all together
    (``[*]``) when not every item has one, so reordering a list changes nothing;
    lists of lists keep their index. One line per path with the sorted union of
    its values, plus ``-`` if some config lacks the path. *evaluated* are the
    symbols the trials evaluated, listed after ``symbols``.
    """
    flat = [_flatten(config, "", {}) for config in configs]
    if not flat:
        return []
    varied = {
        path
        for path in set().union(*flat) - set(_ALWAYS)
        if len({frozenset(values.get(path, ())) for values in flat}) > 1
        or any(path not in values for values in flat)
    }
    rows = [(path, _path_values(path, flat)) for path in _ALWAYS]
    rows.insert(1, (_EVALUATED, sorted(set(evaluated)) or [_MISSING]))
    rows += [(path, _path_values(path, flat)) for path in sorted(varied, key=_natural)]
    width = min(max(len(path) for path, _ in rows), _MAX_PATH)
    lines = []
    for path, shown in rows:
        prefix = f"{path:<{width}}  "
        room = max(_COVERAGE_WIDTH - len(prefix), _MIN_ROOM)
        lines.append(prefix + _value_list(shown, room))
    return lines


def _path_values(path: str, flat: Sequence[Mapping[str, set[str]]]) -> list[str]:
    found = sorted(set().union(*(values.get(path, ()) for values in flat)), key=_value_key)
    return found + [_MISSING] if any(path not in values for values in flat) else found


def card_ids(research_dir: Path) -> list[str]:
    """The ids of the card files in the register, sorted."""
    return sorted(
        {
            match.group(1)
            for path in (research_dir / "strategies").glob("S-*.md")
            if (match := CARD_FILE_RE.fullmatch(path.name))
        }
    )


def _best(own: Sequence[Trial], unrankable: Mapping[str, str]) -> str:
    best = best_result(own, unrankable)
    if best is None:
        return "best Sharpe -"
    return f"best Sharpe {best.sharpe:.2f} ({best.trial_id} {best.symbol})"


def _holdout(card: ResearchCard, own: Sequence[Trial]) -> str:
    if card.holdout_contaminated:
        return "holdout contaminated"
    looks = sum(trial.kind is TrialKind.HOLDOUT for trial in own)
    return f"holdout used {looks}x" if looks else "holdout unused"


def _criteria_state(card: ResearchCard, own: Sequence[Trial]) -> str:
    if criteria_denials(card, own):
        return "edited by hand - runs are refused"
    if card.criteria_amendments:
        latest = card.criteria_amendments[-1].date
        return f"amended {len(card.criteria_amendments)}x (latest {latest})"
    return "registered"


def _live_lines(research_dir: Path, card_id: str) -> list[str]:
    try:
        live = live_runs(research_dir, card_id)
    except PendingError as exc:
        return [_clip(f"Runs in progress: unknown ({exc})")]
    if not live:
        return ["Runs in progress: none"]
    return [
        "Runs in progress:",
        *(
            _clip(
                f"  {run.run_id}  started {run.trial.ts[:16]}  pid {run.pid}  "
                f"{run.trial.note or ''}".rstrip()
            )
            for run in live
        ),
    ]


def _trial_row(trial: Trial) -> str:
    symbols = sorted(trial.metrics)
    named = ",".join(symbols[:2]) + (f" +{len(symbols) - 2}" if len(symbols) > 2 else "")
    sharpes = [
        float(value)
        for values in trial.metrics.values()
        if isinstance(value := values.get("sharpe"), (int, float))
    ]
    sharpe = f"{max(sharpes):.2f}" if sharpes else "-"
    return _clip(
        f"  {trial.trial_id}  {trial.ts[:10]}  {trial.kind.value:<9} {trial.status.value:<6} "
        f"{named or '-':<17} {sharpe:>6}  {trial.note or ''}".rstrip()
    )


def coverage_section(research_dir: Path, own: Sequence[Trial]) -> list[str]:
    """The coverage lines of ``show``: a header, then the indented :func:`coverage`."""
    configs = []
    left_out = 0
    for fingerprint in dict.fromkeys(trial.fingerprint for trial in own):
        path = config_path(research_dir, fingerprint)
        if not path.exists():
            # Study trials carry synthetic fingerprints without a stored config.
            left_out += sum(trial.fingerprint == fingerprint for trial in own)
            continue
        try:
            configs.append(json.loads(path.read_text(encoding="utf-8")))
        except ValueError as exc:
            raise CommandError(f"configs/{path.name}: {exc}") from exc
    header = f"Coverage: {plural(len(configs), 'config')}"
    if left_out:
        header += f" ({plural(left_out, 'trial')} without a stored config left out)"
    evaluated = {symbol for trial in own for symbol in trial.metrics}
    return [header, *(f"  {line}" for line in coverage(configs, evaluated))]


def _flatten(value: Any, path: str, out: dict[str, set[str]]) -> dict[str, set[str]]:
    """Map each leaf path to the rendered values found there (items sharing a key unite)."""
    if isinstance(value, Mapping):
        for key in sorted(value):
            if key not in _IGNORED_KEYS:
                _flatten(value[key], f"{path}.{key}" if path else str(key), out)
    elif isinstance(value, list) and value and all(isinstance(item, Mapping) for item in value):
        named = all(isinstance(item.get("name"), str) for item in value)
        for item in value:
            _flatten(item, f"{path}[{item['name'] if named else '*'}]", out)
    elif isinstance(value, list) and value and all(isinstance(item, list) for item in value):
        for index, item in enumerate(value):
            _flatten(item, f"{path}[{index}]", out)
    else:
        out.setdefault(path, set()).add(canonical_json(value))
    return out


def _natural(path: str) -> str:
    """Sort key that puts ``a[2]`` before ``a[10]``."""
    return _INDEX.sub(lambda match: f"[{int(match.group(1)):09d}]", path)


def _value_key(value: str) -> tuple[int, float, str]:
    """Numbers numerically, then other values as text."""
    parsed = json.loads(value)
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        return (0, float(parsed), value)
    return (1, 0.0, value)


def _value_list(values: Sequence[str], room: int) -> str:
    """At most ``_MAX_VALUES`` values that fit into *room*, then how many are left out."""
    shown = list(values[:_MAX_VALUES])
    while True:
        rest = len(values) - len(shown)
        text = ", ".join(shown) + (f" … (+{rest} more)" if rest else "")
        if len(text) <= room or len(shown) <= 1:
            return _clip(text, room)
        shown.pop()


def _clip(text: str, width: int = WIDTH) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"
