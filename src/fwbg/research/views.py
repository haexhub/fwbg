"""Read-only views of the research register: ``fwbg research list`` and ``show``."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fwbg.research.card import CARD_FILE_RE, CardError, ResearchCard, Status, load_card
from fwbg.research.commands import CommandError
from fwbg.research.fingerprint import canonical_json
from fwbg.research.gate import criteria_denials
from fwbg.research.ledger import LEDGER_FILENAME, Trial, TrialKind, read_trials
from fwbg.research.pending import live_runs
from fwbg.research.stats import best_result, evaluations

WIDTH = 120
# Indented by two spaces in ``show``.
_COVERAGE_WIDTH = WIDTH - 2
_ALWAYS = ("symbols", "strategy.timeframe", "strategy.start_date", "strategy.end_date")
_MAX_VALUES = 10
_MAX_PATH = 72
_MISSING = "-"
_INDEX = re.compile(r"\[([0-9]+)\]")


def list_lines(
    research_dir: Path, status: Status | None = None, tag: str | None = None
) -> list[str]:
    """One line per card, in id order; a card that does not load shows its error."""
    trials = read_trials(research_dir / LEDGER_FILENAME)
    rows: list[tuple[str, ...] | str] = []
    for card_id in _card_ids(research_dir):
        try:
            card = load_card(research_dir, card_id)
        except CardError as exc:
            # Shown under every filter: it might match, and it needs fixing.
            rows.append(f"{card_id}  unreadable: {exc}")
            continue
        if card is None:
            continue
        if (status is not None and card.status is not status) or (
            tag is not None and tag not in card.tags
        ):
            continue
        own = [trial for trial in trials if trial.strategy == card.id]
        counted = _plural(sum(trial.counted for trial in own), "trial")
        rows.append(
            (counted, _best(own), _holdout(card, own), card.id, card.status.value, card.title)
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


def show_text(research_dir: Path, card_id: str) -> str:
    """A card with its criteria state, body, live runs, trial timeline and coverage."""
    card = load_card(research_dir, card_id)
    if card is None:
        raise CommandError(f"no research card {card_id}")
    own = [t for t in read_trials(research_dir / LEDGER_FILENAME) if t.strategy == card.id]
    name_width = max(len(name) for name in card.success_criteria)
    counted = sum(trial.counted for trial in own)
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
        *(_trial_row(trial) for trial in own),
        "",
        *_coverage_section(research_dir, own),
    ]
    return "\n".join(lines)


def coverage(configs: Sequence[Mapping[str, Any]]) -> list[str]:
    """The paths whose values differ between *configs*, plus symbols and the window.

    One line per path with its sorted distinct values; ``-`` marks configs
    without the path.
    """
    flat = [_flatten(config, "", {}) for config in configs]
    if not flat:
        return []
    varied = {
        path
        for path in set().union(*flat) - set(_ALWAYS)
        if len({values.get(path) for values in flat}) > 1
    }
    listed = [*_ALWAYS, *sorted(varied, key=_natural)]
    width = min(max(len(path) for path in listed), _MAX_PATH)
    lines = []
    for path in listed:
        found = sorted({values.get(path) for values in flat}, key=_value_key)
        shown = [_MISSING if value is None else value for value in found]
        prefix = f"{path:<{width}}  "
        lines.append(prefix + _value_list(shown, _COVERAGE_WIDTH - len(prefix)))
    return lines


def _card_ids(research_dir: Path) -> list[str]:
    return sorted(
        {
            match.group(1)
            for path in (research_dir / "strategies").glob("S-*.md")
            if (match := CARD_FILE_RE.fullmatch(path.name))
        }
    )


def _best(own: Sequence[Trial]) -> str:
    best = best_result(own)
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
    live = live_runs(research_dir, card_id)
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


def _coverage_section(research_dir: Path, own: Sequence[Trial]) -> list[str]:
    configs = []
    left_out = 0
    for fingerprint in dict.fromkeys(trial.fingerprint for trial in own):
        path = research_dir / "configs" / f"{fingerprint.removeprefix('sha256:')}.json"
        if not path.exists():
            # Study trials carry synthetic fingerprints without a stored config.
            left_out += sum(trial.fingerprint == fingerprint for trial in own)
            continue
        try:
            configs.append(json.loads(path.read_text(encoding="utf-8")))
        except ValueError as exc:
            raise CommandError(f"configs/{path.name}: {exc}") from exc
    header = f"Coverage: {_plural(len(configs), 'config')}"
    if left_out:
        header += f" ({_plural(left_out, 'trial')} without a stored config left out)"
    return [header, *(f"  {line}" for line in coverage(configs))]


def _flatten(value: Any, path: str, out: dict[str, str]) -> dict[str, str]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            _flatten(value[key], f"{path}.{key}" if path else str(key), out)
    elif (
        isinstance(value, list)
        and value
        and all(isinstance(item, (Mapping, list)) for item in value)
    ):
        for index, item in enumerate(value):
            _flatten(item, f"{path}[{index}]", out)
    else:
        out[path] = canonical_json(value)
    return out


def _natural(path: str) -> str:
    """Sort key that puts ``a[2]`` before ``a[10]``."""
    return _INDEX.sub(lambda match: f"[{int(match.group(1)):09d}]", path)


def _value_key(value: str | None) -> tuple[int, float, str]:
    """Numbers numerically, then other values as text, a missing path last."""
    if value is None:
        return (2, 0.0, "")
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


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _clip(text: str, width: int = WIDTH) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"
