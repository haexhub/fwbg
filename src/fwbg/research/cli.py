"""``fwbg research``: manage the research register, its cards and the legacy backfill."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import yaml

from fwbg.api.run_service import ACTIVE_STATUSES
from fwbg.api.workspace import get_research_dir, get_strategies_dir, get_test_results_dir
from fwbg.research.backfill import apply_mapping, ungated_runs
from fwbg.research.card import CARD_FILE_RE, CRITERIA_NAMES, CardError, Status, UniqueKeyLoader
from fwbg.research.commands import (
    VERDICT_STATUSES,
    CommandError,
    amend_criteria,
    new_card,
    reopen,
    write_verdict,
)
from fwbg.research.ledger import LEDGER_FILENAME, LedgerError, TrialKind, TrialStatus
from fwbg.research.legacy import LegacyRun, LegacyRunError, scan_runs
from fwbg.research.mapping import BackfillError, draft_mapping, load_mapping
from fwbg.research.pending import PendingError
from fwbg.research.repo import CRITERIA_TEMPLATE, ResearchRepoError, init_research_repo
from fwbg.research.views import list_lines, show_text

_OPERATOR_ERRORS = (
    BackfillError,
    CardError,
    CommandError,
    LedgerError,
    LegacyRunError,
    PendingError,
    ResearchRepoError,
    OSError,
    UnicodeDecodeError,
    yaml.YAMLError,
)


class _MappingDumper(yaml.SafeDumper):
    """Writes multi-line text as literal blocks.

    Quoted multi-line YAML folds a line the operator appends into the line
    before it; a literal block keeps every line break.
    """


def _represent_text(dumper: yaml.SafeDumper, text: str) -> yaml.ScalarNode:
    """Represent multiline YAML text as a literal block to preserve edited prose."""
    style = "|" if "\n" in text else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", text, style=style)


_MappingDumper.add_representer(str, _represent_text)

_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def _iso_date(text: str) -> date:
    """A plain ``YYYY-MM-DD`` date (``fromisoformat`` alone also takes week dates)."""
    try:
        if _ISO_DATE.fullmatch(text):
            return date.fromisoformat(text)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"expected a YYYY-MM-DD date, got {text!r}")


def _criterion(text: str) -> tuple[str, float]:
    """``name=value`` with a numeric value; the name is checked by the command."""
    name, sep, value = text.partition("=")
    try:
        if sep and name.strip():
            return name.strip(), float(value)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"expected name=number, got {text!r}")


def _tag_list(text: str) -> list[str]:
    """Comma-separated tags; at least one must be non-empty."""
    tags = [tag.strip() for tag in text.split(",") if tag.strip()]
    if not tags:
        raise argparse.ArgumentTypeError(f"expected at least one tag, got {text!r}")
    return tags


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch research commands and return a nonzero status for operator errors."""
    parser = argparse.ArgumentParser(
        prog="fwbg research",
        description="Manage the research register: cards, trials and the legacy backfill.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="create the research register (idempotent)")
    _add_card_commands(commands)
    listing = commands.add_parser("list", help="one line per research card")
    listing.add_argument("--status", choices=[status.value for status in Status])
    listing.add_argument("--tag", help="only cards with this tag")
    shown = commands.add_parser("show", help="a card with its trials and coverage")
    shown.add_argument("card_id", metavar="S-NNNN")
    backfill = commands.add_parser("backfill", help="import legacy runs and studies")
    actions = backfill.add_subparsers(dest="action", required=True)
    scan = actions.add_parser("scan", help="write a draft mapping of legacy runs to cards")
    scan.add_argument("--out", type=Path, required=True, help="mapping file to create")
    apply = actions.add_parser("apply", help="write the cards and trials of a mapping")
    apply.add_argument("mapping", type=Path, help="reviewed mapping file")
    for sub in (scan, apply):
        sub.add_argument("--results", type=Path, help="run directory root (default: workspace)")
    args = parser.parse_args(argv)
    names = [name for name, _ in getattr(args, "criteria", [])]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        parser.error(f"--set given more than once for {', '.join(repeated)}")

    try:
        if args.command == "init":
            research_dir = get_research_dir()
            created = init_research_repo(research_dir)
            print(f"{'Created' if created else 'Already initialised'}: {research_dir}")
            return 0
        if args.command in _CARD_COMMANDS:
            _CARD_COMMANDS[args.command](args, get_research_dir())
            return 0
        results_dir = args.results or get_test_results_dir()
        if args.action == "scan":
            return _scan(results_dir, args.out)
        return _apply(results_dir, args.mapping, get_research_dir())
    except _OPERATOR_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _add_card_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The write commands on research cards: new, amend, verdict, reopen."""
    criteria_help = f"success criterion name=value ({', '.join(sorted(CRITERIA_NAMES))})"
    new = commands.add_parser("new", help="register a new active research card")
    new.add_argument("title", help="one-line card title")
    new.add_argument("--tags", type=_tag_list, required=True, help="comma-separated tags")
    new.add_argument("--cutoff", type=_iso_date, help="holdout cutoff (default: 18 months ago)")
    new.add_argument("--slug", help="file-name slug (default: from the title)")
    new.add_argument("--hypothesis", help="text of the Hypothesis section")
    new.add_argument("--mechanism", help="text of the Mechanism section")
    amend = commands.add_parser("amend", help="change a card's success criteria")
    amend.add_argument("card_id", metavar="S-NNNN")
    verdict = commands.add_parser("verdict", help="close an active card with a verdict")
    verdict.add_argument("card_id", metavar="S-NNNN")
    verdict.add_argument("status", choices=[status.value for status in VERDICT_STATUSES])
    verdict.add_argument("--reopen-if", required=True, help="what would justify reopening")
    reopened = commands.add_parser("reopen", help="make a parked or rejected card active")
    reopened.add_argument("card_id", metavar="S-NNNN")
    for sub, required in ((new, False), (amend, True)):
        sub.add_argument(
            "--set",
            dest="criteria",
            type=_criterion,
            action="append",
            default=[],
            required=required,
            metavar="NAME=VALUE",
            help=criteria_help,
        )
    for sub in (amend, verdict, reopened):
        sub.add_argument("--reason", required=True, help="one-line reason, kept on the card")


def _new(args: argparse.Namespace, research_dir: Path) -> None:
    path = new_card(
        research_dir,
        args.title,
        args.tags,
        date.today(),
        cutoff=args.cutoff,
        slug=args.slug,
        criteria=dict(args.criteria),
        hypothesis=args.hypothesis,
        mechanism=args.mechanism,
    )
    match = CARD_FILE_RE.fullmatch(path.name)
    print(f"Registered {match.group(1) if match else path.stem}: {path}")


def _amend(args: argparse.Namespace, research_dir: Path) -> None:
    card = amend_criteria(
        research_dir, args.card_id, dict(args.criteria), args.reason, date.today()
    )
    print(f"Amended {card.id} (amendment {len(card.criteria_amendments)})")


def _verdict(args: argparse.Namespace, research_dir: Path) -> None:
    card = write_verdict(
        research_dir, args.card_id, Status(args.status), args.reason, args.reopen_if, date.today()
    )
    print(f"{card.id} is now {card.status.value}")


def _reopen(args: argparse.Namespace, research_dir: Path) -> None:
    card = reopen(research_dir, args.card_id, args.reason, date.today())
    print(f"{card.id} is active again")


def _list(args: argparse.Namespace, research_dir: Path) -> None:
    status = Status(args.status) if args.status else None
    for line in list_lines(research_dir, status=status, tag=args.tag):
        print(line)


def _show(args: argparse.Namespace, research_dir: Path) -> None:
    print(show_text(research_dir, args.card_id))


_CARD_COMMANDS = {
    "new": _new,
    "amend": _amend,
    "verdict": _verdict,
    "reopen": _reopen,
    "list": _list,
    "show": _show,
}


def _scan(results_dir: Path, out: Path) -> int:
    """Write a new draft mapping and report the scanned run counts."""
    if out.exists():
        print(f"Error: {out} exists; refusing to overwrite a mapping", file=sys.stderr)
        return 1
    runs = _scan_runs(results_dir)
    # The scan never creates the register; without one no run went through the gate.
    research_dir = get_research_dir(create=False)
    if (research_dir / LEDGER_FILENAME).exists():
        # The same set apply_mapping expects in the mapping.
        runs = ungated_runs(runs, research_dir)
    draft = draft_mapping(runs)
    text = yaml.dump(draft, Dumper=_MappingDumper, sort_keys=False, allow_unicode=True)
    with open(out, "x", encoding="utf-8") as handle:
        handle.write(text)
    smoke = sum(1 for run in runs if run.kind is TrialKind.SMOKE)
    counted = sum(
        1 for run in runs if run.status is TrialStatus.OK and run.kind is not TrialKind.SMOKE
    )
    failed = sum(
        1 for run in runs if run.status is TrialStatus.FAILED and run.kind is not TrialKind.SMOKE
    )
    unresolved = sum(1 for run in runs if run.unresolved)
    print(
        f"{len(runs)} runs: {counted} counted, {failed} failed, {smoke} smoke, "
        f"{unresolved} unresolvable; {len(draft['cards'])} draft cards -> {out}"
    )
    return 0


def _apply(results_dir: Path, mapping_path: Path, research_dir: Path) -> int:
    """Load the reviewed mapping and criteria, apply them and report the writes."""
    template_path = research_dir / CRITERIA_TEMPLATE
    if not template_path.exists():
        print(
            f"Error: research register not initialised at {research_dir}; "
            "run 'fwbg research init'",
            file=sys.stderr,
        )
        return 1
    plans = load_mapping(mapping_path.read_text(encoding="utf-8"))
    runs = _scan_runs(results_dir)
    try:
        template = yaml.load(template_path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except (yaml.YAMLError, ValueError) as exc:
        # PyYAML raises a plain ValueError for impossible dates such as 2026-02-30.
        print(f"Error: {CRITERIA_TEMPLATE}: {exc}", file=sys.stderr)
        return 1
    if not isinstance(template, dict):
        print(f"Error: {CRITERIA_TEMPLATE} must be a mapping of criteria", file=sys.stderr)
        return 1
    result = apply_mapping(plans, runs, research_dir, template, date.today())
    print(
        f"{result.cards_written} cards written, {result.trials_appended} trials appended, "
        f"{'committed' if result.committed else 'nothing to commit'}"
    )
    return 0


def _scan_runs(results_dir: Path) -> list[LegacyRun]:
    """Scan legacy runs and warn about active or unreadable progress files."""
    runs = scan_runs(results_dir, get_strategies_dir().parent)
    active = _runs_in_progress(results_dir)
    if active:
        print(
            "Warning: these runs may still be in progress; stop the API and active runs "
            f"before applying: {', '.join(active)}",
            file=sys.stderr,
        )
    return runs


def _runs_in_progress(results_dir: Path) -> list[str]:
    """Run ids whose ``progress.json`` reports an active run or cannot be read."""
    active = []
    for progress in sorted(results_dir.glob("*/progress.json")):
        try:
            state = json.loads(progress.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = None
        # Stale "running" files exist, so this only warns; an unreadable file warns too.
        if not isinstance(state, dict) or str(state.get("status")) in ACTIVE_STATUSES:
            active.append(progress.parent.name)
    return active
