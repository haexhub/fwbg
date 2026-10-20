"""``fwbg research``: set up the research register and backfill legacy runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import yaml

from fwbg.api.run_service import ACTIVE_STATUSES
from fwbg.api.workspace import get_research_dir, get_strategies_dir, get_test_results_dir
from fwbg.research.backfill import apply_mapping
from fwbg.research.card import CardError, UniqueKeyLoader
from fwbg.research.ledger import LedgerError, TrialKind, TrialStatus
from fwbg.research.legacy import LegacyRun, LegacyRunError, scan_runs
from fwbg.research.mapping import BackfillError, draft_mapping, load_mapping
from fwbg.research.repo import CRITERIA_TEMPLATE, ResearchRepoError, init_research_repo

_OPERATOR_ERRORS = (
    BackfillError,
    CardError,
    LedgerError,
    LegacyRunError,
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


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch research commands and return a nonzero status for operator errors."""
    parser = argparse.ArgumentParser(
        prog="fwbg research",
        description="Set up the research register and backfill legacy runs.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="create the research register (idempotent)")
    backfill = commands.add_parser("backfill", help="import legacy runs and studies")
    actions = backfill.add_subparsers(dest="action", required=True)
    scan = actions.add_parser("scan", help="write a draft mapping of legacy runs to cards")
    scan.add_argument("--out", type=Path, required=True, help="mapping file to create")
    apply = actions.add_parser("apply", help="write the cards and trials of a mapping")
    apply.add_argument("mapping", type=Path, help="reviewed mapping file")
    for sub in (scan, apply):
        sub.add_argument("--results", type=Path, help="run directory root (default: workspace)")
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            research_dir = get_research_dir()
            created = init_research_repo(research_dir)
            print(f"{'Created' if created else 'Already initialised'}: {research_dir}")
            return 0
        results_dir = args.results or get_test_results_dir()
        if args.action == "scan":
            return _scan(results_dir, args.out)
        return _apply(results_dir, args.mapping, get_research_dir())
    except _OPERATOR_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _scan(results_dir: Path, out: Path) -> int:
    """Write a new draft mapping and report the scanned run counts."""
    if out.exists():
        print(f"Error: {out} exists; refusing to overwrite a mapping", file=sys.stderr)
        return 1
    runs = _scan_runs(results_dir)
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
