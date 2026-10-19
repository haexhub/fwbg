"""``fwbg research``: set up the research register and backfill legacy runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import yaml

from fwbg.api.workspace import get_research_dir, get_strategies_dir, get_test_results_dir
from fwbg.research.backfill import BackfillError, apply_mapping, draft_mapping, parse_mapping
from fwbg.research.card import CardError
from fwbg.research.ledger import LedgerError
from fwbg.research.legacy import LegacyRun, LegacyRunError, scan_runs
from fwbg.research.repo import CRITERIA_TEMPLATE, ResearchRepoError, init_research_repo


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fwbg research", description=__doc__)
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

    research_dir = get_research_dir()
    try:
        if args.command == "init":
            created = init_research_repo(research_dir)
            print(f"{'Created' if created else 'Already initialised'}: {research_dir}")
            return 0
        runs = scan_runs(args.results or get_test_results_dir(), get_strategies_dir().parent)
        if args.action == "scan":
            return _scan(runs, args.out)
        return _apply(runs, args.mapping, research_dir)
    except (BackfillError, CardError, LedgerError, LegacyRunError, ResearchRepoError) as exc:
        print(f"Error: {exc}")
        return 1


def _scan(runs: list[LegacyRun], out: Path) -> int:
    if out.exists():
        print(f"Error: {out} exists; refusing to overwrite a mapping")
        return 1
    out.write_text(
        yaml.safe_dump(draft_mapping(runs), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    unresolved = sum(1 for run in runs if run.unresolved)
    counted = sum(1 for run in runs if run.status.value == "ok" and run.kind.value == "legacy")
    print(f"{len(runs)} runs ({counted} counted, {unresolved} unresolvable) -> {out}")
    return 0


def _apply(runs: list[LegacyRun], mapping_path: Path, research_dir: Path) -> int:
    plans = parse_mapping(yaml.safe_load(mapping_path.read_text(encoding="utf-8")))
    template = yaml.safe_load((research_dir / CRITERIA_TEMPLATE).read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        print(f"Error: {CRITERIA_TEMPLATE} must be a mapping of criteria")
        return 1
    result = apply_mapping(plans, runs, research_dir, template, date.today())
    print(
        f"{result.cards_written} cards written, {result.trials_appended} trials appended, "
        f"{'committed' if result.committed else 'nothing to commit'}"
    )
    return 0
