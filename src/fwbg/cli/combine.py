"""``fwbg combine``: a gated portfolio run over frozen strategy sleeves (docs/portfolio-combine.md)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fwbg.api.workspace import get_research_dir
from fwbg.cli.commands import load_strategy_from_file
from fwbg.data import config as data_config
from fwbg.optimization.sleeves import PortfolioSpecError, run_portfolio
from fwbg.results import storage as results_storage
from fwbg.research.runs import (
    GATE_ENV,
    ResearchRun,
    add_run_arguments,
    gate_enabled,
    prepare_metadata,
    run_gated,
)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="fwbg combine",
        description="Combine frozen strategy sleeves into one portfolio run (inverse-volatility weights)",
    )
    parser.add_argument(
        "--spec", required=True, help="portfolio spec (JSON) naming the sleeve strategy files"
    )
    parser.add_argument("--run-id", help="run id (default: generated)")
    parser.add_argument("--description", help="free-text run description")
    add_run_arguments(parser)
    args = parser.parse_args(argv)

    run_id = args.run_id or results_storage.generate_run_id(args.description)

    def prepare(metadata):
        return prepare_metadata(metadata, data_config.TIMEFRAME)

    def start(before_run=None):
        try:
            return run_portfolio(
                args.spec,
                load_strategy=load_strategy_from_file,
                prepare=prepare,
                run_id=run_id,
                before_run=before_run,
                description=args.description,
            )
        except PortfolioSpecError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return None

    if gate_enabled():
        research = ResearchRun(
            get_research_dir(),
            run_id,
            (Path(results_storage.RESULTS_BASE_PATH) / run_id).resolve(),
            research_id=args.research_id,
            note=args.note,
            reason=args.reason,
            check_only=args.check or args.check_json,
            holdout=args.holdout,
        )
        result = run_gated(research, start, as_json=args.check_json)
    else:
        if args.holdout or args.check or args.check_json:
            parser.error(f"--holdout/--check need the research gate ({GATE_ENV} is off)")
        print(f"Warning: {GATE_ENV}=off - this run is neither gated nor recorded", file=sys.stderr)
        result = start()
    if result is None:
        sys.exit(1)
