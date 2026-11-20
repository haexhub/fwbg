"""Research-register gate for the portfolio commands (``run``, ``search``, ``swing-study``).

Like ``fwbg analyze``, each command is put on record before it reads price data
and sees only data before the card's holdout cutoff. ``run`` and ``search`` read
an open-ended price history, so their window is clamped to the cutoff;
``swing-study`` states its window in the protocol, so a protocol that reaches
past the cutoff is refused (clamping it would change the frozen design).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from fwbg.research.analysis import record_analysis
from fwbg.research.runs import GATE_ENV, GateDenied, gate_enabled


def add_research_arguments(parser: argparse.ArgumentParser) -> None:
    """The research flags every gated portfolio command takes."""
    group = parser.add_argument_group("research register")
    group.add_argument(
        "--research-id", metavar="S-NNNN", help="research card this run serves"
    )
    group.add_argument(
        "--note", help="one line: what changed since the last iteration and why"
    )
    group.add_argument("--reason", help="why an identical configuration runs again")


def gate_portfolio_command(args: argparse.Namespace) -> str | None:
    """Record the command and return the exclusive end date its data may reach.

    Returns ``None`` when the gate is switched off (test suite only) or the
    command's own window already ends before the cutoff.

    Raises:
        SystemExit: code 2 when the register refuses the command or the inputs
            needed to describe it cannot be read.
    """
    if not gate_enabled():
        print(
            f"Warning: {GATE_ENV}=off - this portfolio run is neither gated nor recorded",
            file=sys.stderr,
        )
        return None
    try:
        config, symbols = _describe(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            f"Portfolio error: cannot describe the run for the register: {error}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    try:
        trial, window = record_analysis(
            args.research_id, config, symbols, args.note, reason=args.reason
        )
    except GateDenied as exc:
        print("Research gate refused the portfolio run:", file=sys.stderr)
        for denial in exc.denials:
            print(f"  - [{denial.code.value}] {denial.message}", file=sys.stderr)
        raise SystemExit(2) from None
    print(
        f"Research: recorded as {trial.trial_id} of {trial.strategy}; "
        f"data before {window['end_date']} (end exclusive)"
    )
    return None if args.command == "swing-study" else window["end_date"]


def _describe(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    """The configuration the register fingerprints, and the symbols the command evaluates."""
    if args.command == "run":
        config_path: Path = args.config
        config = _read_json(config_path)
        digest = _file_digest(config_path.parent / config["prices"])
        described = {
            "analysis": "portfolio_run",
            "timeframe": "DAY",
            "config": _without_name(config),
            "cost_multiplier": args.cost_multiplier,
            "prices_sha256": digest,
        }
        return described, _symbols(config)
    if args.command == "search":
        protocol = _read_json(args.protocol)
        base_path = args.protocol.parent / protocol["base_config"]
        base = _read_json(base_path)
        described = {
            "analysis": "portfolio_search",
            "timeframe": "DAY",
            "protocol": protocol,
            "base_config": _without_name(base),
            "prices_sha256": _file_digest(base_path.parent / base["prices"]),
        }
        return described, _symbols(base)
    protocol = _read_json(args.protocol)
    manifest = (
        args.protocol.parent / protocol["data_directory"]
    ).resolve() / "manifest.json"
    described = {
        "analysis": "portfolio_swing_study",
        "timeframe": "DAY",
        "protocol": protocol,
        "data_manifest_sha256": _file_digest(manifest),
        "start_date": protocol["warmup_start"],
        # The protocol slices bars with .loc[..evaluation_end], inclusive; the gate's end is exclusive.
        "end_date": _day_after(protocol["evaluation_end"]),
    }
    return described, [market["symbol"] for market in protocol["markets"]]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _without_name(config: Mapping[str, Any]) -> dict[str, Any]:
    """Renaming a config must not make it look new."""
    return {key: value for key, value in config.items() if key != "name"}


def _symbols(config: Mapping[str, Any]) -> list[str]:
    weights = config["weights"]
    if not isinstance(weights, Mapping):
        raise ValueError("weights must be an object mapping assets to weights")
    return [str(asset) for asset in weights]


def _day_after(value: object) -> object:
    """``value`` plus one day as YYYY-MM-DD; anything unparsable is left for the gate to refuse."""
    if not isinstance(value, str):
        return value
    try:
        return (pd.Timestamp(value).date() + timedelta(days=1)).isoformat()
    except (ValueError, TypeError):
        return value


def clamp_prices(prices: pd.DataFrame, end_date: str | None) -> pd.DataFrame:
    """Prices before *end_date* (exclusive); all of them when it is ``None``."""
    if end_date is None:
        return prices
    return prices.loc[prices.index < pd.Timestamp(end_date)]


__all__: Sequence[str] = (
    "add_research_arguments",
    "clamp_prices",
    "gate_portfolio_command",
)
