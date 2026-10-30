"""CLI handler for `fwbg analyze` — MFE/MAE exit analysis."""

import argparse
import os
import sys

from fwbg_sdk.enums import Timeframe

from fwbg.api.workspace import get_research_dir
from fwbg.core.config import StrategyConfig
from fwbg.exploration.exit_analyzer import analyze_asset, format_terminal_output, write_json
from fwbg.data import config as data_config
from fwbg.research.card import CARD_ID_RE, CardError, load_card
from fwbg.research.gate import Denial, DenialCode
from fwbg.research.runs import GATE_ENV, GateDenied, gate_enabled
from fwbg.research.study import register_trial


def run_analyze(argv):
    parser = argparse.ArgumentParser(
        description="Analyze optimal TP/SL ranges via MFE/MAE analysis",
        epilog="""
Every analysis needs --research-id and --note, reads only data before the
card's holdout cutoff and is recorded in the research register.

Examples:
  fwbg analyze BRENT_HOUR.csv --research-id S-0001 --note "TP/SL ranges"
  fwbg analyze BRENT_HOUR.csv --strategy-file strategies/configs/exploration_scalping.json \
      --research-id S-0001 --note "ranges for the scalping exits"
  fwbg analyze --asset-class COMMODITY --max-bars 96 --research-id S-0001 --note "..."
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("asset", nargs="?", help="Asset file (e.g. BRENT_HOUR.csv)")
    parser.add_argument(
        "--asset-class",
        help="Analyze all assets of a class (FOREX, INDEX, COMMODITY, CRYPTO)",
    )
    parser.add_argument(
        "--strategy-file", help="Strategy JSON: exit strategy, datasource and timeframe"
    )
    parser.add_argument(
        "--data-path", metavar="DIR", help="Data directory (overrides the strategy's datasource)"
    )
    parser.add_argument(
        "--max-bars", type=int, default=48, help="Forward window in bars (default: 48)"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: {workspace}/test_results/exploration)"
    )
    group = parser.add_argument_group("research register")
    group.add_argument("--research-id", metavar="S-NNNN", help="research card the analysis serves")
    group.add_argument("--note", help="one line: what the analysis is for")
    group.add_argument("--reason", help="why an identical analysis runs again")
    args = parser.parse_args(argv)

    if args.output_dir is None:
        from fwbg.results.storage import RESULTS_BASE_PATH
        args.output_dir = os.path.join(RESULTS_BASE_PATH, "exploration")

    if not args.asset and not args.asset_class:
        parser.error("Either asset file or --asset-class required")

    exit_strategy = "atr_based"
    exit_params = {"atr_period": 14}
    strategy = StrategyConfig()
    timeframe = data_config.TIMEFRAME
    if args.strategy_file:
        try:
            strategy = StrategyConfig.from_json_file(args.strategy_file)
            if strategy.timeframe:
                timeframe = Timeframe.from_str(strategy.timeframe).canonical
        except (OSError, ValueError) as exc:
            parser.error(f"--strategy-file {args.strategy_file}: {exc}")
        exit_strategy, exit_params = _exit_strategy(strategy, exit_strategy, exit_params)

    data_config.use_data_path(strategy.datasource, args.data_path)
    files = _resolve_files(args.asset, args.asset_class, timeframe)
    window = {"start_date": strategy.start_date, "end_date": strategy.end_date}
    if gate_enabled():
        if not args.research_id:
            parser.error("--research-id S-NNNN is required: the register must know which data "
                         "shaped a strategy")
        config = {
            "analysis": "mfe_mae",
            "exit_strategies": [{"name": exit_strategy, "params": exit_params}],
            "max_bars": args.max_bars,
            "timeframe": _file_timeframe(files[0], timeframe),
            "datasource": strategy.datasource,
            "_cli_data_path": os.path.abspath(args.data_path) if args.data_path else None,
            **window,
        }
        window = _record_analysis(
            args, {key: value for key, value in config.items() if value is not None}, files
        )
    else:
        print(f"Warning: {GATE_ENV}=off - this analysis is neither gated nor recorded",
              file=sys.stderr)
    os.makedirs(args.output_dir, exist_ok=True)

    for data_file in files:
        symbol = os.path.basename(data_file).replace(".csv", "")
        print(f"\n{'=' * 60}")
        print(f"  Analyzing {symbol}")
        print(f"{'=' * 60}")

        result = analyze_asset(
            data_file, exit_strategy, exit_params, args.max_bars,
            window["start_date"], window["end_date"],
        )
        print(format_terminal_output(result))

        out_path = os.path.join(args.output_dir, f"{symbol}.json")
        write_json(result, out_path)
        print(f"\n  -> {out_path}")


def _record_analysis(args, config, files):
    """Put the analysis on record before it reads data; return its window.

    Like a gated run, the analysis is recorded before it reads any data, so a
    refusal leaves the data unseen, and an open-ended window ends at the card's
    holdout cutoff (end exclusive). It is an iteration without metrics: it
    evaluates no strategy, so it adds no evaluation to the card's trial count,
    but its configuration and window are on record and a repeat needs --reason.
    Exits with 2 if the register refuses it.
    """
    symbols = [_file_symbol(path) for path in files]
    try:
        if "end_date" not in config:
            config["end_date"] = _holdout_cutoff(args.research_id)
        trial = register_trial(
            args.research_id, config, symbols, {}, args.note, reason=args.reason
        )
    except GateDenied as exc:
        print("Research gate refused the analysis:", file=sys.stderr)
        for denial in exc.denials:
            print(f"  - [{denial.code.value}] {denial.message}", file=sys.stderr)
        raise SystemExit(2) from None
    print(f"Research: recorded as {trial.trial_id} of {trial.strategy}; "
          f"data before {config['end_date']} (end exclusive)")
    return {"start_date": config.get("start_date"), "end_date": config["end_date"]}


def _holdout_cutoff(research_id):
    """The card's holdout cutoff as YYYY-MM-DD; raises GateDenied without a readable card."""
    try:
        card = None
        if CARD_ID_RE.fullmatch(research_id):
            card = load_card(get_research_dir(create=False), research_id)
    except (OSError, CardError) as exc:
        denial = Denial(DenialCode.REGISTER_UNREADABLE, f"research register: {exc}")
        raise GateDenied([denial]) from exc
    if card is None:
        raise GateDenied([Denial(DenialCode.CARD_NOT_FOUND, f"no research card {research_id}")])
    return card.holdout_cutoff.isoformat()


def _file_symbol(path):
    return os.path.basename(path).removesuffix(".csv").partition("_")[0].upper()


def _file_timeframe(path, default):
    """The timeframe a ``SYMBOL_TIMEFRAME.csv`` file name states, else *default*."""
    return os.path.basename(path).removesuffix(".csv").partition("_")[2] or default


def _exit_strategy(strategy, default_name, default_params):
    """Name and params of the strategy's first exit strategy, else the defaults."""
    if not strategy.exit_strategies:
        return default_name, default_params
    first = strategy.exit_strategies[0]
    if len(strategy.exit_strategies) > 1:
        print(
            f"Note: the strategy has {len(strategy.exit_strategies)} exit strategies; "
            f"analyzing with the first ({first.name})",
            file=sys.stderr,
        )
    return first.name, dict(first.params)


def _resolve_files(asset, asset_class, timeframe):
    """Resolve data file paths from asset name or asset class."""
    if asset:
        # Try direct/absolute path first (no DATA_PATH needed)
        if os.path.isabs(asset) and os.path.exists(asset):
            return [asset]
        if os.path.exists(asset):
            return [asset]
        # Try under DATA_PATH
        if data_config.DATA_PATH:
            path = os.path.join(data_config.DATA_PATH, asset)
            if os.path.exists(path):
                return [path]
        raise FileNotFoundError(f"Data file not found: {asset}")

    if not data_config.DATA_PATH:
        raise FileNotFoundError("DATA_PATH not configured — set 'datasource' in strategy or use --data-path")

    from fwbg.data.assets import AssetRegistry

    registry = AssetRegistry()
    symbols = registry.symbols_by_class(asset_class.upper())
    files = []
    for sym in symbols:
        path = os.path.join(data_config.DATA_PATH, f"{sym}_{timeframe}.csv")
        if os.path.exists(path):
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No data files for asset class {asset_class}")
    return sorted(files)
