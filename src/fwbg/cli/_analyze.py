"""CLI handler for `fwbg analyze` — MFE/MAE exit analysis."""

import argparse
import json
import os

from fwbg.exploration.exit_analyzer import analyze_asset, format_terminal_output, write_json
from fwbg.data import config as data_config


def run_analyze(argv):
    parser = argparse.ArgumentParser(
        description="Analyze optimal TP/SL ranges via MFE/MAE analysis",
        epilog="""
Examples:
  fwbg analyze BRENT_HOUR.csv
  fwbg analyze BRENT_HOUR.csv --strategy-file strategies/configs/exploration_scalping.json
  fwbg analyze --asset-class COMMODITY --max-bars 96
        """,
    )
    parser.add_argument("asset", nargs="?", help="Asset file (e.g. BRENT_HOUR.csv)")
    parser.add_argument(
        "--asset-class",
        help="Analyze all assets of a class (FOREX, INDEX, COMMODITY, CRYPTO)",
    )
    parser.add_argument("--strategy-file", help="Strategy JSON for exit strategy config")
    parser.add_argument(
        "--max-bars", type=int, default=48, help="Forward window in bars (default: 48)"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: {workspace}/test_results/exploration)"
    )
    args = parser.parse_args(argv)

    if args.output_dir is None:
        from fwbg.results.storage import RESULTS_BASE_PATH
        args.output_dir = os.path.join(RESULTS_BASE_PATH, "exploration")

    if not args.asset and not args.asset_class:
        parser.error("Either asset file or --asset-class required")

    # Load exit strategy config from strategy file
    exit_strategy = "atr_based"
    exit_params = {"atr_period": 14}
    if args.strategy_file:
        with open(args.strategy_file) as f:
            strat = json.load(f)
        exit_strategy = strat.get("exit_strategy", "atr_based")
        exit_params = strat.get("exit_params", {})

    files = _resolve_files(args.asset, args.asset_class)
    os.makedirs(args.output_dir, exist_ok=True)

    for data_file in files:
        symbol = os.path.basename(data_file).replace(".csv", "")
        print(f"\n{'=' * 60}")
        print(f"  Analyzing {symbol}")
        print(f"{'=' * 60}")

        result = analyze_asset(data_file, exit_strategy, exit_params, args.max_bars)
        print(format_terminal_output(result))

        out_path = os.path.join(args.output_dir, f"{symbol}.json")
        write_json(result, out_path)
        print(f"\n  -> {out_path}")


def _resolve_files(asset, asset_class):
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
        path = os.path.join(data_config.DATA_PATH, f"{sym}_{data_config.TIMEFRAME}.csv")
        if os.path.exists(path):
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No data files for asset class {asset_class}")
    return sorted(files)
