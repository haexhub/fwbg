"""CLI and auditable artifacts for monthly portfolio experiments."""

import argparse
import json
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import load_inputs
from .engine import run_portfolio, summarize
from .robustness import assess_robustness, format_robustness


def _metrics_table(metrics):
    """Format portfolio performance metrics as a Markdown table."""
    rows = [
        "| Portfolio | CAGR | Max. drawdown | Volatility | Sharpe vs cash |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        sharpe = values["sharpe_excess_cash"]
        rows.append(
            f"| {name} | {values['cagr']:.2%} | {values['max_drawdown']:.2%} | "
            f"{values['annual_volatility']:.2%} | {sharpe:.2f} |"
            if sharpe is not None
            else f"| {name} | {values['cagr']:.2%} | {values['max_drawdown']:.2%} | "
            f"{values['annual_volatility']:.2%} | — |"
        )
    return "\n".join(rows)


def run_experiment(config_path: Path, output: Path, cost_multiplier: float = 1.0) -> dict:
    """Run a configured monthly backtest and write artifacts to a new directory.

    Scale transaction costs by a positive finite multiplier and return the
    report containing performance metrics, robustness checks and provenance.
    """
    import math

    if not math.isfinite(cost_multiplier) or cost_multiplier <= 0:
        raise ValueError("cost-multiplier must be positive and finite")
    config, settings, prices, provenance = load_inputs(config_path)
    settings = replace(settings, one_way_cost_bps=settings.one_way_cost_bps * cost_multiplier)
    result = run_portfolio(prices, settings)
    robustness, monthly, folds = assess_robustness(result, prices, settings)
    metrics = {"full": summarize(result.equity)}
    if config.get("split_date"):
        split = pd.Timestamp(config["split_date"])
        before = result.equity.loc[result.equity.index < split]
        after = result.equity.loc[result.equity.index >= split]
        if len(before) < 3 or len(after) < 2:
            raise ValueError("split_date must leave at least two return observations on each side")
        metrics["before_split"] = summarize(before)
        # Preserve the existing portfolio and the first return after the split.
        metrics["after_split"] = summarize(pd.concat([before.iloc[-1:], after]))
    output.mkdir(parents=True, exist_ok=False)
    result.equity.to_csv(output / "equity.csv")
    result.weights.to_csv(output / "weights.csv", index=False)
    result.signals.to_csv(output / "signals.csv", index=False)
    result.trades.to_csv(output / "trades.csv", index=False)
    monthly.to_csv(output / "monthly_returns.csv")
    folds.to_csv(output / "robustness_folds.csv", index=False)
    (output / "robustness.json").write_text(
        json.dumps(robustness, indent=2, allow_nan=False) + "\n"
    )
    # Aggregate actual return observations, excluding an initial valuation-only
    # year (e.g. the 2005-12-30 baseline before the 2006 experiment starts).
    daily_growth = result.equity.div(result.equity.shift()).iloc[1:]
    (daily_growth.resample("YE").prod() - 1).to_csv(output / "annual_returns.csv")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "effective_one_way_cost_bps": settings.one_way_cost_bps,
        "cost_multiplier": cost_multiplier,
        "cash_return": 0.0,
        "robustness": robustness,
        "provenance": provenance,
        "metrics": metrics,
        "trade_costs": result.trades.groupby("portfolio")["cost"].sum().to_dict(),
        "status": "historical_research_only",
    }
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    paragraphs = [
        f"# {config['name']}",
        f"Currency: {config['base_currency']}. Valuation period: "
        f"{result.equity.index[0].date()} to {result.equity.index[-1].date()}. "
        f"First execution: {result.equity.index[1].date()}.",
        f"One-way costs: {settings.one_way_cost_bps:g} bps per traded notional. "
        "Uninvested cash return: 0% (no interest or cash-yield assumptions).",
        "Completed month-end signals; execution at next session close. Daily mark-to-market, "
        "fractional holdings, no leverage. Static rebalances monthly; equity_buy_hold trades "
        "once. Open positions remain marked at the end (no assumed liquidation).",
    ]
    for segment, values in metrics.items():
        paragraphs.extend([f"## {segment}", _metrics_table(values)])
    paragraphs.append(format_robustness(robustness))
    paragraphs.extend(
        [
            "## Interpretation",
            "Historical comparison with conditional statistical checks, not a live-trading "
            "recommendation or complete correction for selection bias. The chronological split does not remove publication/selection bias. "
            "No parameter search is performed. Compare both return and drawdown against static allocation.",
            "Trading costs are assumptions, not broker quotes. No taxes or CFD funding are modeled. "
            "Adjusted prices already include distributions and fund expenses. Uninvested cash "
            "earns zero. FX reference conversions are valuation "
            "proxies, not simultaneous executable quotes. See input provenance in results.json.",
            "The first/last calendar years may be partial. Results are mark-to-market in the base "
            "currency. Historical data revisions and the chosen surviving ETF proxies limit inference.",
        ]
    )
    (output / "report.md").write_text("\n\n".join(paragraphs) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    (result.equity / result.equity.iloc[0]).plot(ax=axes[0], logy=True)
    axes[0].set_ylabel("Growth of 1 (log scale)")
    (result.equity / result.equity.cummax() - 1).plot(ax=axes[1], legend=False)
    axes[1].set_ylabel("Drawdown")
    fig.suptitle(f"{config['name']} — {config['base_currency']}, net of assumed costs")
    fig.tight_layout()
    fig.savefig(output / "equity.png", dpi=150)
    plt.close(fig)
    print(_metrics_table(metrics["full"]))
    print(format_robustness(robustness))
    print(f"\nArtifacts: {output.resolve()}")
    return report


def main(argv: list[str] | None = None) -> None:
    """Parse portfolio subcommands and report supported errors with exit code 2."""
    parser = argparse.ArgumentParser(description="FWBG monthly portfolio research")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run a frozen experiment from local CSV inputs")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path)
    run.add_argument("--cost-multiplier", type=float, default=1.0)
    search = commands.add_parser(
        "search", help="Compare a predeclared candidate set with selection checks"
    )
    search.add_argument("--protocol", type=Path, required=True)
    search.add_argument("--output", type=Path, required=True)
    swing = commands.add_parser("swing-study", help="Run the frozen daily entry/exit study")
    swing.add_argument("--protocol", type=Path, required=True)
    swing.add_argument("--output", type=Path, required=True)
    download = commands.add_parser(
        "download-example", help="Download the five-asset Yahoo/ECB bundle"
    )
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--start", default="2004-01-01")
    download.add_argument("--end", required=True, help="Exclusive end date; not later than today")
    args = parser.parse_args(argv)
    try:
        if args.command == "download-example":
            from .download import prepare_example

            metadata = prepare_example(args.output, args.start, args.end)
            print(
                f"Prepared {metadata['common_start']} to {metadata['common_end']} in {args.output}"
            )
        elif args.command == "swing-study":
            from .swing_study import run_swing_study

            run_swing_study(args.protocol, args.output)
        elif args.command == "search":
            from .search import run_search

            run_search(args.protocol, args.output)
        else:
            output = args.output or Path("test_results/portfolio") / (
                datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            )
            run_experiment(args.config, output, args.cost_multiplier)
    except (ValueError, OSError, ImportError, KeyError, TypeError) as error:
        parser.exit(2, f"Portfolio error: {error}\n")


if __name__ == "__main__":
    main()
