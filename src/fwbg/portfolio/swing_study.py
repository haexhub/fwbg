"""Run the frozen daily entry/exit study, retaining every trial and stress."""

import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from inspect import getfile
from pathlib import Path

import numpy as np
import pandas as pd

from fwbg.optimization.overfitting import probability_of_backtest_overfitting
from fwbg.optimization.robust_validation import create_walk_forward_folds

from .data import file_fingerprint
from .engine import summarize
from .robustness import NATIVE_SOURCES, _check_series, monthly_returns
from .search import holm_adjust
from .swing import SwingSettings, indicators, run_swing, validate_bars


def metrics(equity: pd.Series) -> dict:
    """Summarize strategy equity against a constant cash benchmark earning zero."""
    return summarize(pd.DataFrame({"strategy": equity, "cash": 10000.0}))["strategy"]


def worst_p(checks: list[dict]) -> float:
    """Return the largest sign-test p-value, respecting exact block-test floors.

    Unevaluated checks return 1.0 so missing evidence cannot pass a gate.
    """
    values = []
    for check in checks:
        if check.get("status") != "evaluated":
            return 1.0
        values.append(check["monte_carlo"]["p_value"])
        for block in check["block_sign_permutations"].values():
            # A finite MC estimate must not beat the smallest attainable exact
            # sign-test p-value; this matters with only ten half-year blocks.
            values.append(max(block["p_value"], 2.0 ** -block["observations"]))
    return float(max(values))


def run_swing_study(protocol_path: Path, output: Path) -> dict:
    """Run the frozen daily study on fingerprinted data and save every trial.

    Compare matched buy-and-hold benchmarks across cost/leverage scenarios,
    check historical prefixes, and return the summary with PBO and Holm gates.
    """
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol["schema_version"] != 1
        or protocol["cash_return"] != 0
        or protocol["base_currency"] != "USD"
    ):
        raise ValueError("expected schema 1, USD and zero cash return")
    if protocol["initial_capital"] != 10000 or protocol["leverage"] != [1.0, 1.5]:
        raise ValueError("unexpected capital/leverage settings")
    expected_parameters = dict(
        rsi_period=2,
        rsi_entry=10,
        rsi_exit=70,
        rsi_trend_sma=200,
        rsi_exit_sma=5,
        rsi_time_exit_sessions=10,
        breakout_entry_sessions=55,
        breakout_exit_sessions=20,
        ema_fast=20,
        ema_medium=50,
        ema_slow=200,
        atr_period=14,
        trailing_atr_multiple=3,
    )
    if protocol["parameters"] != expected_parameters:
        raise ValueError("parameters differ from implemented frozen rule definitions")
    data_dir = (protocol_path.parent / protocol["data_directory"]).resolve()
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    manifest: dict = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": file_fingerprint(protocol_path),
        "data_manifest": file_fingerprint(data_dir / "manifest.json"),
        "implementation": {p.name: file_fingerprint(p) for p in Path(__file__).parent.glob("*.py")},
        "native_implementations": {
            key: file_fingerprint(path) for key, path in NATIVE_SOURCES.items()
        },
        "indicator_source": file_fingerprint(Path(getfile(indicators))),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validation = protocol["validation"]
    results: dict = {}
    return_sets: dict = {}
    comparisons, fold_rows = [], []
    bars_by_market = {}
    for market in protocol["markets"]:
        symbol = market["symbol"]
        if symbol not in data_manifest["markets"]:
            raise ValueError(
                f"missing declared market {symbol}; report data exclusions before changing trial set"
            )
        path = data_dir / symbol / "bars.csv"
        expected_hash = data_manifest["markets"][symbol]["files"]["bars.csv"]["sha256"]
        if file_fingerprint(path)["sha256"] != expected_hash:
            raise ValueError(f"input snapshot changed: {symbol}")
        bars = pd.read_csv(path, index_col=0, parse_dates=True).loc[
            protocol["warmup_start"] : protocol["evaluation_end"]
        ]
        validate_bars(bars)
        bars_by_market[symbol] = bars
        indicators(bars).to_csv(output / f"{symbol}_indicators.csv")
    for scenario in protocol["scenarios"]:
        scenario_id = scenario["id"]
        return_sets[scenario_id] = {}
        for market in protocol["markets"]:
            symbol = market["symbol"]
            bars = bars_by_market[symbol]
            print(f"Running {scenario_id}: {symbol}, six rules x two leverage levels", flush=True)
            for leverage in protocol["leverage"]:
                settings = SwingSettings(
                    "buy_hold",
                    start=protocol["evaluation_start"],
                    end=protocol["evaluation_end"],
                    capital=protocol["initial_capital"],
                    leverage=leverage,
                    cost_bps=market["cost_bps"],
                    cost_multiplier=scenario["cost_multiplier"],
                    borrow_rate=scenario["borrow_rate"],
                )
                benchmark = run_swing(bars, settings)
                benchmark_path = output / f"{symbol}_buy_hold_L{leverage:g}_{scenario_id}"
                benchmark_path.mkdir()
                benchmark.equity.to_csv(benchmark_path / "equity.csv")
                benchmark.ledger.to_csv(benchmark_path / "ledger.csv", index=False)
                benchmark.positions.to_csv(benchmark_path / "positions.csv", index=False)
                for rule in protocol["rules"]:
                    key = f"{symbol}_{rule}_L{leverage:g}"
                    settings = replace(settings, rule=rule)
                    result = run_swing(bars, settings)
                    folder = output / f"{key}_{scenario_id}"
                    folder.mkdir()
                    result.equity.to_csv(folder / "equity.csv")
                    result.ledger.to_csv(folder / "ledger.csv", index=False)
                    result.positions.to_csv(folder / "positions.csv", index=False)
                    result.signals.to_csv(folder / "signals.csv", index=False)
                    if result.bankrupt or benchmark.bankrupt:
                        (folder / "failure.json").write_text(
                            json.dumps(
                                {
                                    "bankrupt": result.bankrupt,
                                    "benchmark_bankrupt": benchmark.bankrupt,
                                }
                            )
                        )
                        raise ValueError(
                            f"bankruptcy in {key}; preserved failure, do not discard this trial"
                        )
                    paired = pd.DataFrame(
                        {"strategy": result.equity, "benchmark": benchmark.equity}
                    )
                    monthly = monthly_returns(paired)
                    if len(monthly) != 84:
                        raise ValueError(f"{key}: expected 84 complete months, got {len(monthly)}")
                    monthly.to_csv(folder / "monthly_returns.csv")
                    return_sets[scenario_id][key] = monthly.strategy
                    test_size = (len(monthly) - validation["initial_months"]) // validation["folds"]
                    folds = create_walk_forward_folds(
                        monthly,
                        n_folds=validation["folds"],
                        test_size=test_size,
                        min_train_size=validation["initial_months"],
                        anchored=True,
                        embargo_bars=0,
                    )
                    test = pd.concat([fold.test_df for fold in folds])
                    checks = {}
                    for label, frame in (("full", monthly), ("chronological", test)):
                        checks[label] = {
                            "profit": _check_series(
                                frame.strategy,
                                permutations=validation["permutations"],
                                seed=validation["seed"],
                            ),
                            "active": _check_series(
                                frame.strategy,
                                benchmark=frame.benchmark,
                                permutations=validation["permutations"],
                                seed=validation["seed"],
                            ),
                        }
                    prefix_ok = True
                    positive = 0
                    for fold in folds:
                        begin = fold.test_df.index[0].to_period("M").start_time
                        end = fold.test_df.index[-1]
                        segment = pd.concat(
                            [
                                result.equity.loc[result.equity.index < begin].iloc[-1:],
                                result.equity.loc[begin:end],
                            ]
                        )
                        prefix = run_swing(
                            bars, replace(settings, end=str(segment.index[-1].date()))
                        )
                        invariant = (
                            prefix.equity.equals(result.equity.loc[: segment.index[-1]])
                            and prefix.ledger.equals(
                                result.ledger.loc[result.ledger.date <= segment.index[-1]]
                            )
                            and prefix.signals.equals(
                                result.signals.loc[result.signals.date <= segment.index[-1]]
                            )
                        )
                        prefix_ok = prefix_ok and invariant
                        values = metrics(segment)
                        positive += values["total_return"] > 0
                        fold_rows.append(
                            {
                                "candidate": key,
                                "scenario": scenario_id,
                                "fold": fold.fold_id,
                                "prefix_invariant": invariant,
                                **values,
                            }
                        )
                    split = pd.Timestamp(protocol["split_date"])
                    before = result.equity.loc[result.equity.index < split].iloc[-1:]
                    later = pd.concat([before, result.equity.loc[split:]])
                    full_metrics, later_metrics = metrics(result.equity), metrics(later)
                    raw_profit = worst_p([checks[label]["profit"] for label in checks])
                    raw_active = worst_p([checks[label]["active"] for label in checks])
                    report: dict = {
                        "settings": asdict(settings),
                        "metrics": full_metrics,
                        "later_metrics": later_metrics,
                        "benchmark_metrics": metrics(benchmark.equity),
                        "checks": checks,
                        "prefix_invariant": prefix_ok,
                        "profit_p": raw_profit,
                        "active_p": raw_active,
                        "funding_paid": float(result.positions.funding.sum()),
                        "commissions_paid": float(result.ledger.commission.sum()),
                        "spread_cost": float(result.ledger.spread_cost.sum()),
                    }
                    (folder / "results.json").write_text(
                        json.dumps(report, indent=2, allow_nan=False) + "\n"
                    )
                    results[key, scenario_id] = report
                    comparisons.append(
                        {
                            "candidate": key,
                            "market": symbol,
                            "rule": rule,
                            "leverage": leverage,
                            "scenario": scenario_id,
                            **full_metrics,
                            "later_cagr": later_metrics["cagr"],
                            "later_drawdown": later_metrics["max_drawdown"],
                            "benchmark_cagr": report["benchmark_metrics"]["cagr"],
                            "benchmark_drawdown": report["benchmark_metrics"]["max_drawdown"],
                            "positive_folds": int(positive),
                            "entries": int(result.ledger.reason.eq("entry").sum()),
                            "invested_fraction": float(result.positions.quantity.gt(0).mean()),
                            "funding_paid": report["funding_paid"],
                            "profit_p": raw_profit,
                            "active_p": raw_active,
                        }
                    )
    pd.DataFrame(fold_rows).to_csv(output / "folds.csv", index=False)
    pbo = {}
    matrices = {scenario: pd.DataFrame(values) for scenario, values in return_sets.items()}
    for scenario, frame in matrices.items():
        frame.to_csv(output / f"all_monthly_returns_{scenario}.csv")
        for market in protocol["markets"]:
            symbol = market["symbol"]
            subset = frame[[key for key in frame if key.startswith(symbol + "_")]].iloc[-60:]
            matrix = np.stack(
                [subset.iloc[i * 10 : (i + 1) * 10].mean().to_numpy() for i in range(6)], axis=1
            )
            pd.DataFrame(matrix, index=subset.columns).to_csv(
                output / f"pbo_{symbol}_{scenario}.csv"
            )
            pbo[symbol, scenario] = probability_of_backtest_overfitting(matrix)
    keys = list(matrices["base"].columns)
    scenario_ids = [scenario["id"] for scenario in protocol["scenarios"]]
    raw = {
        f"{key}:{kind}": max(results[key, scenario][f"{kind}_p"] for scenario in scenario_ids)
        for key in keys
        for kind in ("profit", "active")
    }
    adjusted = holm_adjust(raw)
    gates = {}
    for key in keys:
        trials = [results[key, scenario] for scenario in scenario_ids]
        market = key.split("_")[0]
        target = all(
            r[segment]["cagr"] > validation["minimum_cagr"]
            and r[segment]["max_drawdown"] >= -validation["max_drawdown"]
            for r in trials
            for segment in ("metrics", "later_metrics")
        )
        stable_selection = all(not pbo[market, scenario]["is_overfit"] for scenario in scenario_ids)
        base_checks = all(
            r["prefix_invariant"]
            and all(
                r["checks"][sample]["profit"]["checks_passed"]
                for sample in ("full", "chronological")
            )
            for r in trials
        )
        gates[key] = {
            "target_all_scenarios": target,
            "holm_profit_p": adjusted[key + ":profit"],
            "holm_active_p": adjusted[key + ":active"],
            "pbo_pass": stable_selection,
            "native_profit_checks": base_checks,
            "validated_target": bool(
                target
                and stable_selection
                and base_checks
                and adjusted[key + ":profit"] < validation["holm_alpha"]
            ),
            "validated_excess": bool(
                all(
                    r["prefix_invariant"]
                    and all(
                        r["checks"][sample]["active"]["checks_passed"]
                        for sample in ("full", "chronological")
                    )
                    for r in trials
                )
                and stable_selection
                and adjusted[key + ":active"] < validation["holm_alpha"]
            ),
        }
    table = pd.DataFrame(comparisons).merge(
        pd.DataFrame.from_dict(gates, orient="index"), left_on="candidate", right_index=True
    )
    table.to_csv(output / "comparison.csv", index=False)
    train = matrices["base"].loc[lambda f: f.index < pd.Timestamp(protocol["split_date"])]
    winner = str(train.mean().idxmax())
    summary = {
        "status": "historical_research_only",
        "trial_count": len(keys),
        "scenario_runs": len(comparisons),
        "cash_return": 0,
        "currency": "USD",
        "gates": gates,
        "pbo": {f"{market}:{scenario}": value for (market, scenario), value in pbo.items()},
        "holm": {"hypotheses": len(raw), "raw_p": raw, "adjusted_p": adjusted},
        "chronological_selection": {
            "candidate": winner,
            "training_scores": train.mean().to_dict(),
            "later_metrics": {
                scenario: results[winner, scenario]["later_metrics"] for scenario in scenario_ids
            },
        },
        "power_limit": {
            "half_year_test_blocks": 10,
            "smallest_exact_sign_p": 2.0**-10,
            "smallest_possible_first_holm_p": len(raw) * 2.0**-10,
            "interpretation": "With this short history and 192 hypotheses, the conservative all-block Holm gate cannot pass even ten positive half-years. Failure does not establish absence of an edge; longer data or independently preregistered forward evidence is needed.",
        },
        "limitations": protocol["limitations"],
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    base = table.loc[table.scenario.eq("base")].sort_values("cagr", ascending=False)
    lines = [
        "# Daily entry/exit study",
        "",
        f"{len(keys)} declared candidates, {len(comparisons)} runs. USD; {protocol['evaluation_start']}–{protocol['evaluation_end']}; cash yield 0%. Six fixed rules, eight markets, 1x/1.5x leverage.",
        "",
        "Dukascopy bid/ask bars with verified Yahoo corporate-action adjustments for ETFs. Entry at next open. Stops respect opening gaps and use only prior-bar trail updates. Margin borrowing costs 8% annually (12% stress), not IG CFD funding.",
        "",
        "## All candidates, base costs",
        "",
        "| Candidate | CAGR | Daily max DD | Later CAGR | Matched hold CAGR | Target at all stresses | Holm profit p |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in base.itertuples():
        lines.append(
            f"| {row.candidate} | {row.cagr:.2%} | {row.max_drawdown:.2%} | {row.later_cagr:.2%} | {row.benchmark_cagr:.2%} | {row.target_all_scenarios} | {row.holm_profit_p:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        f"Frozen selection using only the first 24 months: {winner}. Later performance is in results.json. The sample is retrospective, not a fresh holdout.",
        "",
        "The target is >15% CAGR and <=40% DAILY closing-equity drawdown in full and later periods under every cost scenario. Intraday losses can be larger. Passing this numerical screen is distinct from passing statistical validation or beating buy-and-hold.",
        "",
        "Native monthly, 3/6-month block sign tests and N=1 PSR are retained. Native PBO uses 12 rule/leverage candidates per market, six ten-month windows, 20 symmetric partitions. Holm covers all 192 profit/excess hypotheses across markets. All prefix valuations, orders and signals are rerun.",
        "",
        summary["power_limit"]["interpretation"],
        "",
        "The minimum first Holm-adjusted p is 192/1024 = 0.1875; 0.05 cannot be reached. We retain the declared conservative gate and disclose its limited power instead of loosening it after searching.",
        "",
        "N=1 native DSR is PSR; there is no complete DSR correction for all prior research. No strategy is labeled validated solely because its CAGR exceeds the target.",
        "",
        "## Limitations",
        "",
    ]
    lines += ["- " + item for item in protocol["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 7))
    for leverage, group in base.groupby("leverage"):
        axis.scatter(
            -100 * group.max_drawdown, 100 * group.cagr, label=f"{leverage:g}x", alpha=0.75
        )
    axis.axhline(15, color="black", linestyle="--", linewidth=1)
    axis.axvline(40, color="black", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Maximaler täglicher Drawdown (%)",
        ylabel="CAGR nach Basiskosten (%)",
        title="Alle Entry-/Exit-Kandidaten — USD, Cash 0 %",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "risk_return.png", dpi=150)
    plt.close(figure)
    manifest.update(
        status="complete",
        finished_at=datetime.now(timezone.utc).isoformat(),
        scenario_runs=len(comparisons),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Artifacts: {output.resolve()}", flush=True)
    return summary
