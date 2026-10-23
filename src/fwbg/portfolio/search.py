"""Auditable comparison of predeclared rules, including unsuccessful candidates."""

import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fwbg.optimization.overfitting import probability_of_backtest_overfitting

from .cli import run_experiment
from .data import file_fingerprint, load_inputs


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Familywise p-value correction valid without independence of hypotheses."""
    if not p_values or not all(np.isfinite(p) and 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("Holm requires finite p-values in [0, 1]")
    ranked = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted = {}
    previous = 0.0
    for rank, key in enumerate(ranked):
        previous = min(1.0, max(previous, (len(ranked) - rank) * p_values[key]))
        adjusted[key] = previous
    return adjusted


def joint_p(robustness: dict, comparison: str) -> float:
    """Intersection of full/chronological and monthly/blocked sign tests.

    The largest p is conservative for requiring every component to pass. The
    samples overlap: these are not independent replications of the evidence.
    """
    values = []
    for sample in ("full_sample", "walk_forward_oos"):
        checks = (robustness.get(sample) or {}).get(comparison, {})
        if checks.get("status") != "evaluated":
            return 1.0
        values.append(checks["monte_carlo"]["p_value"])
        values.extend(b["p_value"] for b in checks["block_sign_permutations"].values())
    return float(max(values))


def run_search(protocol_path: Path, output: Path) -> dict:
    """Evaluate declared monthly candidates at every cost stress and save results.

    Require matching benchmarks and histories, retain every trial, and return
    the summary with native PBO, Holm gates and chronological selection.
    """
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != 1:
        raise ValueError("search schema_version must be 1")
    candidates = protocol["candidates"]
    ids = [candidate["id"] for candidate in candidates]
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError("search requires distinct candidate identifiers")
    if any(not key or not all(c.isalnum() or c == "_" for c in key) for key in ids):
        raise ValueError("candidate identifiers must contain only letters, digits and underscores")
    if protocol["permutations"] != 10000 or protocol["seed"] != 42:
        raise ValueError(
            "this protocol uses the fixed-rule engine's 10000 permutations and seed 42"
        )
    multipliers = protocol["cost_multipliers"]
    if multipliers != [1, 2, 5]:
        raise ValueError("search requires predeclared 1x, 2x, 5x cost scenarios")
    base_path = protocol_path.parent / protocol["base_config"]
    config, _, _, provenance = load_inputs(base_path)
    if not config.get("split_date"):
        raise ValueError("search requires a chronological split_date")
    for key in ("prices", "metadata"):
        if key in config:
            config[key] = str((base_path.parent / config[key]).resolve())
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": file_fingerprint(protocol_path),
        "base_provenance": provenance,
        "search_implementation": file_fingerprint(Path(__file__)),
        "candidate_ids": ids,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    reports: dict = {}
    monthly_by_cost: dict = {}
    comparison_rows = []
    for multiplier in multipliers:
        series = {}
        benchmark_returns = None
        for candidate in candidates:
            key = candidate["id"]
            candidate_config = {
                **config,
                **{k: v for k, v in candidate.items() if k != "id"},
                "name": key,
                "selection_count": protocol["selection_count"],
                "equity_assets": protocol["equity_assets"],
                "defensive_asset": protocol["defensive_asset"],
            }
            candidate_path = output / f"{key}.json"
            candidate_path.write_text(json.dumps(candidate_config, indent=2) + "\n")
            run_dir = output / f"{key}_cost{multiplier}"
            print(f"Running {key}: costs x{multiplier}", flush=True)
            with (output / f"{key}_cost{multiplier}.log").open("w") as log:
                with contextlib.redirect_stdout(log):
                    report = run_experiment(candidate_path, run_dir, multiplier)
            reports[key, multiplier] = report
            monthly = pd.read_csv(run_dir / "monthly_returns.csv", index_col=0, parse_dates=True)
            if benchmark_returns is None:
                benchmark_returns = monthly[["static", "equity_buy_hold", "cash"]]
            elif not benchmark_returns.equals(monthly[["static", "equity_buy_hold", "cash"]]):
                raise ValueError("candidate benchmarks differ; evaluation dates must match")
            series[key] = monthly.trend
            folds = pd.read_csv(run_dir / "robustness_folds.csv")
            own_folds = folds.loc[folds.portfolio == "trend"]
            weights = pd.read_csv(run_dir / "weights.csv")
            comparison_rows.append(
                {
                    "candidate": key,
                    "cost_multiplier": multiplier,
                    "one_way_cost_bps": report["effective_one_way_cost_bps"],
                    **report["metrics"]["full"]["trend"],
                    "before_split_cagr": report["metrics"]["before_split"]["trend"]["cagr"],
                    "after_split_cagr": report["metrics"]["after_split"]["trend"]["cagr"],
                    "positive_folds": int((own_folds.total_return > 0).sum()),
                    "folds": len(own_folds),
                    "average_risky_weight": float(
                        1 - weights.loc[weights.portfolio == "trend", "cash"].mean()
                    ),
                    "profit_p": joint_p(report["robustness"], "trend"),
                    "active_p": joint_p(report["robustness"], "trend_minus_static"),
                }
            )
        # All runs must have the same complete monthly observations and benchmarks.
        if not all(value.index.equals(series[ids[0]].index) for value in series.values()):
            raise ValueError("candidate histories differ; use a common evaluation start")
        frame = pd.DataFrame(series)
        frame.to_csv(output / f"candidate_monthly_returns_cost{multiplier}.csv")
        monthly_by_cost[multiplier] = frame

    pbo = {}
    for multiplier, frame in monthly_by_cost.items():
        reference = reports[ids[0], multiplier]["robustness"]["walk_forward"]
        if reference["status"] != "evaluated":
            raise ValueError("insufficient data for the predeclared eight-window search")
        size = reference["test_months_each"]
        test_frame = frame.iloc[-8 * size :]
        matrix = np.stack(
            [test_frame.iloc[n * size : (n + 1) * size].mean().to_numpy() for n in range(8)], axis=1
        )
        pd.DataFrame(matrix, index=ids, columns=range(1, 9)).to_csv(
            output / f"pbo_matrix_cost{multiplier}.csv"
        )
        pbo[str(multiplier)] = {
            **probability_of_backtest_overfitting(matrix),
            "score": protocol["pbo_score"],
            "candidate_count": len(ids),
            "months_per_window": size,
        }
    # Stresses are not additional winning candidates: all scenarios must pass.
    raw = {
        f"{key}:{label}": max(joint_p(reports[key, m]["robustness"], field) for m in multipliers)
        for key in ids
        for label, field in (("profit", "trend"), ("active", "trend_minus_static"))
    }
    adjusted = holm_adjust(raw)
    rows = pd.DataFrame(comparison_rows)
    pbo_pass = all(value["pbo"] is not None and not value["is_overfit"] for value in pbo.values())
    gates = {}
    for key in ids:
        robustness_pass = all(
            reports[key, m]["robustness"]["profitability_checks_passed"] for m in multipliers
        )
        active_pass = all(
            reports[key, m]["robustness"]["benchmark_outperformance_checks_passed"]
            for m in multipliers
        )
        gates[key] = {
            "holm_profit_p": adjusted[f"{key}:profit"],
            "holm_active_p": adjusted[f"{key}:active"],
            "profitability_filter_passed": bool(
                robustness_pass and adjusted[f"{key}:profit"] < 0.05 and pbo_pass
            ),
            "benchmark_filter_passed": bool(
                active_pass and adjusted[f"{key}:active"] < 0.05 and pbo_pass
            ),
        }
    rows = rows.merge(
        pd.DataFrame.from_dict(gates, orient="index"), left_on="candidate", right_index=True
    )
    rows.to_csv(output / "comparison.csv", index=False)
    split = pd.Timestamp(config["split_date"])
    training = monthly_by_cost[1].loc[lambda frame: frame.index < split]
    winner = str(training.mean().idxmax())
    selection = {
        "rule": protocol["selection_rule"],
        "selected_candidate": winner,
        "training_mean_monthly_returns": training.mean().to_dict(),
        "test_metrics_by_cost": {
            str(m): reports[winner, m]["metrics"]["after_split"]["trend"] for m in multipliers
        },
    }
    summary = {
        "status": "historical_research_only",
        "cash_return": 0.0,
        "pbo": pbo,
        "multiple_testing": {
            "method": protocol["multiple_testing"],
            "raw_p": raw,
            "holm_p": adjusted,
        },
        "gates": gates,
        "chronological_selection": selection,
        "limitations": protocol["limitations"]
        + [
            "Single-rule native DSR values use N=1 and are PSR diagnostics. Holm provides this search's multiple-testing correction.",
            "The shared DSR routine's N>1 expected-max term has no search-specific Sharpe variance scale; it is not used as a calibrated search-level filter.",
            "Native PBO diagnoses ranking instability within this candidate set, not a probability of future loss. It does not validate profitability.",
            "Sign tests depend on symmetry/dependence assumptions; blocking does not guarantee independent observations. Holm only corrects this declared family.",
        ],
    }
    (output / "search_results.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Weitere Strategien: festgelegter Vergleich",
        f"Echte Läufe im fwbg-Portfolio-Modul, keine ML-/TP-SL-Optimierung. Alle {len(ids)} Kandidaten sind enthalten. {config['base_currency']}, ungehebelt, monatliche Signale, Ausführung am folgenden Handelstag zum Schlusskurs, Cashrendite 0 %. Basis: {reports[ids[0], 1]['effective_one_way_cost_bps']:g} bp je gehandeltem Nominal; Stress: doppelte und fünffache Kosten.",
        "",
        f"Auswertung: {config.get('start')} bis {config.get('end')}; Zeitaufteilung: {config['split_date']}. Universum: {', '.join(config['weights'])}.",
        "",
        "| Regel | CAGR | Max. Drawdown | CAGR nach Zeitaufteilung | CAGR bei 5x Kosten | Positive Fenster | Holm p Gewinn | Gewinnfilter |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in ids:
        row = rows.loc[(rows.candidate == key) & (rows.cost_multiplier == 1)].iloc[0]
        stress = rows.loc[(rows.candidate == key) & (rows.cost_multiplier == 5)].iloc[0]
        lines.append(
            f"| {key} | {row.cagr:.2%} | {row.max_drawdown:.2%} | {row.after_split_cagr:.2%} | {stress.cagr:.2%} | {row.positive_folds}/{row.folds} | {row.holm_profit_p:.4f} | {'bestanden' if row.profitability_filter_passed else 'nicht bestanden'} |"
        )
    benchmark = reports[ids[0], 1]["metrics"]["full"]
    pbo_text = ", ".join(f"{m}x Kosten: {pbo[str(m)]['pbo']:.2%}" for m in multipliers)
    lines.extend(
        [
            "",
            f"Vergleich: statische Gleichgewichtung {benchmark['static']['cagr']:.2%} CAGR / {benchmark['static']['max_drawdown']:.2%} maximaler Drawdown; SPY Buy-and-hold {benchmark['equity_buy_hold']['cagr']:.2%} / {benchmark['equity_buy_hold']['max_drawdown']:.2%}.",
            "",
            f"Native PBO (je 70 symmetrische Aufteilungen): {pbo_text}. Ranking nach durchschnittlicher Monatsrendite, ohne Risikoangleichung.",
            "",
            f"Auswahl ausschließlich anhand des Zeitraums vor {config['split_date']}: {winner}. Die zugehörigen späteren Ergebnisse stehen in search_results.json. Da diese Markthistorie bereits angesehen wurde, ist dies kein unangetasteter Holdout.",
            "",
            f"Der Gewinnfilter verlangt die ursprünglichen Monats-/Blocktests und PSR-Prüfungen im Gesamt- und chronologischen Zeitraum, die Holm-Korrektur über alle {2 * len(ids)} Hypothesen und PBO <= 50 %, jeweils auch unter Kostenstress. Niedrigerer Drawdown allein ist kein Beleg für Mehrertrag. Benchmarkfilter separat:",
            "",
        ]
    )
    lines.extend(
        f"- {key}: {'bestanden' if gates[key]['benchmark_filter_passed'] else 'nicht bestanden'}, Holm p={gates[key]['holm_active_p']:.4f}."
        for key in ids
    )
    lines.extend(["", "Einschränkungen:", ""] + [f"- {item}" for item in summary["limitations"]])
    lines.extend(
        ["", "Ideenquellen (angepasste Regeln, keine exakten Replikationen):", ""]
        + [f"- {url}" for url in protocol["sources"]]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    # Include the initial value so the first month's return remains visible.
    equity = pd.read_csv(output / f"{ids[0]}_cost1" / "equity.csv", index_col=0, parse_dates=True)
    paths = equity[["static", "equity_buy_hold"]].copy()
    for key in ids:
        paths[key] = pd.read_csv(
            output / f"{key}_cost1" / "equity.csv", index_col=0, parse_dates=True
        ).trend
    paths.div(paths.iloc[0]).plot(ax=ax, logy=True)
    ax.set_ylabel("Vermögen / Startkapital (logarithmisch)")
    ax.set_title(f"FWBG: alle {len(ids)} Regeln, {config['base_currency']}, Cash 0 %, Basiskosten")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    manifest.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Artifacts: {output.resolve()}")
    return summary
