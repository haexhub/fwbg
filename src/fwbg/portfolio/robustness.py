"""Adapt FWBG's statistical checks to a fixed monthly portfolio rule.

Observations are non-overlapping portfolio returns, never order counts. The
shared Sharpe implementation with N=1 is a single-strategy PSR, not a correction
for an unknown history of earlier experiments. PBO requires a candidate search
and is explicitly unavailable here. None of these checks fits or selects rules.
"""

from dataclasses import replace
from inspect import getfile
from pathlib import Path

import numpy as np
import pandas as pd

from fwbg.optimization.overfitting import compute_overfitting_metrics
from fwbg.optimization.robust_validation import create_walk_forward_folds
from fwbg.simulation.trade import (
    monte_carlo_equity_from_returns,
    monte_carlo_permutation_test,
)

from .data import file_fingerprint
from .engine import PortfolioResult, PortfolioSettings, run_portfolio, summarize

PORTFOLIOS = ("trend", "static", "equity_buy_hold")
NATIVE_SOURCES = {
    "overfitting": Path(getfile(compute_overfitting_metrics)),
    "monte_carlo": Path(getfile(monte_carlo_permutation_test)),
    "folds": Path(getfile(create_walk_forward_folds)),
}


def monthly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    """Compound daily returns; exclude unverified/partial terminal months."""
    if len(equity) < 3 or not np.isfinite(equity.to_numpy()).all() or (equity <= 0).any().any():
        raise ValueError("robustness requires positive finite equity and at least three dates")
    growth = equity.div(equity.shift()).iloc[1:]
    monthly = growth.resample("ME").prod() - 1
    # The first month's baseline must precede the month. Otherwise its return
    # is partial and must not count as a whole observation.
    if equity.index[0].to_period("M") == monthly.index[0].to_period("M"):
        monthly = monthly.iloc[1:]
    last = equity.index[-1]
    last_weekday = pd.offsets.BMonthEnd().rollforward(last)
    if not last.is_month_end and last != last_weekday:
        monthly = monthly.loc[monthly.index.to_period("M") < last.to_period("M")]
    return monthly


def _blocked_returns(values: np.ndarray, horizon: int) -> np.ndarray:
    """Compound consecutive fixed-length return blocks, discarding the tail."""
    count = len(values) // horizon
    return np.asarray((1 + values[: count * horizon].reshape(count, horizon)).prod(axis=1) - 1)


def _check_series(
    returns: pd.Series,
    *,
    permutations: int,
    seed: int,
    benchmark: pd.Series | None = None,
) -> dict:
    """Assess monthly returns with native sign, Sharpe and compounded block tests.

    An optional aligned benchmark produces active returns; fewer than 24
    observations yield an insufficient-data result that cannot pass checks.
    """
    original = returns.to_numpy(dtype=float)
    reference = benchmark.to_numpy(dtype=float) if benchmark is not None else None
    observed = original if reference is None else original - reference
    if len(observed) < 24:
        return {
            "status": "insufficient_data",
            "observations": len(observed),
            "checks_passed": False,
            "reason": "At least 24 complete months required",
        }
    if not np.isfinite(observed).all():
        raise ValueError("robustness returns must be finite")
    mc = monte_carlo_permutation_test(
        observed.tolist(), n_permutations=permutations, random_seed=seed
    )
    std = float(observed.std())
    overfitting = compute_overfitting_metrics(
        trade_returns=observed.tolist(),
        observed_sr=float(observed.mean() / std) if std > 1e-12 else 0.0,
        n_strategies=1,
        grid_results_by_fold={},
        n_trades=len(observed),
    )
    sharpe = {
        **overfitting["dsr"],
        "interpretation": "single_strategy_PSR; no correction for prior strategy searches",
        "observation_unit": "monthly return (not trade)",
    }
    blocks = {}
    for horizon in (3, 6):
        values = _blocked_returns(original, horizon)
        if reference is not None:
            values -= _blocked_returns(reference, horizon)
        test = monte_carlo_permutation_test(
            values.tolist(), n_permutations=permutations, random_seed=seed
        )
        blocks[str(horizon)] = {
            **test,
            "block_months": horizon,
            "observations": len(values),
            "discarded_tail_months": len(original) % horizon,
        }
    autocorrelation = float(pd.Series(observed).autocorr()) if std > 1e-12 else float("nan")
    return {
        "status": "evaluated",
        "observations": len(observed),
        "monthly_mean_return": float(observed.mean()),
        "monthly_lag1_autocorrelation": autocorrelation if np.isfinite(autocorrelation) else None,
        "monte_carlo": mc,
        "dsr_single_strategy": sharpe,
        "block_sign_permutations": blocks,
        "checks_passed": bool(
            mc["is_significant"]
            and sharpe["is_significant"]
            and all(b["is_significant"] for b in blocks.values())
        ),
    }


def _checks(frame: pd.DataFrame, permutations: int, seed: int) -> dict:
    """Check each portfolio's returns and the trend-minus-static comparison."""
    checks = {
        name: _check_series(frame[name], permutations=permutations, seed=seed)
        for name in PORTFOLIOS
    }
    checks["trend_minus_static"] = _check_series(
        frame["trend"], benchmark=frame["static"], permutations=permutations, seed=seed
    )
    return checks


def assess_robustness(
    result: PortfolioResult,
    prices: pd.DataFrame,
    settings: PortfolioSettings,
    *,
    permutations: int = 10_000,
    equity_simulations: int = 500,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Run shared FWBG checks and eight chronological, fixed-rule evaluations.

    Two initial years precede eight non-overlapping test windows. There is no
    training or inner search; holdings persist across boundaries. Prefix reruns
    verify that later prices cannot change an earlier fold's values/signals.
    """
    if permutations < 1 or equity_simulations < 1:
        raise ValueError("simulation counts must be positive")
    monthly = monthly_returns(result.equity)
    report: dict = {
        "cash_return": 0.0,
        "observation_unit": "non-overlapping calendar-month returns",
        "seed": seed,
        "permutations": permutations,
        "thresholds": {"sign_permutation_p_below": 0.05, "single_strategy_PSR_above": 0.95},
        "full_sample": _checks(monthly, permutations, seed),
        "pbo": {
            "status": "not_applicable",
            "pbo": None,
            "n_cscv_splits": 0,
            "reason": "One fixed rule, no candidate-selection matrix. Benchmarks and cost stresses are not tuning trials.",
        },
        "multiple_testing": {
            "status": "not_estimated",
            "fixed_rules_in_this_experiment": 1,
            "prior_search_count": None,
            "reason": "N=1 reduces DSR to PSR. It does not correct for prior user or published strategy searches.",
        },
        "native_implementations": {
            name: file_fingerprint(path) for name, path in NATIVE_SOURCES.items()
        },
        "not_applicable": [
            "ML feature stability, inner-CV selection and holdout/inner ratios: no fitted model or optimizer.",
            "Closed-trade minimum/win-rate filters: allocation transactions are not independent round-trip trades.",
        ],
        "assumptions": [
            "Sign permutation assumes symmetric signs under the null; monthly observations need not be independent.",
            "Non-overlapping 3/6-month block tests retain within-block dependence but do not eliminate dependence between blocks.",
            "The historical sample has already been inspected; chronological tests are not a new untouched holdout.",
            "Profitability above zero is distinct from outperformance of the static portfolio; neither establishes risk-adjusted alpha.",
        ],
    }
    report["equity_permutation_diagnostic"] = {
        "trend": monte_carlo_equity_from_returns(
            monthly.trend.tolist(), n_simulations=equity_simulations, random_seed=seed
        ),
        "used_as_filter": False,
        "reason": "Permuting fixed percentage returns leaves terminal compounded wealth unchanged; the percentiles are not a risk interval.",
    }
    fold_rows = []
    test_size = (len(monthly) - 24) // 8
    if test_size < 12:
        report["walk_forward"] = {
            "status": "insufficient_data",
            "reason": "Need 24 history months plus 8 test windows of at least 12 months",
        }
        report["walk_forward_oos"] = None
    else:
        folds = create_walk_forward_folds(
            monthly,
            n_folds=8,
            test_size=test_size,
            min_train_size=24,
            anchored=True,
            embargo_bars=0,
        )
        for fold in folds:
            begin = fold.test_df.index[0].to_period("M").start_time
            end = fold.test_df.index[-1]
            before = result.equity.loc[result.equity.index < begin].iloc[-1:]
            segment = pd.concat([before, result.equity.loc[begin:end]])
            prefix = run_portfolio(
                prices, replace(settings, end=segment.index[-1].date().isoformat())
            )
            prefix_matches = prefix.equity.equals(result.equity.loc[: segment.index[-1]])
            expected_signals = result.signals.loc[
                result.signals.execution_date <= segment.index[-1]
            ]
            prefix_matches = prefix_matches and prefix.signals.equals(expected_signals)
            for name, metrics in summarize(segment).items():
                if name == "cash":
                    continue
                fold_rows.append(
                    {
                        "fold": fold.fold_id + 1,
                        "portfolio": name,
                        "history_months": len(fold.train_df),
                        "test_months": len(fold.test_df),
                        "prefix_invariant": prefix_matches,
                        **metrics,
                    }
                )
        oos = pd.concat([fold.test_df for fold in folds])
        report["walk_forward_oos"] = _checks(oos, permutations, seed)
        report["walk_forward"] = {
            "status": "evaluated",
            "n_folds": len(folds),
            "test_months_each": test_size,
            "initial_history_months": folds[0].test_start,
            "method": "Shared FWBG fold planner; fixed-rule evaluation without fitting or parameter selection",
            "all_prefixes_invariant": all(row["prefix_invariant"] for row in fold_rows),
            "profitable_folds": {
                name: sum(row["total_return"] > 0 for row in fold_rows if row["portfolio"] == name)
                for name in PORTFOLIOS
            },
        }
    oos_checks = report["walk_forward_oos"]
    prefix_ok = report["walk_forward"].get("all_prefixes_invariant", False)
    report["profitability_checks_passed"] = bool(
        prefix_ok
        and oos_checks
        and report["full_sample"]["trend"]["checks_passed"]
        and oos_checks["trend"]["checks_passed"]
    )
    report["benchmark_outperformance_checks_passed"] = bool(
        prefix_ok
        and oos_checks
        and report["full_sample"]["trend_minus_static"]["checks_passed"]
        and oos_checks["trend_minus_static"]["checks_passed"]
    )
    report["overfitting_validation_complete"] = False
    return report, monthly, pd.DataFrame(fold_rows)


def format_robustness(report: dict) -> str:
    """Render robustness results, gates and statistical limitations as Markdown."""
    lines = [
        "## Robustness (shared FWBG implementations)",
        "Uninvested cash return: **0%**. Tests use non-overlapping months, not order counts.",
    ]
    for section in ("full_sample", "walk_forward_oos"):
        checks = report[section]
        if checks is None:
            continue
        lines.extend(
            [
                f"\n### {section}\n",
                "| Series | Months | MC p | PSR (N=1) | 3-month block p | 6-month block p | Checks |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for name, values in checks.items():
            if values["status"] != "evaluated":
                lines.append(
                    f"| {name} | {values['observations']} | — | — | — | — | insufficient data |"
                )
                continue
            lines.append(
                f"| {name} | {values['observations']} | {values['monte_carlo']['p_value']:.5f} | "
                f"{values['dsr_single_strategy']['dsr']:.5f} | "
                f"{values['block_sign_permutations']['3']['p_value']:.5f} | "
                f"{values['block_sign_permutations']['6']['p_value']:.5f} | "
                f"{'pass' if values['checks_passed'] else 'fail'} |"
            )
    wf = report["walk_forward"]
    if wf["status"] == "evaluated":
        lines.append(
            f"\nChronological folds: {wf['n_folds']}, {wf['test_months_each']} months each, "
            f"after {wf['initial_history_months']} history months. Positive folds: "
            f"{wf['profitable_folds']}. Prefix invariance: {wf['all_prefixes_invariant']}."
        )
    lines.extend(
        [
            "\n**PBO: not applicable** — one fixed rule, no candidate-selection matrix. "
            "The shared DSR routine with N=1 is PSR, **not** a correction for prior searches.",
            f"\nProfitability checks passed: **{report['profitability_checks_passed']}**. "
            f"Benchmark-outperformance checks passed: **{report['benchmark_outperformance_checks_passed']}**. "
            "Complete overfitting validation: **False**.",
            "\nPass requires monthly Monte Carlo p < 0.05, PSR > 0.95, and both block-test p-values < 0.05; "
            "the summary additionally requires the same checks on the chronological test windows and prefix invariance.",
            "\nMonthly/blocked sign tests assume symmetric null signs and do not prove independence or alpha. "
            "All periods are retrospective. The native equity-order permutation is recorded only as a diagnostic: "
            "reordering percentage returns cannot change their final compounded wealth.",
        ]
    )
    return "\n\n".join(lines[:2]) + "\n\n" + "\n".join(lines[2:])
