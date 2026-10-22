"""Kausale (walk-forward-ehrliche) Risk-Kalibrierung fürs Reporting.

Das Deployment-Sizing (assets.json / bot.py) wird weiterhin auf allen Folds
kombiniert kalibriert — beim echten Forward-Trading ist die gesamte bisherige
Historie die korrekte Kalibrierungsbasis. Fürs REPORTING dagegen darf Fold i
nur mit Risk-Parametern bewertet werden, die auf den Folds 0..i-1 kalibriert
wurden. Sonst werden risk_per_trade und Circuit-Breaker per Binary-/Grid-
Search auf genau der Trade-Sequenz optimiert, die anschließend bewertet wird
(Hindsight-Optimierung von Calmar/Equity/Max-DD).
"""
from typing import Any, Dict, List

import numpy as np

from fwbg.simulation.equity import simulate_equity_timeline
from fwbg.simulation.trade import (
    monte_carlo_equity_from_returns,
    monte_carlo_permutation_test,
)


def circuit_breaker_mask(
    binary_trades: List[float],
    pause_after_losses: int,
    pause_bars: int,
) -> List[bool]:
    """Vorwärts-Maske der tatsächlich ausgeführten Trades unter Circuit Breaker.

    Repliziert die Pause-Logik von ``simulate_with_circuit_breaker`` (Pause
    nach N Verlusten in Serie, ``pause_bars`` Trades aussetzen), gibt aber
    pro Trade zurück, ob er ausgeführt worden wäre — damit lassen sich die
    zugehörigen PnL-Werte kausal filtern.
    """
    if pause_after_losses <= 0:
        return [True] * len(binary_trades)

    mask = []
    consecutive_losses = 0
    pause_remaining = 0
    for trade in binary_trades:
        if pause_remaining > 0:
            pause_remaining -= 1
            mask.append(False)
            continue
        mask.append(True)
        if trade > 0:
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            if consecutive_losses >= pause_after_losses:
                pause_remaining = pause_bars
                consecutive_losses = 0
    return mask


def sized_returns(trades, fk, circuit_breaker, prior_pnls):
    """Size *trades* with risk parameters calibrated beforehand (no hindsight).

    fk and circuit breaker come from earlier data; the loss scale (cf.
    ``pnl_to_returns``) is the mean loss of *prior_pnls*. Returns (returns,
    pnl, indices of the trades the circuit breaker let through).
    """
    mask = circuit_breaker_mask(
        [t["result"] for t in trades],
        circuit_breaker["pause_after_losses"] if circuit_breaker.get("enabled") else 0,
        circuit_breaker.get("pause_bars", 0),
    )
    losses = [abs(p) for p in prior_pnls if p < 0]
    scale = (
        float(np.mean(losses)) if losses
        else float(np.mean(np.abs(prior_pnls))) or 1.0
    )
    taken = [i for i, keep in enumerate(mask) if keep]
    pnl = [trades[i]["pnl_raw"] for i in taken]
    return [fk * p / scale for p in pnl], pnl, taken


def monte_carlo_summary(pnl, returns) -> Dict[str, Any]:
    """Permutation test on *pnl* and equity Monte Carlo on *returns*, as reported."""
    mc_perm = monte_carlo_permutation_test(pnl, n_permutations=1000)
    mc_equity = monte_carlo_equity_from_returns(returns, n_simulations=500)
    return {
        "p_value": mc_perm["p_value"],
        "is_significant": mc_perm["is_significant"],
        "percentile": mc_perm["percentile"],
        "equity_median": mc_equity["median_equity"],
        "equity_p5": mc_equity["p5_equity"],
        "equity_p95": mc_equity["p95_equity"],
        "bankruptcy_rate": mc_equity["bankruptcy_rate"],
    }


def equity_replay(exit_times, returns, max_points=2000) -> Dict[str, Any]:
    """Replay *returns* at their real exit times (time-true equity curve).

    Instead of compounding by trade index, the returns are replayed
    chronologically over the whole history → max DD and annual return from
    the real calendar span. Curves longer than *max_points* are thinned.
    """
    equity = simulate_equity_timeline(exit_times, returns)
    pts = equity["equity_points"]
    if len(pts) > max_points:
        step = len(pts) // max_points + 1
        equity["equity_points"] = pts[::step] + [pts[-1]]
    return equity


def slim_trades(trades_detailed) -> List[Dict[str, Any]]:
    """The fields of each detailed trade that results.json keeps."""
    return [
        {
            "entry_time": t.get("entry_time"),
            "exit_time": t.get("exit_time"),
            "direction": t.get("direction"),
            "result": t.get("result"),
            "pnl_raw": t.get("pnl_raw"),
            "exit_reason": t.get("exit_reason"),
        }
        for t in trades_detailed
    ]


def compute_causal_reporting(
    unified_fold_results: List[Dict[str, Any]],
    risk_mgr,
    rrr: float,
    risk_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Berechnet kausale Per-Trade-Returns über die Walk-Forward-Folds.

    Für Fold i (chronologisch, i > 0) wird ``compute_risk_params`` nur mit
    den Trades der Folds 0..i-1 aufgerufen; das Ergebnis (risk_per_trade,
    Circuit-Breaker) wird ohne erneute Suche auf Fold i angewandt. Auch die
    Loss-Skala der Return-Umrechnung (avg. Loss, vgl. ``pnl_to_returns``)
    stammt ausschließlich aus den Prior-Folds. Das RRR ist das des Settings,
    mit dem Fold i getradet wurde (``fold_result["rrr"]``), sonst ``rrr``.

    Fold 0 hat keine Prior-Daten und wird ausgeschlossen. Liefert die
    Prior-Kalibrierung fk <= 0, gilt der Fold als nicht getradet.

    Returns:
        dict mit:
            - causal_returns: Per-Trade-Returns (Kapital-Fraktionen) der
              kausal getradeten Trades, chronologisch
            - causal_pnl: zugehörige rohe PnL-Werte
            - causal_trade_indices: Index je Trade in der flachen
              Aggregat-Trade-Liste (Fold-Reihenfolge), für Timestamp-Lookup
            - per_fold_risk: risk_per_trade je Fold (Fold 0: None)
            - untraded_folds: Fold-Indizes mit fk <= 0 auf Prior-Daten
            - excluded_trades: Anzahl nicht bewerteter Trades
              (Fold 0 + nicht getradete Folds + CB-Pausen)
            - fold_0_excluded: True
    """
    causal_returns: List[float] = []
    causal_pnl: List[float] = []
    causal_trade_indices: List[int] = []
    per_fold_risk: List[Any] = []
    untraded_folds: List[int] = []
    excluded_trades = 0

    prior_binary: List[float] = []
    prior_pnls: List[float] = []
    prior_rv: List[float] = []
    flat_offset = 0

    for fold_idx, fold_result in enumerate(unified_fold_results):
        fold_trades = fold_result["trades"]
        fold_binary = [t["result"] for t in fold_trades]
        fold_pnls = [t["pnl_raw"] for t in fold_trades]
        fold_rv = [t["rv_at_entry"] for t in fold_trades if "rv_at_entry" in t]

        if not prior_binary:
            # Fold 0 (bzw. führende leere Folds): keine Prior-Daten.
            per_fold_risk.append(None)
            excluded_trades += len(fold_trades)
        else:
            prior_wr = sum(1 for b in prior_binary if b > 0) / len(prior_binary)
            risk_result = risk_mgr.compute_risk_params(
                prior_binary, prior_wr, fold_result.get("rrr", rrr),
                rv_values=prior_rv if len(prior_rv) == len(prior_binary) else None,
                **risk_params,
            )
            fk = risk_result["risk_per_trade"]
            per_fold_risk.append(fk)

            if fk <= 0:
                untraded_folds.append(fold_idx)
                excluded_trades += len(fold_trades)
            else:
                returns, pnl, taken = sized_returns(
                    fold_trades, fk, risk_result["circuit_breaker"], prior_pnls
                )
                causal_returns.extend(returns)
                causal_pnl.extend(pnl)
                causal_trade_indices.extend(flat_offset + i for i in taken)
                excluded_trades += len(fold_trades) - len(taken)

        prior_binary.extend(fold_binary)
        prior_pnls.extend(fold_pnls)
        prior_rv.extend(fold_rv)
        flat_offset += len(fold_trades)

    return {
        "causal_returns": causal_returns,
        "causal_pnl": causal_pnl,
        "causal_trade_indices": causal_trade_indices,
        "per_fold_risk": per_fold_risk,
        "untraded_folds": untraded_folds,
        "excluded_trades": excluded_trades,
        "fold_0_excluded": True,
    }
