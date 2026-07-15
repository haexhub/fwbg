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
    stammt ausschließlich aus den Prior-Folds.

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
                prior_binary, prior_wr, rrr,
                rv_values=prior_rv if len(prior_rv) == len(prior_binary) else None,
                **risk_params,
            )
            fk = risk_result["risk_per_trade"]
            per_fold_risk.append(fk)

            if fk <= 0:
                untraded_folds.append(fold_idx)
                excluded_trades += len(fold_trades)
            else:
                cb = risk_result["circuit_breaker"]
                mask = circuit_breaker_mask(
                    fold_binary,
                    cb["pause_after_losses"] if cb.get("enabled") else 0,
                    cb.get("pause_bars", 0),
                )
                prior_losses = [abs(p) for p in prior_pnls if p < 0]
                scale = (
                    float(np.mean(prior_losses)) if prior_losses
                    else float(np.mean(np.abs(prior_pnls))) or 1.0
                )
                for i, (pnl, taken) in enumerate(zip(fold_pnls, mask)):
                    if taken:
                        causal_returns.append(fk * pnl / scale)
                        causal_pnl.append(pnl)
                        causal_trade_indices.append(flat_offset + i)
                    else:
                        excluded_trades += 1

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
