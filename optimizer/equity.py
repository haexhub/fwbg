"""
Equity-Simulation und Portfolio-Filter.

Enthält:
- Equity-Kurven-Simulation
- Korrelationsfilter für Währungs-Diversifikation
"""

from .config import CORR_THRESHOLD


def simulate_equity(trades, kelly_risk, rrr, start_equity=100.0, compound_cap=1e6):
    """
    Simuliert die Equity-Kurve basierend auf Trade-Ergebnissen.

    Args:
        trades: Liste von Trade-Ergebnissen (>0 = Gewinn, <=0 = Verlust)
        kelly_risk: Risk pro Trade als Anteil des Kapitals (z.B. 0.02 = 2%)
        rrr: Risk-Reward-Ratio (z.B. 2.0 = TP ist 2x SL)
        start_equity: Startkapital (default: 100.0)
        compound_cap: Ab diesem Equity-Wert wird nicht mehr kompoundiert,
                     sondern mit fixer Positionsgröße weitergehandelt (default: 1e6)

    Returns:
        dict mit:
            - equity_curve: Liste der Equity-Werte
            - final_equity: Endkapital
            - max_drawdown: Maximaler Drawdown (0.0-1.0)
            - drawdowns: Liste der Drawdown-Werte in Prozent
    """
    equity = start_equity
    equity_curve = [equity]
    peak = equity
    max_dd = 0
    drawdowns = [0.0]

    for trade_result in trades:
        # Effektive Equity für Positionsberechnung (gecappt)
        effective_equity = min(equity, compound_cap)

        if trade_result > 0:
            # Gewinn: Kelly * RRR (basierend auf effektiver Equity)
            equity += effective_equity * kelly_risk * rrr
        else:
            # Verlust: Kelly (basierend auf effektiver Equity)
            equity -= effective_equity * kelly_risk

        equity_curve.append(equity)

        # Drawdown berechnen
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        drawdowns.append(dd * 100)

        # Bankrott-Check
        if equity <= 0:
            equity = 0
            max_dd = 1.0
            break

    return {
        "equity_curve": equity_curve,
        "final_equity": equity,
        "max_drawdown": max_dd,
        "drawdowns": drawdowns,
    }


def filter_correlated_assets(results, threshold=CORR_THRESHOLD):
    """
    Filtert Assets mit zu hoher Währungskorrelation.
    Verhindert z.B. 5x USD-Long gleichzeitig.
    """
    if not results:
        return []

    sorted_results = sorted(results, key=lambda x: x["pnl"], reverse=True)
    selected = []
    currency_exposure = {}

    for r in sorted_results:
        currencies = r.get("currencies", [])
        if not currencies:
            selected.append(r)
            continue

        max_exposure = max((currency_exposure.get(c, 0) for c in currencies), default=0)

        max_allowed = int(1 / (1 - threshold)) if threshold < 1 else 10

        if max_exposure < max_allowed:
            selected.append(r)
            for c in currencies:
                currency_exposure[c] = currency_exposure.get(c, 0) + 1

    return selected
