"""
Equity-Simulation und Portfolio-Filter.

Enthält:
- Equity-Kurven-Simulation
- Korrelationsfilter für Währungs-Diversifikation
"""

from fwbg.data.config import CORR_THRESHOLD


def simulate_equity(trades, risk_per_trade, rrr, start_equity=100.0, compound_cap=1e6):
    """
    Simuliert die Equity-Kurve basierend auf Trade-Ergebnissen.

    Args:
        trades: Liste von Trade-Ergebnissen (>0 = Gewinn, <=0 = Verlust)
        risk_per_trade: Risk pro Trade als Anteil des Kapitals (z.B. 0.02 = 2%)
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
            # Gewinn: Risk * RRR (basierend auf effektiver Equity)
            equity += effective_equity * risk_per_trade * rrr
        else:
            # Verlust: Risk (basierend auf effektiver Equity)
            equity -= effective_equity * risk_per_trade

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


def simulate_equity_from_pnl(pnl_raw, fk, start_equity=100.0, compound_cap=1e6):
    """Equity-Simulation mit tatsächlichen PnL-Werten statt binärem Kelly.

    Im Gegensatz zu simulate_equity (das feste +risk*rrr / -risk verwendet)
    nutzt diese Funktion die echten Trade-PnL-Magnitudes. Skalierung:
    durchschnittlicher Loss-Return = -fk (konsistent mit Kelly-Sizing).

    Args:
        pnl_raw: Liste der tatsächlichen PnL-Werte (positiv=Gewinn, negativ=Verlust)
        fk: Kelly-Fraktion (Risiko pro Trade)
        start_equity: Startkapital (default: 100.0)
        compound_cap: Ab diesem Equity-Wert wird nicht mehr kompoundiert

    Returns:
        dict mit equity_curve, final_equity, max_drawdown, drawdowns
    """
    if not pnl_raw:
        return {
            "equity_curve": [start_equity],
            "final_equity": start_equity,
            "max_drawdown": 0.0,
            "drawdowns": [0.0],
        }

    from fwbg.simulation.trade import pnl_to_returns
    returns = pnl_to_returns(pnl_raw, fk)

    equity = start_equity
    equity_curve = [equity]
    peak = equity
    max_dd = 0.0
    drawdowns = [0.0]

    for r in returns:
        effective_equity = min(equity, compound_cap)
        equity += effective_equity * r
        equity_curve.append(equity)

        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        drawdowns.append(dd * 100)

        if equity <= 0:
            equity = 0.0
            max_dd = 1.0
            break

    return {
        "equity_curve": equity_curve,
        "final_equity": equity,
        "max_drawdown": max_dd,
        "drawdowns": drawdowns,
    }


def simulate_equity_timeline(exit_times, returns, start_equity=100.0, compound_cap=1e6):
    """Zeitbasierter Equity-Replay statt Trade-Index-Simulation.

    Realisiert die Per-Trade-Returns chronologisch zu ihren echten
    Exit-Zeitpunkten über die komplette Historie. Liefert damit eine
    zeitachsen-treue Equity-Kurve und einen Annual-Return aus der realen
    Kalenderspanne (statt aus der Fold-Bar-Summe).

    Args:
        exit_times: Exit-Timestamps (ISO-Strings oder datetime), parallel zu returns
        returns: Per-Trade-Returns als Kapital-Fraktionen (z.B. kausal kalibriert)
        start_equity: Startkapital (default: 100.0)
        compound_cap: Ab diesem Equity-Wert wird nicht mehr kompoundiert

    Returns:
        dict mit:
            - equity_points: [[iso_timestamp, equity], ...] chronologisch,
              erster Punkt = Startkapital
            - final_equity, max_drawdown
            - start_time, end_time (ISO), years (Kalenderspanne)
            - annual_return: annualisierter Return in Prozent
    """
    if not returns or len(exit_times) != len(returns):
        return {
            "equity_points": [],
            "final_equity": start_equity,
            "max_drawdown": 0.0,
            "start_time": None,
            "end_time": None,
            "years": 0.0,
            "annual_return": 0.0,
        }

    import pandas as pd

    times = pd.to_datetime(list(exit_times))
    order = times.argsort()

    equity = start_equity
    peak = equity
    max_dd = 0.0
    equity_points = [[times[order[0]].isoformat(), start_equity]]

    for idx in order:
        effective_equity = min(equity, compound_cap)
        equity += effective_equity * returns[idx]
        if equity <= 0:
            equity = 0.0
            max_dd = 1.0
            equity_points.append([times[idx].isoformat(), 0.0])
            break
        equity_points.append([times[idx].isoformat(), equity])
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    start_time = times[order[0]]
    end_time = times[order[-1]]
    years = max((end_time - start_time).total_seconds() / (365.25 * 24 * 3600), 0.0)
    if equity > 0 and years > 0:
        annual_return = ((equity / start_equity) ** (1 / years) - 1) * 100
    else:
        annual_return = -100.0 if equity <= 0 else 0.0

    return {
        "equity_points": equity_points,
        "final_equity": equity,
        "max_drawdown": max_dd,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "years": years,
        "annual_return": annual_return,
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
