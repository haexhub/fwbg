"""
Trade-Simulation und Metriken
"""
import numpy as np
import pandas as pd

from .config import MAX_TRADE_BARS, RELEVANCE_THRESHOLD, FEATURE_STABILITY_MIN, DATA_PATH


# Cache für Sub-Stunden-Daten (wird einmal pro Symbol geladen)
_m15_cache = {}  # 15-Min-Daten
_m30_cache = {}  # 30-Min-Daten


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Berechnet annualisierte Sharpe Ratio aus Trade-Returns."""
    if len(returns) < 2:
        return 0.0
    excess_returns = np.array(returns) - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252 * 6)


def calculate_equity_smoothness(trades, kelly_risk, rrr, window_size=50):
    """
    Berechnet einen Smoothness-Score für die Equity-Kurve.

    Höherer Score = glattere Equity-Kurve mit kleineren Sprüngen.
    Basiert auf der Varianz der rollenden Returns.

    Args:
        trades: Liste von Trade-Ergebnissen (1.0 = Win, -1.0 = Loss)
        kelly_risk: Risk pro Trade
        rrr: Risk-Reward-Ratio
        window_size: Fenster für rollende Berechnung

    Returns:
        dict mit:
            - smoothness_score: 0-1 Score (1 = perfekt glatt)
            - return_volatility: Standardabweichung der Returns
            - max_single_move: Größter einzelner Move (positiv oder negativ)
            - sortino_ratio: Sortino Ratio (nur Downside-Volatilität)
    """
    if len(trades) < window_size:
        return {
            "smoothness_score": 0.5,
            "return_volatility": 0.0,
            "max_single_move": 0.0,
            "sortino_ratio": 0.0,
        }

    # Berechne Trade-Returns
    returns = []
    for t in trades:
        if t > 0:
            returns.append(kelly_risk * rrr)
        else:
            returns.append(-kelly_risk)

    returns = np.array(returns)

    # Basis-Metriken
    return_volatility = np.std(returns)
    max_single_move = max(abs(np.max(returns)), abs(np.min(returns)))

    # Sortino Ratio (nur negative Returns für Downside-Volatilität)
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0:
        downside_std = np.std(negative_returns)
        mean_return = np.mean(returns)
        sortino = mean_return / downside_std if downside_std > 0 else 0
    else:
        sortino = float('inf') if np.mean(returns) > 0 else 0

    # Rollende Varianz der Returns
    rolling_vars = []
    for i in range(len(returns) - window_size):
        window = returns[i:i + window_size]
        rolling_vars.append(np.var(window))

    if rolling_vars:
        # Konsistenz: niedrige Varianz der rollenden Varianz = konsistent glatt
        consistency = 1.0 / (1.0 + np.std(rolling_vars) * 100)
    else:
        consistency = 0.5

    # Smoothness Score: Kombination aus niedriger Volatilität und Konsistenz
    # Normalisiere Volatilität auf 0-1 Skala (5% Kelly = ~5% max single move)
    vol_score = 1.0 / (1.0 + return_volatility * 10)

    smoothness_score = (vol_score * 0.5 + consistency * 0.5)

    return {
        "smoothness_score": float(smoothness_score),
        "return_volatility": float(return_volatility),
        "max_single_move": float(max_single_move),
        "sortino_ratio": float(min(sortino, 10.0)),  # Cap bei 10
        "consistency": float(consistency),
    }


def calculate_calmar_ratio(returns, kelly_risk, rrr):
    """Berechnet Calmar Ratio (Return / Max Drawdown)."""
    if not returns:
        return 0.0

    equity = [100.0]
    for r in returns:
        if r > 0:
            equity.append(equity[-1] * (1 + kelly_risk * rrr))
        else:
            equity.append(equity[-1] * (1 - kelly_risk))

    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd

    max_dd = max(max_dd, 0.01)
    total_return = (equity[-1] - equity[0]) / equity[0]
    return min(10.0, total_return / max_dd)


def check_feature_stability(fold_importances, threshold=RELEVANCE_THRESHOLD):
    """
    Prüft ob Features über alle Folds konsistent wichtig sind.
    Returns: Liste der stabilen Features
    """
    if not fold_importances:
        return []

    feature_counts = {}
    for imps in fold_importances:
        for feat, imp in imps.items():
            if imp > threshold:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1

    stable_features = [
        f for f, count in feature_counts.items()
        if count >= FEATURE_STABILITY_MIN
    ]

    return stable_features


def _load_ohlc_csv(path):
    """
    Lädt eine OHLC CSV-Datei mit flexibler Format-Erkennung.
    Unterstützt:
    - Mit Header: Time,Open,High,Low,Close,Volume
    - Ohne Header (MetaTrader): datetime,O,H,L,C,V
    """
    try:
        # Prüfe ob Header existiert
        with open(path, "r") as f:
            first_line = f.readline()

        if "Time" in first_line or "Open" in first_line:
            # Hat Header
            df = pd.read_csv(path, parse_dates=["Time"], index_col="Time")
        else:
            # Kein Header (MetaTrader Format)
            df = pd.read_csv(
                path,
                names=["Time", "O", "H", "L", "C", "V"],
                parse_dates=["Time"],
                index_col="Time"
            )
            # Rename columns to match expected format
            df.columns = ["Open", "High", "Low", "Close", "Volume"]

        # Rename columns to single letters for consistency (H, L, etc.)
        if "High" in df.columns:
            df = df.rename(columns={"Open": "O", "High": "H", "Low": "L", "Close": "C", "Volume": "V"})

        return df
    except Exception:
        return None


def load_sub_hourly_data(symbol):
    """
    Lädt Sub-Stunden-Daten für ein Symbol (mit Caching).
    Versucht zuerst 15-Min, dann 30-Min als Fallback.

    Returns: (DataFrame, resolution) oder (None, None)
    """
    # Versuche 15-Min-Daten
    if symbol not in _m15_cache:
        m15_path = f"{DATA_PATH}/{symbol}_MINUTE_15.csv"
        _m15_cache[symbol] = _load_ohlc_csv(m15_path)

    if _m15_cache[symbol] is not None:
        return _m15_cache[symbol], 15

    # Fallback: 30-Min-Daten
    if symbol not in _m30_cache:
        m30_path = f"{DATA_PATH}/{symbol}_MINUTE_30.csv"
        _m30_cache[symbol] = _load_ohlc_csv(m30_path)

    if _m30_cache[symbol] is not None:
        return _m30_cache[symbol], 30

    return None, None


def resolve_tp_sl_collision(symbol, hour_timestamp, direction, tp, sl):
    """
    Bei gleichzeitigem TP/SL Hit: Schaut in Sub-Stunden-Daten um die Reihenfolge zu bestimmen.
    Nutzt 15-Min-Daten wenn verfügbar, sonst 30-Min als Fallback.

    Returns: 1.0 (TP zuerst), -1.0 (SL zuerst), None (keine Daten)
    """
    sub_df, resolution = load_sub_hourly_data(symbol)
    if sub_df is None:
        return None

    # Finde die Sub-Stunden-Bars innerhalb der Stunde
    hour_start = hour_timestamp
    hour_end = hour_timestamp + pd.Timedelta(hours=1)

    try:
        sub_bars = sub_df.loc[hour_start:hour_end]
        if len(sub_bars) == 0:
            return None
    except Exception:
        return None

    # Gehe durch die Bars und prüfe was zuerst passiert
    for _, bar in sub_bars.iterrows():
        if direction == 1:  # Long
            tp_hit = bar["H"] >= tp
            sl_hit = bar["L"] <= sl
        else:  # Short
            tp_hit = bar["L"] <= tp
            sl_hit = bar["H"] >= sl

        if tp_hit and sl_hit:
            # Auch im Sub-Bar beide erreicht - konservativ Loss
            return -1.0
        elif tp_hit:
            return 1.0
        elif sl_hit:
            return -1.0

    # Keines erreicht (sollte nicht passieren)
    return None


# Alias für Abwärtskompatibilität
def resolve_tp_sl_collision_m15(symbol, hour_timestamp, direction, tp, sl):
    """Alias für resolve_tp_sl_collision (Abwärtskompatibilität)."""
    return resolve_tp_sl_collision(symbol, hour_timestamp, direction, tp, sl)


def simulate_pro_trade(closes, highs, lows, atrs, idx, direction, tp_m, sl_m, spread,
                       max_bars=None, trailing_start=0.5, timestamps=None, symbol=None,
                       opens=None):
    """
    Simuliert einen Trade und gibt detaillierte Informationen zurück.

    - Signal bei Bar idx, Entry bei Open von Bar idx+1 (kein Look-Ahead!)
    - Trade läuft bis TP oder SL erreicht wird (kein Timeout-Exit!)
    - Bei gleichzeitigem TP/SL im selben Bar: Schaut in 15-Min-Daten (falls verfügbar)

    Args:
        closes: Close-Preise Array
        highs: High-Preise Array
        lows: Low-Preise Array
        atrs: ATR-Werte Array
        idx: Index des Signal-Bars
        direction: 1 für Long, -1 für Short
        tp_m: Take-Profit Multiplikator (in Spreads)
        sl_m: Stop-Loss Multiplikator (in Spreads)
        spread: Spread des Assets
        opens: Optional - Open-Preise Array für realistischen Entry
        timestamps: Optional - Array von Timestamps für M15-Lookup
        symbol: Optional - Symbol-Name für M15-Lookup

    Returns:
        dict mit Trade-Details oder None bei ungültigem Trade:
            - result: 1.0=Win, -1.0=Loss
            - direction: "LONG" oder "SHORT"
            - signal_idx, entry_idx, exit_idx: Bar-Indizes
            - signal_time, entry_time, exit_time: Zeitstempel (falls vorhanden)
            - entry_price_raw: Preis vor Kosten
            - entry_price: Effektiver Entry inkl. Spread+Slippage
            - exit_price: Exit-Preis (TP oder SL Level)
            - tp_level, sl_level: TP/SL Levels
            - spread, slippage, total_cost: Kosten
            - tp_distance, sl_distance: Distanzen in Preiseinheiten
            - pnl_raw: PnL vor Positionsgrößen-Berechnung
            - bars_held: Dauer in Bars
    """
    if max_bars is None:
        max_bars = MAX_TRADE_BARS

    # Entry bei idx+1 (nächster Bar nach Signal)
    entry_idx = idx + 1
    if entry_idx + max_bars >= len(closes):
        return None

    tp_distance = spread * tp_m
    sl_distance = spread * sl_m
    slippage = spread * 0.5

    # Entry: Open des nächsten Bars (realistisch, kein Look-Ahead)
    # Fallback auf Close falls Opens nicht verfügbar
    if opens is not None:
        entry_price = opens[entry_idx]
    else:
        entry_price = closes[idx]  # Fallback (weniger realistisch)

    if direction == 1:  # Long
        entry = entry_price + spread + slippage
        tp = entry + tp_distance - slippage
        sl = entry - sl_distance - slippage
    else:  # Short
        entry = entry_price - spread - slippage
        tp = entry - tp_distance + slippage
        sl = entry + sl_distance + slippage

    # Hilfsfunktion für Rückgabe
    def make_result(result, exit_idx, exit_price):
        bars_held = exit_idx - entry_idx

        # Berechne PnL in Pips/Points
        if direction == 1:
            pnl_raw = exit_price - entry
        else:
            pnl_raw = entry - exit_price

        return {
            "result": result,
            "direction": "LONG" if direction == 1 else "SHORT",
            "signal_idx": idx,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "bars_held": bars_held,
            "signal_time": str(timestamps[idx]) if timestamps is not None else None,
            "entry_time": str(timestamps[entry_idx]) if timestamps is not None else None,
            "exit_time": str(timestamps[exit_idx]) if timestamps is not None else None,
            "entry_price_raw": float(entry_price),
            "entry_price": float(entry),
            "exit_price": float(exit_price),
            "tp_level": float(tp),
            "sl_level": float(sl),
            "spread": float(spread),
            "slippage": float(slippage),
            "total_cost": float(spread + slippage),
            "tp_distance": float(tp_distance),
            "sl_distance": float(sl_distance),
            "pnl_raw": float(pnl_raw),
        }

    for j in range(entry_idx, min(entry_idx + max_bars, len(closes))):
        if direction == 1:  # Long
            tp_hit = highs[j] >= tp
            sl_hit = lows[j] <= sl

            if tp_hit and sl_hit:
                # Beide im selben Bar - versuche Sub-Stunden Lookup
                if timestamps is not None and symbol is not None:
                    result = resolve_tp_sl_collision(symbol, timestamps[j], direction, tp, sl)
                    if result is not None:
                        exit_price = tp if result > 0 else sl
                        return make_result(result, j, exit_price)
                # Fallback: konservativ Loss
                return make_result(-1.0, j, sl)
            elif tp_hit:
                return make_result(1.0, j, tp)
            elif sl_hit:
                return make_result(-1.0, j, sl)

        else:  # Short
            tp_hit = lows[j] <= tp
            sl_hit = highs[j] >= sl

            if tp_hit and sl_hit:
                # Beide im selben Bar - versuche Sub-Stunden Lookup
                if timestamps is not None and symbol is not None:
                    result = resolve_tp_sl_collision(symbol, timestamps[j], direction, tp, sl)
                    if result is not None:
                        exit_price = tp if result > 0 else sl
                        return make_result(result, j, exit_price)
                # Fallback: konservativ Loss
                return make_result(-1.0, j, sl)
            elif tp_hit:
                return make_result(1.0, j, tp)
            elif sl_hit:
                return make_result(-1.0, j, sl)

    # Timeout - kein TP/SL erreicht
    return None


def calculate_max_drawdown(returns, kelly_risk, rrr):
    """Berechnet Maximum Drawdown aus Trade-Returns."""
    if not returns:
        return 0.0

    equity = [100.0]
    for r in returns:
        if r > 0:
            equity.append(equity[-1] * (1 + kelly_risk * rrr))
        else:
            equity.append(equity[-1] * (1 - kelly_risk))

    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd

    return max_dd


def calculate_annual_return(returns, kelly_risk, rrr, total_bars, bars_per_year=8760):
    """Berechnet annualisierte Rendite."""
    if not returns or total_bars == 0:
        return 0.0

    equity = 100.0
    for r in returns:
        if r > 0:
            equity *= (1 + kelly_risk * rrr)
        else:
            equity *= (1 - kelly_risk)

    total_return = (equity - 100.0) / 100.0
    years = total_bars / bars_per_year
    if years <= 0:
        return 0.0

    if total_return <= -1:
        return -100.0

    annual_return = ((1 + total_return) ** (1 / years) - 1) * 100
    return annual_return


def monte_carlo_permutation_test(trades, n_permutations=1000, random_seed=42):
    """
    Monte Carlo Signifikanz-Test für Trading-Strategien.

    Prüft ob die beobachtete Win-Rate signifikant besser ist als eine
    zufällige Strategie (50% Win-Rate) mittels Bootstrap.

    Args:
        trades: Liste von Trade-Ergebnissen (1.0 = Win, -1.0 = Loss)
        n_permutations: Anzahl Bootstrap-Samples
        random_seed: Seed für Reproduzierbarkeit

    Returns:
        dict mit:
            - p_value: P-Wert (< 0.05 = signifikant besser als Zufall)
            - observed_pnl: Beobachtete PnL
            - observed_win_rate: Beobachtete Win-Rate
            - mean_random_pnl: Durchschnittliche PnL bei 50% Win-Rate
            - percentile: In welchem Perzentil die beobachtete PnL liegt
            - is_significant: True wenn p < 0.05
    """
    if len(trades) < 10:
        return {
            "p_value": 1.0,
            "observed_pnl": sum(trades),
            "observed_win_rate": 0.5,
            "mean_random_pnl": 0.0,
            "percentile": 50.0,
            "is_significant": False,
            "n_permutations": 0,
        }

    np.random.seed(random_seed)
    trades_arr = np.array(trades)
    n_trades = len(trades_arr)
    observed_pnl = np.sum(trades_arr)
    observed_wins = np.sum(trades_arr > 0)
    observed_wr = observed_wins / n_trades

    # Null-Hypothese: 50% Win-Rate (Zufall)
    # Generiere Bootstrap-Samples mit 50% Win-Rate
    random_pnls = []
    for _ in range(n_permutations):
        # Simuliere n_trades mit 50% Win-Rate
        random_wins = np.sum(np.random.random(n_trades) > 0.5)
        random_losses = n_trades - random_wins
        random_pnl = random_wins * 1.0 + random_losses * (-1.0)
        random_pnls.append(random_pnl)

    random_pnls = np.array(random_pnls)

    # P-Wert: Anteil der zufälligen PnLs die >= beobachtete PnL sind
    # (einseitiger Test: ist die Strategie besser als 50% Win-Rate?)
    p_value = (np.sum(random_pnls >= observed_pnl) + 1) / (n_permutations + 1)

    # Perzentil
    percentile = 100 * (np.sum(random_pnls < observed_pnl) / n_permutations)

    return {
        "p_value": p_value,
        "observed_pnl": float(observed_pnl),
        "observed_win_rate": float(observed_wr),
        "mean_random_pnl": float(np.mean(random_pnls)),
        "std_random_pnl": float(np.std(random_pnls)),
        "percentile": percentile,
        "is_significant": p_value < 0.05,
        "n_permutations": n_permutations,
    }


def monte_carlo_equity_simulation(trades, kelly_risk, rrr, n_simulations=1000, random_seed=42):
    """
    Monte Carlo Simulation der Equity-Kurve mit zufälligen Trade-Reihenfolgen.

    Berechnet Konfidenzintervalle für die finale Equity basierend auf
    verschiedenen möglichen Trade-Reihenfolgen.

    Args:
        trades: Liste von Trade-Ergebnissen (1.0 = Win, -1.0 = Loss)
        kelly_risk: Risk pro Trade (z.B. 0.02 = 2%)
        rrr: Risk-Reward-Ratio
        n_simulations: Anzahl der Simulationen
        random_seed: Seed für Reproduzierbarkeit

    Returns:
        dict mit:
            - median_equity: Median der finalen Equities
            - p5_equity: 5. Perzentil (Worst Case)
            - p95_equity: 95. Perzentil (Best Case)
            - bankruptcy_rate: Anteil der Simulationen die Bankrott gehen
            - observed_equity: Equity mit originaler Reihenfolge
    """
    if len(trades) < 10:
        return {
            "median_equity": 100.0,
            "p5_equity": 100.0,
            "p95_equity": 100.0,
            "bankruptcy_rate": 0.0,
            "observed_equity": 100.0,
            "n_simulations": 0,
        }

    np.random.seed(random_seed)
    trades_arr = np.array(trades)

    def simulate_equity(trade_sequence):
        equity = 100.0
        for t in trade_sequence:
            if t > 0:
                equity *= 1 + (kelly_risk * rrr)
            else:
                equity *= 1 - kelly_risk
            if equity <= 0:
                return 0.0
        return equity

    # Originale Equity
    observed_equity = simulate_equity(trades_arr)

    # Monte Carlo Simulationen
    final_equities = []
    bankruptcies = 0
    for _ in range(n_simulations):
        permuted = np.random.permutation(trades_arr)
        final_eq = simulate_equity(permuted)
        final_equities.append(final_eq)
        if final_eq <= 0:
            bankruptcies += 1

    final_equities = np.array(final_equities)

    return {
        "median_equity": float(np.median(final_equities)),
        "mean_equity": float(np.mean(final_equities)),
        "p5_equity": float(np.percentile(final_equities, 5)),
        "p25_equity": float(np.percentile(final_equities, 25)),
        "p75_equity": float(np.percentile(final_equities, 75)),
        "p95_equity": float(np.percentile(final_equities, 95)),
        "bankruptcy_rate": bankruptcies / n_simulations,
        "observed_equity": float(observed_equity),
        "n_simulations": n_simulations,
    }


def adjust_kelly_for_target_dd(trades, base_kelly, rrr, target_max_dd=0.30):
    """
    Passt den Kelly-Faktor an, um einen Ziel-Max-Drawdown zu erreichen.

    Args:
        trades: Liste von Trade-Ergebnissen (1.0 = Win, -1.0 = Loss)
        base_kelly: Basis Kelly-Faktor (z.B. 0.05)
        rrr: Risk-Reward-Ratio
        target_max_dd: Gewünschter maximaler Drawdown (z.B. 0.30 = 30%)

    Returns:
        dict mit:
            - adjusted_kelly: Angepasster Kelly-Faktor
            - original_dd: Max DD mit Basis-Kelly
            - adjusted_dd: Max DD mit angepasstem Kelly
            - scale_factor: Skalierungsfaktor (adjusted_kelly / base_kelly)
    """
    if not trades or base_kelly <= 0:
        return {
            "adjusted_kelly": base_kelly,
            "original_dd": 0.0,
            "adjusted_dd": 0.0,
            "scale_factor": 1.0,
        }

    # Berechne Max DD mit Basis-Kelly
    original_dd = calculate_max_drawdown(trades, base_kelly, rrr)

    if original_dd <= target_max_dd:
        # DD ist bereits unter Ziel - keine Anpassung nötig
        return {
            "adjusted_kelly": base_kelly,
            "original_dd": original_dd,
            "adjusted_dd": original_dd,
            "scale_factor": 1.0,
        }

    # Binäre Suche nach dem optimalen Kelly
    low_kelly = 0.001
    high_kelly = base_kelly

    for _ in range(20):  # Max 20 Iterationen
        mid_kelly = (low_kelly + high_kelly) / 2
        mid_dd = calculate_max_drawdown(trades, mid_kelly, rrr)

        if mid_dd > target_max_dd:
            high_kelly = mid_kelly
        else:
            low_kelly = mid_kelly

        if abs(mid_dd - target_max_dd) < 0.01:  # 1% Toleranz
            break

    adjusted_kelly = low_kelly
    adjusted_dd = calculate_max_drawdown(trades, adjusted_kelly, rrr)

    return {
        "adjusted_kelly": adjusted_kelly,
        "original_dd": original_dd,
        "adjusted_dd": adjusted_dd,
        "scale_factor": adjusted_kelly / base_kelly if base_kelly > 0 else 1.0,
    }


def simulate_with_circuit_breaker(trades, kelly_risk, rrr, pause_after_losses, pause_bars):
    """
    Simuliert Trading mit Circuit Breaker - pausiert nach N Verlusten in Serie.

    Args:
        trades: Liste von Trade-Ergebnissen (1.0 = Win, -1.0 = Loss)
        kelly_risk: Risk pro Trade
        rrr: Risk-Reward-Ratio
        pause_after_losses: Nach wie vielen Verlusten in Serie pausieren (0 = deaktiviert)
        pause_bars: Wie viele Trades/Bars pausiert wird

    Returns:
        dict mit:
            - trades_taken: Anzahl tatsächlich ausgeführter Trades
            - trades_skipped: Anzahl übersprungener Trades
            - pauses_triggered: Anzahl ausgelöster Pausen
            - final_equity: End-Equity
            - max_drawdown: Max Drawdown
            - win_rate: Effektive Win-Rate (nur ausgeführte Trades)
            - filtered_trades: Liste der tatsächlich ausgeführten Trades
    """
    if not trades or pause_after_losses <= 0:
        # Circuit Breaker deaktiviert - normale Simulation
        dd = calculate_max_drawdown(trades, kelly_risk, rrr) if trades else 0.0
        wr = sum(1 for t in trades if t > 0) / len(trades) if trades else 0.0

        # Berechne finale Equity
        equity = 100.0
        for t in trades:
            if t > 0:
                equity *= 1 + (kelly_risk * rrr)
            else:
                equity *= 1 - kelly_risk

        return {
            "trades_taken": len(trades),
            "trades_skipped": 0,
            "pauses_triggered": 0,
            "final_equity": equity,
            "max_drawdown": dd,
            "win_rate": wr,
            "filtered_trades": trades,
        }

    # Mit Circuit Breaker
    equity = 100.0
    peak = equity
    max_dd = 0.0

    consecutive_losses = 0
    pause_remaining = 0
    pauses_triggered = 0

    filtered_trades = []
    trades_skipped = 0

    for trade in trades:
        # Prüfe ob wir noch in einer Pause sind
        if pause_remaining > 0:
            pause_remaining -= 1
            trades_skipped += 1
            continue

        # Trade ausführen
        filtered_trades.append(trade)

        if trade > 0:
            equity *= 1 + (kelly_risk * rrr)
            consecutive_losses = 0
        else:
            equity *= 1 - kelly_risk
            consecutive_losses += 1

            # Circuit Breaker auslösen?
            if consecutive_losses >= pause_after_losses:
                pause_remaining = pause_bars
                pauses_triggered += 1
                consecutive_losses = 0  # Reset nach Pause

        # Drawdown tracken
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

        # Bankrott?
        if equity <= 0:
            max_dd = 1.0
            break

    wr = sum(1 for t in filtered_trades if t > 0) / len(filtered_trades) if filtered_trades else 0.0

    return {
        "trades_taken": len(filtered_trades),
        "trades_skipped": trades_skipped,
        "pauses_triggered": pauses_triggered,
        "final_equity": equity,
        "max_drawdown": max_dd,
        "win_rate": wr,
        "filtered_trades": filtered_trades,
    }


def find_optimal_circuit_breaker(trades, kelly_risk, rrr,
                                  loss_range=(3, 10),
                                  pause_range=(5, 50)):
    """
    Findet die optimalen Circuit Breaker Parameter durch Grid-Search.

    Args:
        trades: Liste von Trade-Ergebnissen
        kelly_risk: Risk pro Trade
        rrr: Risk-Reward-Ratio
        loss_range: (min, max) Verluste vor Pause
        pause_range: (min, max) Pause-Länge

    Returns:
        dict mit optimalen Parametern und Metriken
    """
    if not trades:
        return {
            "optimal_pause_after_losses": 0,
            "optimal_pause_bars": 0,
            "improvement": 0.0,
            "baseline_dd": 0.0,
            "optimized_dd": 0.0,
        }

    # Baseline ohne Circuit Breaker
    baseline = simulate_with_circuit_breaker(trades, kelly_risk, rrr, 0, 0)
    baseline_dd = baseline["max_drawdown"]
    baseline_equity = baseline["final_equity"]

    best_score = baseline_equity / (1 + baseline_dd)  # Rendite/Risiko Verhältnis
    best_params = (0, 0)
    best_result = baseline

    # Grid-Search
    for pause_after in range(loss_range[0], loss_range[1] + 1):
        for pause_bars in range(pause_range[0], pause_range[1] + 1, 5):  # 5er Schritte
            result = simulate_with_circuit_breaker(
                trades, kelly_risk, rrr, pause_after, pause_bars
            )

            # Score: Maximiere Rendite bei minimiertem DD
            # Bestrafe zu viele übersprungene Trades
            skip_penalty = 1 - (result["trades_skipped"] / len(trades)) * 0.5
            score = (result["final_equity"] / (1 + result["max_drawdown"])) * skip_penalty

            if score > best_score:
                best_score = score
                best_params = (pause_after, pause_bars)
                best_result = result

    return {
        "optimal_pause_after_losses": best_params[0],
        "optimal_pause_bars": best_params[1],
        "baseline_dd": baseline_dd,
        "baseline_equity": baseline_equity,
        "optimized_dd": best_result["max_drawdown"],
        "optimized_equity": best_result["final_equity"],
        "trades_taken": best_result["trades_taken"],
        "trades_skipped": best_result["trades_skipped"],
        "pauses_triggered": best_result["pauses_triggered"],
        "improvement": (best_result["final_equity"] - baseline_equity) / baseline_equity if baseline_equity > 0 else 0,
        "dd_reduction": baseline_dd - best_result["max_drawdown"],
    }
