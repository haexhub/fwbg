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
                       max_bars=None, trailing_start=0.5, timestamps=None, symbol=None):
    """
    Simuliert einen Trade.

    - Trade läuft bis TP oder SL erreicht wird (kein Timeout-Exit!)
    - Bei gleichzeitigem TP/SL im selben Bar: Schaut in 15-Min-Daten (falls verfügbar)

    Args:
        timestamps: Optional - Array von Timestamps für M15-Lookup
        symbol: Optional - Symbol-Name für M15-Lookup

    Returns: (result, bars_held) - result: 1.0=Win, -1.0=Loss, 0.0=Invalid
    """
    if max_bars is None:
        max_bars = MAX_TRADE_BARS

    if idx + max_bars >= len(closes):
        return 0.0, 0

    tp_distance = spread * tp_m
    sl_distance = spread * sl_m
    slippage = spread * 0.5

    if direction == 1:  # Long
        entry = closes[idx] + spread + slippage
        tp = entry + tp_distance - slippage
        sl = entry - sl_distance - slippage
    else:  # Short
        entry = closes[idx] - spread - slippage
        tp = entry - tp_distance + slippage
        sl = entry + sl_distance + slippage

    for j in range(idx + 1, min(idx + max_bars, len(closes))):
        if direction == 1:  # Long
            tp_hit = highs[j] >= tp
            sl_hit = lows[j] <= sl

            if tp_hit and sl_hit:
                # Beide im selben Bar - versuche Sub-Stunden Lookup
                if timestamps is not None and symbol is not None:
                    result = resolve_tp_sl_collision(symbol, timestamps[j], direction, tp, sl)
                    if result is not None:
                        return result, j - idx
                # Fallback: konservativ Loss
                return -1.0, j - idx
            elif tp_hit:
                return 1.0, j - idx
            elif sl_hit:
                return -1.0, j - idx

        else:  # Short
            tp_hit = lows[j] <= tp
            sl_hit = highs[j] >= sl

            if tp_hit and sl_hit:
                # Beide im selben Bar - versuche Sub-Stunden Lookup
                if timestamps is not None and symbol is not None:
                    result = resolve_tp_sl_collision(symbol, timestamps[j], direction, tp, sl)
                    if result is not None:
                        return result, j - idx
                # Fallback: konservativ Loss
                return -1.0, j - idx
            elif tp_hit:
                return 1.0, j - idx
            elif sl_hit:
                return -1.0, j - idx

    return 0.0, max_bars


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
