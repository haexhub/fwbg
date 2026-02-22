"""
Trade-Simulation und Metriken
"""
import numpy as np
import pandas as pd
from numba import njit

from fwbg.data.config import DATA_PATH

# Import shared numba function from numba_core to avoid duplication
from fwbg.simulation.numba_core import _simulate_trade_numba


@njit(cache=True)
def compute_targets_numba(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    tp_distance: float,
    sl_distance: float,
    spread: float,
    slippage: float,
    max_bars: int,
    timeout_bars: int,
) -> tuple:
    """
    Berechnet Long/Short Targets für alle Bars.

    HINWEIS: Sequentiell, um Thread-Kontention mit XGBoost zu vermeiden.
    Numba JIT macht die Berechnung bereits sehr schnell.

    Returns:
        (targets_long, targets_short) - Arrays mit 1.0 für Win, 0.0 sonst
    """
    n = len(closes)
    targets_long = np.zeros(n, dtype=np.float64)
    targets_short = np.zeros(n, dtype=np.float64)

    # Sequentiell - Parallelisierung auf höherer Ebene (Feature-Gruppen)
    for i in range(n - 1):
        # Long Trade
        result_long, _, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, 1,
            tp_distance, sl_distance, spread, slippage, max_bars, timeout_bars
        )
        if result_long == 1.0:
            targets_long[i] = 1.0

        # Short Trade
        result_short, _, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows, i, -1,
            tp_distance, sl_distance, spread, slippage, max_bars, timeout_bars
        )
        if result_short == 1.0:
            targets_short[i] = 1.0

    return targets_long, targets_short


# Cache für Sub-Stunden-Daten (wird einmal pro Symbol geladen)
# Thread-safe durch Lock geschützt
import threading  # noqa: E402
_cache_lock = threading.Lock()
_m15_cache = {}  # 15-Min-Daten
_m30_cache = {}  # 30-Min-Daten


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, trades_per_year=None):
    """
    Berechnet annualisierte Sharpe Ratio aus Trade-Returns.

    Args:
        returns: Liste von Trade-Returns
        risk_free_rate: Risikofreier Zinssatz (Default: 0)
        trades_per_year: Anzahl Trades pro Jahr für Annualisierung.
                        Wenn None, wird sqrt(len(returns)) verwendet (konservativ)

    Returns:
        Annualisierte Sharpe Ratio
    """
    if len(returns) < 2:
        return 0.0
    excess_returns = np.array(returns) - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0

    # Annualisierungsfaktor basierend auf tatsächlicher Trade-Frequenz
    # Wenn nicht angegeben: konservativ sqrt(n_trades) verwenden
    if trades_per_year is None:
        # Konservative Schätzung: Trades im Sample entsprechen einem Jahr
        annualization = np.sqrt(len(returns))
    else:
        annualization = np.sqrt(trades_per_year)

    return np.mean(excess_returns) / np.std(excess_returns) * annualization


def calculate_equity_smoothness(trades, risk_per_trade, rrr, window_size=50):
    """
    Berechnet einen Smoothness-Score für die Equity-Kurve.

    Höherer Score = glattere Equity-Kurve mit kleineren Sprüngen.
    Basiert auf der Varianz der rollenden Returns.

    Args:
        trades: Liste von Trade-Ergebnissen (1.0 = Win, -1.0 = Loss)
        risk_per_trade: Risk pro Trade
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
            returns.append(risk_per_trade * rrr)
        else:
            returns.append(-risk_per_trade)

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
    # Normalisiere Volatilität auf 0-1 Skala (5% risk = ~5% max single move)
    vol_score = 1.0 / (1.0 + return_volatility * 10)

    smoothness_score = (vol_score * 0.5 + consistency * 0.5)

    return {
        "smoothness_score": float(smoothness_score),
        "return_volatility": float(return_volatility),
        "max_single_move": float(max_single_move),
        "sortino_ratio": float(min(sortino, 10.0)),  # Cap bei 10
        "consistency": float(consistency),
    }


def calculate_calmar_ratio(returns, risk_per_trade, rrr):
    """Berechnet Calmar Ratio (Return / Max Drawdown)."""
    if not returns:
        return 0.0

    equity = [100.0]
    for r in returns:
        if r > 0:
            equity.append(equity[-1] * (1 + risk_per_trade * rrr))
        else:
            equity.append(equity[-1] * (1 - risk_per_trade))

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


def calculate_calmar_from_returns(trade_returns):
    """Calmar Ratio from pre-computed per-trade returns (for variable position sizing)."""
    if not trade_returns:
        return 0.0

    equity = [100.0]
    for r in trade_returns:
        equity.append(equity[-1] * (1 + r))

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
    Lädt Sub-Stunden-Daten für ein Symbol (mit Thread-safe Caching).
    Versucht zuerst 15-Min, dann 30-Min als Fallback.

    Returns: (DataFrame, resolution) oder (None, None)
    """
    with _cache_lock:
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


def simulate_pro_trade(closes, highs, lows, idx, direction, tp_distance, sl_distance, spread,
                       max_bars=None, timestamps=None, symbol=None,
                       opens=None, timeout_bars=None):
    """
    Simuliert einen Trade und gibt detaillierte Informationen zurück.

    Exit-Strategy-agnostisch: nimmt fertig berechnete TP/SL-Distanzen.
    Die Distanz-Berechnung (fixed, ATR-basiert, etc.) obliegt dem Exit-Strategy-Plugin.

    - Signal bei Bar idx, Entry bei Open von Bar idx+1 (kein Look-Ahead!)
    - Trade läuft bis TP oder SL erreicht wird
    - Bei gleichzeitigem TP/SL im selben Bar: Schaut in 15-Min-Daten (falls verfügbar)
    - Bei Timeout: Schließt zum Close-Preis und wertet als Win/Loss

    Args:
        closes: Close-Preise Array
        highs: High-Preise Array
        lows: Low-Preise Array
        idx: Index des Signal-Bars
        direction: 1 für Long, -1 für Short
        tp_distance: Take-Profit Distanz in Preiseinheiten
        sl_distance: Stop-Loss Distanz in Preiseinheiten
        spread: Spread des Assets (für Slippage-Berechnung)
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
            - exit_price: Exit-Preis (TP, SL, oder Close bei Timeout)
            - tp_level, sl_level: TP/SL Levels
            - spread, slippage, total_cost: Kosten
            - tp_distance, sl_distance: Distanzen in Preiseinheiten
            - pnl_raw: PnL vor Positionsgrößen-Berechnung
            - bars_held: Dauer in Bars
    """
    # Entry bei idx+1 (nächster Bar nach Signal)
    entry_idx = idx + 1
    if entry_idx >= len(closes):
        return None

    # max_bars bestimmt wie weit wir maximal simulieren
    # None oder sehr hohe Werte = bis zum Ende der Daten
    if max_bars is None or max_bars > len(closes) - entry_idx:
        max_bars = len(closes) - entry_idx

    slippage = spread * 0.5

    # Entry: Open des nächsten Bars (realistisch, kein Look-Ahead)
    # WICHTIG: Kein Fallback auf closes[idx] - das wäre Lookahead Bias!
    if opens is not None:
        entry_price = opens[entry_idx]
    else:
        # Fallback: Close des ENTRY-Bars (nicht Signal-Bar!) als Approximation
        # Dies ist konservativ - wir nehmen an Entry passiert zum Close des Entry-Bars
        entry_price = closes[entry_idx]

    # WICHTIG: Slippage wirkt IMMER gegen den Trader:
    # - Entry-Slippage: schlechterer Einstieg (in entry eingerechnet)
    # - Exit-Slippage: TP/SL sind Trigger-Levels, nicht Exit-Preise
    if direction == 1:  # Long
        entry = entry_price + spread + slippage  # Kaufe teurer
        tp = entry + tp_distance  # TP-Level (Trigger)
        sl = entry - sl_distance  # SL-Level (Trigger)
    else:  # Short
        entry = entry_price - spread - slippage  # Verkaufe billiger
        tp = entry - tp_distance  # TP-Level (Trigger)
        sl = entry + sl_distance  # SL-Level (Trigger)

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

    # Timeout-Index berechnen (muss INNERHALB des Loops geprüft werden,
    # exakt wie in _simulate_trade_numba — sonst Mismatch zwischen Labels und Trading)
    timeout_idx = -1
    if timeout_bars is not None and timeout_bars > 0:
        timeout_idx = min(entry_idx + timeout_bars - 1, len(closes) - 1)

    for j in range(entry_idx, min(entry_idx + max_bars, len(closes))):
        # Timeout-Check ZUERST (Priorität über TP/SL, matches _simulate_trade_numba)
        if timeout_idx > 0 and j >= timeout_idx:
            exit_price = closes[j]
            if direction == 1:
                pnl = exit_price - entry
            else:
                pnl = entry - exit_price
            result = 1.0 if pnl > 0 else -1.0
            trade_result = make_result(result, j, exit_price)
            trade_result["exit_reason"] = "timeout"
            trade_result["timeout_bars"] = timeout_bars
            return trade_result

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

    # Kein Exit (weder TP/SL noch Timeout innerhalb max_bars)
    return None


def calculate_max_drawdown(returns, risk_per_trade, rrr):
    """Berechnet Maximum Drawdown aus Trade-Returns."""
    if not returns:
        return 0.0

    equity = [100.0]
    for r in returns:
        if r > 0:
            equity.append(equity[-1] * (1 + risk_per_trade * rrr))
        else:
            equity.append(equity[-1] * (1 - risk_per_trade))

    peak = equity[0]
    max_dd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak
        if dd > max_dd:
            max_dd = dd

    return max_dd


def calculate_annual_return(returns, risk_per_trade, rrr, total_bars, bars_per_year=8760):
    """Berechnet annualisierte Rendite."""
    if not returns or total_bars == 0:
        return 0.0

    equity = 100.0
    for r in returns:
        if r > 0:
            equity *= (1 + risk_per_trade * rrr)
        else:
            equity *= (1 - risk_per_trade)

    total_return = (equity - 100.0) / 100.0
    years = total_bars / bars_per_year
    if years <= 0:
        return 0.0

    if total_return <= -1:
        return -100.0

    annual_return = ((1 + total_return) ** (1 / years) - 1) * 100
    return annual_return


def pnl_to_returns(pnl_raw, fk):
    """Konvertiert tatsächliche PnL-Werte in Kelly-skalierte Per-Trade-Returns.

    Skaliert so, dass der durchschnittliche Loss-Return exakt -fk ist.
    Gewinner reflektieren die tatsächlich realisierte RRR (nicht die erwartete).

    Args:
        pnl_raw: Liste der tatsächlichen PnL-Werte (positiv=Gewinn, negativ=Verlust)
        fk: Kelly-Fraktion (Risiko pro Trade, z.B. 0.02 = 2%)

    Returns:
        Liste der Per-Trade-Returns als Anteil des Kapitals.
        Bei leerer Eingabe: leere Liste.
    """
    if not pnl_raw:
        return []
    losses = [abs(p) for p in pnl_raw if p < 0]
    scale = float(np.mean(losses)) if losses else float(np.mean(np.abs(pnl_raw))) or 1.0
    return [fk * p / scale for p in pnl_raw]


def monte_carlo_permutation_test(trade_pnls, n_permutations=1000, random_seed=42, rrr=None):
    """
    Monte Carlo Signifikanz-Test für Trading-Strategien (PnL-basiert).

    Verwendet einen Sign-Permutation-Test auf der tatsächlichen PnL-Verteilung.
    Null-Hypothese: E[PnL] = 0 (kein Edge) — jeder Trade ist gleich wahrscheinlich
    positiv oder negativ, unabhängig von seiner Größe.

    Vorteile gegenüber binärem WR-Test:
    - Kein RRR-Parameter nötig (implizit in der PnL-Verteilung enthalten)
    - Breakeven-Trades (PnL=0) fließen korrekt als Null ein
    - Asymmetrische Verteilungen werden erfasst (viele kleine Verluste, wenige große
      Gewinne können trotzdem positiven Edge haben)
    - Funktioniert korrekt für alle Exit-Strategien (atr_trailing, fixed, etc.)

    Args:
        trade_pnls: Liste tatsächlicher PnL-Werte (positiv=Gewinn, negativ=Verlust, 0=Breakeven)
        n_permutations: Anzahl Bootstrap-Samples
        random_seed: Seed für Reproduzierbarkeit
        rrr: Ignoriert (nur für Rückwärtskompatibilität)

    Returns:
        dict mit p_value, observed_pnl, observed_mean_pnl, percentile, is_significant
    """
    pnl_arr = np.array(trade_pnls, dtype=float)
    n_trades = len(pnl_arr)

    if n_trades < 10:
        return {
            "p_value": 1.0,
            "observed_pnl": float(np.sum(pnl_arr)),
            "observed_mean_pnl": 0.0,
            "observed_win_rate": 0.5,
            "mean_random_pnl": 0.0,
            "std_random_pnl": 0.0,
            "percentile": 50.0,
            "is_significant": False,
            "n_permutations": 0,
        }

    rng = np.random.default_rng(random_seed)
    observed_total = float(np.sum(pnl_arr))
    observed_mean = float(np.mean(pnl_arr))
    observed_wr = float(np.sum(pnl_arr > 0) / n_trades)
    abs_pnl = np.abs(pnl_arr)

    # Sign-Permutation: Null-Hypothese E[PnL]=0 — jeder Trade gleich wahrscheinlich
    # positiv oder negativ. Für jede Permutation: zufällige Vorzeichen auf die
    # absoluten PnL-Werte. Breakeven-Trades (abs_pnl=0) bleiben immer 0.
    random_totals = np.empty(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_trades)
        random_totals[i] = np.sum(signs * abs_pnl)

    p_value = float((np.sum(random_totals >= observed_total) + 1) / (n_permutations + 1))
    percentile = float(100 * np.sum(random_totals < observed_total) / n_permutations)

    return {
        "p_value": p_value,
        "observed_pnl": observed_total,
        "observed_mean_pnl": observed_mean,
        "observed_win_rate": observed_wr,
        "mean_random_pnl": float(np.mean(random_totals)),
        "std_random_pnl": float(np.std(random_totals)),
        "percentile": percentile,
        "is_significant": p_value < 0.05,
        "n_permutations": n_permutations,
    }


def monte_carlo_equity_simulation(trades, risk_per_trade, rrr, n_simulations=1000, random_seed=42):
    """
    Monte Carlo Simulation der Equity-Kurve mit zufälligen Trade-Reihenfolgen.

    Berechnet Konfidenzintervalle für die finale Equity basierend auf
    verschiedenen möglichen Trade-Reihenfolgen.

    Args:
        trades: Liste von Trade-Ergebnissen (1.0 = Win, -1.0 = Loss)
        risk_per_trade: Risk pro Trade (z.B. 0.02 = 2%)
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

    # Verwende isolierten RandomState statt globalem np.random.seed()
    rng = np.random.default_rng(random_seed)
    trades_arr = np.array(trades)

    def simulate_equity(trade_sequence):
        equity = 100.0
        for t in trade_sequence:
            if t > 0:
                equity *= 1 + (risk_per_trade * rrr)
            else:
                equity *= 1 - risk_per_trade
            if equity <= 0:
                return 0.0
        return equity

    # Originale Equity
    observed_equity = simulate_equity(trades_arr)

    # Monte Carlo Simulationen (verwende isolierten rng)
    final_equities = []
    bankruptcies = 0
    for _ in range(n_simulations):
        permuted = rng.permutation(trades_arr)
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


def monte_carlo_equity_from_returns(trade_returns, n_simulations=1000, random_seed=42):
    """Monte Carlo equity simulation from pre-computed per-trade returns.

    Like monte_carlo_equity_simulation but accepts variable-sized returns
    instead of binary trades + fixed risk. Used by vol_targeted_kelly.
    """
    if len(trade_returns) < 10:
        return {
            "median_equity": 100.0,
            "p5_equity": 100.0,
            "p95_equity": 100.0,
            "bankruptcy_rate": 0.0,
            "observed_equity": 100.0,
            "n_simulations": 0,
        }

    rng = np.random.default_rng(random_seed)
    returns_arr = np.array(trade_returns)

    def simulate_equity(returns_seq):
        equity = 100.0
        for r in returns_seq:
            equity *= 1 + r
            if equity <= 0:
                return 0.0
        return equity

    observed_equity = simulate_equity(returns_arr)

    final_equities = []
    bankruptcies = 0
    for _ in range(n_simulations):
        permuted = rng.permutation(returns_arr)
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


def adjust_risk_for_target_dd(trades, base_risk, rrr, target_max_dd=0.30):
    """
    Passt den Risk-Faktor an, um einen Ziel-Max-Drawdown zu erreichen.

    Args:
        trades: Liste von Trade-Ergebnissen (1.0 = Win, -1.0 = Loss)
        base_risk: Basis Risk-Faktor (z.B. 0.05)
        rrr: Risk-Reward-Ratio
        target_max_dd: Gewünschter maximaler Drawdown (z.B. 0.30 = 30%)

    Returns:
        dict mit:
            - adjusted_risk: Angepasster Risk-Faktor
            - original_dd: Max DD mit Basis-Risk
            - adjusted_dd: Max DD mit angepasstem Risk
            - scale_factor: Skalierungsfaktor (adjusted_risk / base_risk)
    """
    if not trades or base_risk <= 0:
        return {
            "adjusted_risk": base_risk,
            "original_dd": 0.0,
            "adjusted_dd": 0.0,
            "scale_factor": 1.0,
        }

    # Berechne Max DD mit Basis-Risk
    original_dd = calculate_max_drawdown(trades, base_risk, rrr)

    if original_dd <= target_max_dd:
        # DD ist bereits unter Ziel - keine Anpassung nötig
        return {
            "adjusted_risk": base_risk,
            "original_dd": original_dd,
            "adjusted_dd": original_dd,
            "scale_factor": 1.0,
        }

    # Binäre Suche nach dem optimalen Risk-Faktor
    low_risk = 0.001
    high_risk = base_risk

    for _ in range(20):  # Max 20 Iterationen
        mid_risk = (low_risk + high_risk) / 2
        mid_dd = calculate_max_drawdown(trades, mid_risk, rrr)

        if mid_dd > target_max_dd:
            high_risk = mid_risk
        else:
            low_risk = mid_risk

        if abs(mid_dd - target_max_dd) < 0.01:  # 1% Toleranz
            break

    adjusted_risk = low_risk
    adjusted_dd = calculate_max_drawdown(trades, adjusted_risk, rrr)

    return {
        "adjusted_risk": adjusted_risk,
        "original_dd": original_dd,
        "adjusted_dd": adjusted_dd,
        "scale_factor": adjusted_risk / base_risk if base_risk > 0 else 1.0,
    }


def simulate_with_circuit_breaker(trades, risk_per_trade, rrr, pause_after_losses, pause_bars):
    """
    Simuliert Trading mit Circuit Breaker - pausiert nach N Verlusten in Serie.

    Args:
        trades: Liste von Trade-Ergebnissen (1.0 = Win, -1.0 = Loss)
        risk_per_trade: Risk pro Trade
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
        dd = calculate_max_drawdown(trades, risk_per_trade, rrr) if trades else 0.0
        wr = sum(1 for t in trades if t > 0) / len(trades) if trades else 0.0

        # Berechne finale Equity
        equity = 100.0
        for t in trades:
            if t > 0:
                equity *= 1 + (risk_per_trade * rrr)
            else:
                equity *= 1 - risk_per_trade

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
            equity *= 1 + (risk_per_trade * rrr)
            consecutive_losses = 0
        else:
            equity *= 1 - risk_per_trade
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


def find_optimal_circuit_breaker(trades, risk_per_trade, rrr,
                                  loss_range=(3, 10),
                                  pause_range=(5, 50)):
    """
    Findet die optimalen Circuit Breaker Parameter durch Grid-Search.

    Args:
        trades: Liste von Trade-Ergebnissen
        risk_per_trade: Risk pro Trade
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
    baseline = simulate_with_circuit_breaker(trades, risk_per_trade, rrr, 0, 0)
    baseline_dd = baseline["max_drawdown"]
    baseline_equity = baseline["final_equity"]

    best_score = baseline_equity / (1 + baseline_dd)  # Rendite/Risiko Verhältnis
    best_params = (0, 0)
    best_result = baseline

    # Grid-Search
    for pause_after in range(loss_range[0], loss_range[1] + 1):
        for pause_bars in range(pause_range[0], pause_range[1] + 1, 5):  # 5er Schritte
            result = simulate_with_circuit_breaker(
                trades, risk_per_trade, rrr, pause_after, pause_bars
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
