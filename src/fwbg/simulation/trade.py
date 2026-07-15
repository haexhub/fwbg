"""
Trade-Simulation und Metriken
"""
import numpy as np
import pandas as pd
from fwbg.data import config as data_config
from fwbg.utils.metrics import (
    compute_drawdown_from_equity,
    simulate_equity_from_binary_returns,
    simulate_equity_from_returns as _simulate_equity_from_returns,
    pnl_to_returns as _pnl_to_returns,
)


def compute_session_mask(timestamps, session_start_hour, session_end_hour,
                         ohlc=None):
    """Return boolean array: True for bars within trading session hours.

    Used to restrict trade exits to session hours only.
    Trades may run through off-session periods (overnight holds allowed),
    but TP/SL/timeout exits only trigger on in-session bars.

    Args:
        timestamps: DatetimeIndex of bar timestamps
        session_start_hour: Session start hour (UTC)
        session_end_hour: Session end hour (UTC)
        ohlc: Optional (opens, highs, lows, closes) arrays. When provided,
              flat bars (O==H==L==C) are excluded from the mask. This
              filters out weekends, holidays, and early closes automatically.

    Returns:
        bool array of length n.  True = bar is within session.
    """
    hours = timestamps.hour

    if session_start_hour < session_end_hour:
        mask = (hours >= session_start_hour) & (hours < session_end_hour)
    else:  # midnight crossing (e.g. ASX200 23:00-06:00)
        mask = (hours >= session_start_hour) | (hours < session_end_hour)

    mask = np.asarray(mask, dtype=bool)

    if ohlc is not None:
        opens, highs, lows, closes = ohlc
        flat = (opens == highs) & (highs == lows) & (lows == closes)
        mask = mask & ~np.asarray(flat, dtype=bool)

    return mask


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

    equity = simulate_equity_from_binary_returns(returns, risk_per_trade, rrr)
    max_dd = max(compute_drawdown_from_equity(equity), 0.01)
    total_return = (equity[-1] - equity[0]) / equity[0]
    return min(10.0, total_return / max_dd)


def calculate_calmar_from_returns(trade_returns):
    """Calmar Ratio from pre-computed per-trade returns (for variable position sizing)."""
    if not trade_returns:
        return 0.0

    equity = _simulate_equity_from_returns(trade_returns)
    max_dd = max(compute_drawdown_from_equity(equity), 0.01)
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
    if not data_config.DATA_PATH:
        return None, None

    with _cache_lock:
        # Versuche 15-Min-Daten
        if symbol not in _m15_cache:
            m15_path = f"{data_config.DATA_PATH}/{symbol}_MINUTE_15.csv"
            _m15_cache[symbol] = _load_ohlc_csv(m15_path)

        if _m15_cache[symbol] is not None:
            return _m15_cache[symbol], 15

        # Fallback: 30-Min-Daten
        if symbol not in _m30_cache:
            m30_path = f"{data_config.DATA_PATH}/{symbol}_MINUTE_30.csv"
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

    # Sub-bars within the hour. pd.DataFrame.loc is inclusive on both ends, so
    # we must use an exclusive right edge to avoid leaking the next hour's bar
    # into the TP/SL resolution (lookahead bias).
    hour_start = hour_timestamp
    hour_end_exclusive = hour_timestamp + pd.Timedelta(hours=1)

    try:
        sub_bars = sub_df.loc[(sub_df.index >= hour_start) & (sub_df.index < hour_end_exclusive)]
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
                       opens=None, timeout_bars=None, in_session=None,
                       sl_level_abs=None, entry_delay=1,
                       breakeven_trigger=0.0, trail_distance=0.0,
                       scale_levels=None, scale_qty_mult=1.0):
    """
    Simuliert einen Trade und gibt detaillierte Informationen zurück.

    Exit-Strategy-agnostisch: nimmt fertig berechnete TP/SL-Distanzen.
    Die Distanz-Berechnung (fixed, ATR-basiert, etc.) obliegt dem Exit-Strategy-Plugin.

    - Signal bei Bar idx, Entry bei Bar idx+entry_delay
    - entry_delay=1 (default): Entry bei Open des nächsten Bars (kein Look-Ahead)
    - entry_delay=0: Entry beim Close des Signal-Bars (für Breakout-Strategien
      mit Stop-Orders am Breakout-Level)
    - Trade läuft bis TP oder SL erreicht wird
    - Bei gleichzeitigem TP/SL im selben Bar: Schaut in 15-Min-Daten (falls verfügbar)
    - Bei Timeout: Schließt zum Close-Preis und wertet als Win/Loss

    Trailing Stop:
    - breakeven_trigger: Anteil der TP-Distanz, ab dem SL auf Entry gezogen wird
      (0.5 = 50% des TP). 0.0 = kein Breakeven.
    - trail_distance: Absoluter Abstand für Trailing Stop. Nach Breakeven folgt
      der SL dem besten Preis mit diesem Abstand. 0.0 = kein Trailing.

    Scale-In:
    - scale_levels: Liste von Float-Werten (0-1), Retracement-Fraktionen der
      SL-Distanz. Bei jedem Level wird eine zusätzliche Position eröffnet.
      None = kein Scale-In (Original-Verhalten).
    - scale_qty_mult: Quantity-Multiplikator für Scale-In-Positionen (1.0 = gleiche
      Größe wie Initial-Position).

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
        entry_delay: Bars zwischen Signal und Entry (0=sofort, 1=nächster Bar)
        breakeven_trigger: Anteil TP-Distanz für Breakeven (0.0=aus, 0.5=50%)
        trail_distance: Trailing-Abstand vom besten Preis (0.0=aus)
        scale_levels: Liste von Retracement-Fraktionen für Scale-In (None=aus)
        scale_qty_mult: Quantity pro Scale-In (1.0=gleich wie Initial)

    Returns:
        dict mit Trade-Details oder None bei ungültigem Trade
    """
    entry_idx = idx + entry_delay
    if entry_idx >= len(closes):
        return None

    # max_bars bestimmt wie weit wir maximal simulieren
    # None oder sehr hohe Werte = bis zum Ende der Daten
    if max_bars is None or max_bars > len(closes) - entry_idx:
        max_bars = len(closes) - entry_idx

    slippage = spread * 0.5

    if entry_delay == 0:
        # Sofort-Entry: Close des Signal-Bars (Breakout mit Stop-Order)
        entry_price = closes[idx]
    elif opens is not None:
        entry_price = opens[entry_idx]
    else:
        entry_price = closes[entry_idx]

    # WICHTIG: Slippage wirkt IMMER gegen den Trader:
    # - Entry-Slippage: schlechterer Einstieg (in entry eingerechnet)
    # - Exit-Slippage: TP/SL sind Trigger-Levels, nicht Exit-Preise
    if direction == 1:  # Long
        entry = entry_price + spread + slippage  # Kaufe teurer
        tp = entry + tp_distance  # TP-Level (Trigger)
        sl = sl_level_abs if sl_level_abs is not None else entry - sl_distance
        if sl_level_abs is not None and sl >= entry:
            # Level auf der falschen Seite des Entry (z. B. Distanz-Spalte als
            # Level missbraucht oder Gap über das Level) → Distanz-SL statt
            # sofortigem Phantom-Exit.
            sl = entry - sl_distance
    else:  # Short
        entry = entry_price - spread - slippage  # Verkaufe billiger
        tp = entry - tp_distance  # TP-Level (Trigger)
        sl = sl_level_abs if sl_level_abs is not None else entry + sl_distance
        if sl_level_abs is not None and sl <= entry:
            sl = entry + sl_distance

    # --- Scale-in preparation ---
    use_scale_in = scale_levels is not None and len(scale_levels) > 0
    if use_scale_in:
        positions = [(entry, 1.0, entry_idx)]
        avg_price = entry
        total_qty = 1.0
        n_fills = 1
        # Precompute trigger prices
        scale_trigger_prices = []
        levels_filled = []
        for level in scale_levels:
            if direction == 1:
                scale_trigger_prices.append(entry - level * sl_distance)
            else:
                scale_trigger_prices.append(entry + level * sl_distance)
            levels_filled.append(False)
    else:
        avg_price = entry
        total_qty = 1.0
        n_fills = 1
        positions = None

    # Hilfsfunktion für Rückgabe
    # Captures tp, sl, avg_price, total_qty from outer scope (may be updated by scale-in)
    def make_result(result, exit_idx, exit_price):
        bars_held = exit_idx - entry_idx

        # PnL calculation: use avg_price for scale-in, entry for normal trades
        if use_scale_in:
            if direction == 1:
                pnl_raw = (exit_price - avg_price) * total_qty
            else:
                pnl_raw = (avg_price - exit_price) * total_qty
        else:
            if direction == 1:
                pnl_raw = exit_price - entry
            else:
                pnl_raw = entry - exit_price

        # Breakeven-stop exits where the trailing SL barely moved above entry
        # produce near-zero pnl_raw (floating-point noise, e.g. 7e-8) and
        # must not count as wins. Transaction costs are ALREADY inside the
        # entry price (entry = raw ± (spread + slippage)), so pnl_raw > 0
        # means costs are cleared — comparing against spread + slippage here
        # would double-count them and flip every genuine TP hit whose
        # tp_distance <= spread + slippage into a loss.
        noise_eps = 1e-5 * max(abs(entry), 1.0)
        if result > 0 and pnl_raw <= noise_eps:
            result = -1.0

        res = {
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
            "sl_distance": float(abs(entry - sl)) if sl_level_abs is not None else float(sl_distance),
            "pnl_raw": float(pnl_raw),
            "mae": float(entry - mae_price) if direction == 1 else float(mae_price - entry),
            "mfe": float(mfe_price - entry) if direction == 1 else float(entry - mfe_price),
        }

        if use_scale_in:
            res["avg_entry_price"] = float(avg_price)
            res["total_qty"] = float(total_qty)
            res["n_fills"] = n_fills
            res["scale_in_fills"] = [
                {
                    "price": float(p),
                    "qty": float(q),
                    "fill_time": str(timestamps[bi]) if timestamps is not None else None,
                }
                for p, q, bi in positions[1:]
            ]

        return res

    # MAE/MFE tracking (always active, near-zero overhead)
    mae_price = entry  # worst price against the trade
    mfe_price = entry  # best price for the trade

    # Trailing state
    use_trailing = breakeven_trigger > 0.0 or trail_distance > 0.0
    trailing_active = breakeven_trigger <= 0.0 and trail_distance > 0.0
    best_price = entry
    if direction == 1:
        be_trigger_price = entry + tp_distance * breakeven_trigger if breakeven_trigger > 0.0 else 0.0
    else:
        be_trigger_price = entry - tp_distance * breakeven_trigger if breakeven_trigger > 0.0 else 0.0

    # Session-aware exit: only check TP/SL/timeout on in-session bars.
    # Trades may run through off-session periods (overnight holds).
    # When in_session is None, all bars are eligible for exits (original behavior).
    session_bars_elapsed = 0

    for j in range(entry_idx, min(entry_idx + max_bars, len(closes))):
        # --- Scale-in trigger check (before session/TP/SL checks) ---
        # Scale-in fills are price triggers, not time-dependent (fire on any bar).
        if use_scale_in:
            for k in range(len(scale_levels)):
                if levels_filled[k]:
                    continue
                trigger = scale_trigger_prices[k]
                if direction == 1:
                    if lows[j] <= trigger and trigger > sl:
                        positions.append((trigger, scale_qty_mult, j))
                        total_qty = sum(q for _, q, _ in positions)
                        avg_price = sum(p * q for p, q, _ in positions) / total_qty
                        tp = avg_price + tp_distance
                        levels_filled[k] = True
                        n_fills += 1
                        if breakeven_trigger > 0.0:
                            be_trigger_price = avg_price + tp_distance * breakeven_trigger
                else:
                    if highs[j] >= trigger and trigger < sl:
                        positions.append((trigger, scale_qty_mult, j))
                        total_qty = sum(q for _, q, _ in positions)
                        avg_price = sum(p * q for p, q, _ in positions) / total_qty
                        tp = avg_price - tp_distance
                        levels_filled[k] = True
                        n_fills += 1
                        if breakeven_trigger > 0.0:
                            be_trigger_price = avg_price - tp_distance * breakeven_trigger

        # Track MAE/MFE on ALL bars (including off-session)
        if direction == 1:
            if lows[j] < mae_price:
                mae_price = lows[j]
            if highs[j] > mfe_price:
                mfe_price = highs[j]
        else:
            if highs[j] > mae_price:
                mae_price = highs[j]
            if lows[j] < mfe_price:
                mfe_price = lows[j]

        # Skip off-session bars — no exits outside trading hours
        if in_session is not None and not in_session[j]:
            continue

        session_bars_elapsed += 1

        # Timeout check (counts only session bars when session-aware)
        if timeout_bars is not None and timeout_bars > 0 and session_bars_elapsed >= timeout_bars:
            exit_price = closes[j]
            if use_scale_in:
                if direction == 1:
                    pnl = (exit_price - avg_price) * total_qty
                else:
                    pnl = (avg_price - exit_price) * total_qty
            else:
                if direction == 1:
                    pnl = exit_price - entry
                else:
                    pnl = entry - exit_price
            result = 1.0 if pnl > 0 else -1.0
            trade_result = make_result(result, j, exit_price)
            trade_result["exit_reason"] = "timeout"
            trade_result["timeout_bars"] = timeout_bars
            return trade_result

        # --- Trailing stop logic ---
        if use_trailing:
            # Track best price
            if direction == 1:
                if highs[j] > best_price:
                    best_price = highs[j]
            else:
                if lows[j] < best_price:
                    best_price = lows[j]

            # Breakeven trigger check
            if not trailing_active and breakeven_trigger > 0.0:
                if direction == 1 and best_price >= be_trigger_price:
                    trailing_active = True
                    be_ref = avg_price if use_scale_in else entry
                    if be_ref > sl:
                        sl = be_ref
                elif direction == -1 and best_price <= be_trigger_price:
                    trailing_active = True
                    be_ref = avg_price if use_scale_in else entry
                    if be_ref < sl:
                        sl = be_ref

            # Trail SL behind best price
            if trailing_active and trail_distance > 0.0:
                if direction == 1:
                    new_sl = best_price - trail_distance
                    if new_sl > sl:
                        sl = new_sl
                else:
                    new_sl = best_price + trail_distance
                    if new_sl < sl:
                        sl = new_sl

        # --- TP/SL hit detection ---
        # For SL result with scale-in, compare sl vs avg_price instead of entry
        sl_ref = avg_price if use_scale_in else entry

        if direction == 1:  # Long
            tp_hit = highs[j] >= tp
            sl_hit = lows[j] <= sl

            if tp_hit and sl_hit:
                if timestamps is not None and symbol is not None:
                    result = resolve_tp_sl_collision(symbol, timestamps[j], direction, tp, sl)
                    if result is not None:
                        exit_price = tp if result > 0 else sl
                        return make_result(result, j, exit_price)
                # After breakeven, SL hit is a win if SL > avg_price (or entry)
                sl_result = 1.0 if sl > sl_ref else -1.0
                return make_result(sl_result, j, sl)
            elif tp_hit:
                return make_result(1.0, j, tp)
            elif sl_hit:
                sl_result = 1.0 if sl > sl_ref else -1.0
                return make_result(sl_result, j, sl)

        else:  # Short
            tp_hit = lows[j] <= tp
            sl_hit = highs[j] >= sl

            if tp_hit and sl_hit:
                if timestamps is not None and symbol is not None:
                    result = resolve_tp_sl_collision(symbol, timestamps[j], direction, tp, sl)
                    if result is not None:
                        exit_price = tp if result > 0 else sl
                        return make_result(result, j, exit_price)
                sl_result = 1.0 if sl < sl_ref else -1.0
                return make_result(sl_result, j, sl)
            elif tp_hit:
                return make_result(1.0, j, tp)
            elif sl_hit:
                sl_result = 1.0 if sl < sl_ref else -1.0
                return make_result(sl_result, j, sl)

    # Kein Exit (weder TP/SL noch Timeout innerhalb max_bars)
    return None


def analyze_sl_potential(trades, closes, highs, lows, max_scan_bars=500):
    """For losing trades, check if TP would have been reached with a wider SL.

    Scans forward from each losing trade's entry point WITHOUT SL constraint
    to determine what SL would have been needed for the trade to hit TP.

    Enriches each losing trade dict in-place with:
      - potential_tp_reached: bool - would TP have been hit without SL?
      - required_mae: float - max adverse excursion needed to reach TP
      - bars_to_potential_tp: int - bars from entry until TP would be reached

    Args:
        trades: list of trade dicts (modified in-place for losing trades)
        closes: Close prices array
        highs: High prices array
        lows: Low prices array
        max_scan_bars: maximum bars to scan forward from entry
    """
    for trade in trades:
        if trade.get("result", 0) >= 0:
            continue  # Only analyze losing trades

        entry_idx = trade["entry_idx"]
        entry = trade["entry_price"]
        tp = trade["tp_level"]
        is_long = trade["direction"] == "LONG"

        worst = entry
        end = min(entry_idx + max_scan_bars, len(closes))

        for j in range(entry_idx, end):
            if is_long:
                if lows[j] < worst:
                    worst = lows[j]
                if highs[j] >= tp:
                    trade["potential_tp_reached"] = True
                    trade["required_mae"] = float(entry - worst)
                    trade["bars_to_potential_tp"] = j - entry_idx
                    break
            else:
                if highs[j] > worst:
                    worst = highs[j]
                if lows[j] <= tp:
                    trade["potential_tp_reached"] = True
                    trade["required_mae"] = float(worst - entry)
                    trade["bars_to_potential_tp"] = j - entry_idx
                    break
        else:
            # TP never reached within scan window
            trade["potential_tp_reached"] = False
            if is_long:
                trade["required_mae"] = float(entry - worst)
            else:
                trade["required_mae"] = float(worst - entry)


def analyze_tp_potential(trades, closes, highs, lows, max_scan_bars=500):
    """For winning trades, check how far the price continued after TP was hit.

    Scans forward from each winning trade's exit point to determine how much
    additional profit could have been captured with a wider TP.

    Enriches each winning trade dict in-place with:
      - continuation_mfe: float - max favorable excursion beyond TP level
      - continuation_mae: float - max adverse excursion after TP (reversal depth)
      - bars_of_continuation: int - bars until price reversed to entry level

    Args:
        trades: list of trade dicts (modified in-place for winning trades)
        closes: Close prices array
        highs: High prices array
        lows: Low prices array
        max_scan_bars: maximum bars to scan forward from exit
    """
    for trade in trades:
        if trade.get("result", 0) <= 0:
            continue  # Only analyze winning trades

        exit_idx = trade["exit_idx"]
        entry = trade["entry_price"]
        tp = trade["tp_level"]
        is_long = trade["direction"] == "LONG"

        best_beyond_tp = tp  # best price beyond TP
        worst_after_tp = tp  # worst price after TP (reversal tracking)
        end = min(exit_idx + max_scan_bars, len(closes))

        for j in range(exit_idx + 1, end):
            if is_long:
                if highs[j] > best_beyond_tp:
                    best_beyond_tp = highs[j]
                if lows[j] < worst_after_tp:
                    worst_after_tp = lows[j]
                # Stop if price reverses back to entry
                if lows[j] <= entry:
                    break
            else:
                if lows[j] < best_beyond_tp:
                    best_beyond_tp = lows[j]
                if highs[j] > worst_after_tp:
                    worst_after_tp = highs[j]
                if highs[j] >= entry:
                    break

        if is_long:
            trade["continuation_mfe"] = float(best_beyond_tp - tp)
            trade["continuation_mae"] = float(tp - worst_after_tp)
        else:
            trade["continuation_mfe"] = float(tp - best_beyond_tp)
            trade["continuation_mae"] = float(worst_after_tp - tp)


def attach_regime_labels(trades, df):
    """Enrich each trade dict in-place with vol_regime and trend_regime at entry.

    Descriptive labels for post-hoc diagnostics (Plan 010 WP5) — like
    analyze_sl_potential/analyze_tp_potential, these are computed from the
    whole fold-test window's own distribution, not a rolling/expanding one,
    since they're for post-hoc analysis rather than a trading decision.

    - vol_regime: ATR tercile over the window ("low" / "medium" / "high").
      Uses the "_atr"/"vol_atr" column if the volatility indicator plugin is
      configured, else computes ATR(14) directly.
    - trend_regime: ADX(14) bucket ("ranging" <20, "trending" 20-40,
      "strong_trend" >=40 — standard interpretation). Uses "adx_14" if the
      adx indicator plugin is configured, else computes it directly.

    Args:
        trades: list of trade dicts (modified in-place)
        df: OHLC DataFrame for the fold's test window (same one trades were
            simulated on — entry_idx indexes into it positionally)
    """
    if not trades:
        return

    atr_col = "_atr" if "_atr" in df.columns else ("vol_atr" if "vol_atr" in df.columns else None)
    if atr_col:
        atr = df[atr_col]
    else:
        import ta

        atr = ta.volatility.average_true_range(df["H"], df["L"], df["C"], window=14)

    adx_col = "adx_14" if "adx_14" in df.columns else None
    if adx_col:
        adx = df[adx_col]
    else:
        import ta

        adx = ta.trend.adx(df["H"], df["L"], df["C"], window=14)

    atr_valid = atr.dropna()
    vol_low, vol_high = (
        (atr_valid.quantile(1 / 3), atr_valid.quantile(2 / 3)) if len(atr_valid) else (None, None)
    )
    atr_values = atr.values
    adx_values = adx.values
    n = len(df)

    def _vol_regime(v):
        if v is None or vol_low is None or (isinstance(v, float) and v != v):  # NaN check
            return None
        if v <= vol_low:
            return "low"
        if v <= vol_high:
            return "medium"
        return "high"

    def _trend_regime(v):
        if v is None or (isinstance(v, float) and v != v):
            return None
        if v < 20:
            return "ranging"
        if v < 40:
            return "trending"
        return "strong_trend"

    for trade in trades:
        idx = trade.get("entry_idx")
        if not isinstance(idx, int) or not (0 <= idx < n):
            continue
        trade["vol_regime"] = _vol_regime(atr_values[idx])
        trade["trend_regime"] = _trend_regime(adx_values[idx])


def calculate_max_drawdown(returns, risk_per_trade, rrr):
    """Berechnet Maximum Drawdown aus Trade-Returns."""
    if not returns:
        return 0.0

    equity = simulate_equity_from_binary_returns(returns, risk_per_trade, rrr)
    return compute_drawdown_from_equity(equity)


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
    """
    return _pnl_to_returns(pnl_raw, fk)


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
