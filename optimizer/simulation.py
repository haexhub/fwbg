"""
Trade-Simulation und Metriken
"""
import numpy as np
import pandas as pd

from .config import MAX_TRADE_BARS, RELEVANCE_THRESHOLD, FEATURE_STABILITY_MIN, DATA_PATH


# Cache für 15-Min-Daten (wird einmal pro Symbol geladen)
_m15_cache = {}


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


def load_m15_data(symbol):
    """Lädt 15-Min-Daten für ein Symbol (mit Caching)."""
    if symbol in _m15_cache:
        return _m15_cache[symbol]

    m15_path = f"{DATA_PATH}/{symbol}_MINUTE_15.csv"
    try:
        df = pd.read_csv(m15_path, parse_dates=["Time"], index_col="Time")
        _m15_cache[symbol] = df
        return df
    except Exception:
        _m15_cache[symbol] = None
        return None


def resolve_tp_sl_collision_m15(symbol, hour_timestamp, direction, tp, sl):
    """
    Bei gleichzeitigem TP/SL Hit: Schaut in 15-Min-Daten um die Reihenfolge zu bestimmen.

    Returns: 1.0 (TP zuerst), -1.0 (SL zuerst), None (keine M15-Daten)
    """
    m15_df = load_m15_data(symbol)
    if m15_df is None:
        return None

    # Finde die 4 15-Min-Bars innerhalb der Stunde
    hour_start = hour_timestamp
    hour_end = hour_timestamp + pd.Timedelta(hours=1)

    try:
        m15_bars = m15_df.loc[hour_start:hour_end]
        if len(m15_bars) == 0:
            return None
    except Exception:
        return None

    # Gehe durch die 15-Min-Bars und prüfe was zuerst passiert
    for _, bar in m15_bars.iterrows():
        if direction == 1:  # Long
            tp_hit = bar["H"] >= tp
            sl_hit = bar["L"] <= sl
        else:  # Short
            tp_hit = bar["L"] <= tp
            sl_hit = bar["H"] >= sl

        if tp_hit and sl_hit:
            # Auch im 15-Min-Bar beide erreicht - konservativ Loss
            return -1.0
        elif tp_hit:
            return 1.0
        elif sl_hit:
            return -1.0

    # Keines erreicht in M15 (sollte nicht passieren)
    return None


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
                # Beide im selben Bar - versuche M15 Lookup
                if timestamps is not None and symbol is not None:
                    result = resolve_tp_sl_collision_m15(symbol, timestamps[j], direction, tp, sl)
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
                # Beide im selben Bar - versuche M15 Lookup
                if timestamps is not None and symbol is not None:
                    result = resolve_tp_sl_collision_m15(symbol, timestamps[j], direction, tp, sl)
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
