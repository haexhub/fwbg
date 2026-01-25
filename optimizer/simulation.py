"""
Trade-Simulation und Metriken
"""
import numpy as np

from .config import MAX_TRADE_BARS, RELEVANCE_THRESHOLD, FEATURE_STABILITY_MIN


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


def simulate_pro_trade(closes, highs, lows, atrs, idx, direction, tp_m, sl_m, spread, max_bars=None, trailing_start=0.5):
    """
    Simuliert einen Trade mit Trailing Stop.

    - Trade läuft bis TP oder SL erreicht wird (kein Timeout-Exit!)
    - Trailing Stop aktiviert sich wenn Gewinn >= trailing_start * TP erreicht
    - Trailing Stop sichert 50% des erreichten Gewinns

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

    trailing_activated = False
    best_price = entry
    trailing_sl = sl

    for j in range(idx + 1, min(idx + max_bars, len(closes))):
        if direction == 1:  # Long
            if highs[j] > best_price:
                best_price = highs[j]
                current_profit = best_price - entry

                if current_profit >= tp_distance * trailing_start:
                    trailing_activated = True
                    new_trailing_sl = entry + (current_profit * 0.5)
                    if new_trailing_sl > trailing_sl:
                        trailing_sl = new_trailing_sl

            if highs[j] >= tp:
                return 1.0, j - idx
            if lows[j] <= (trailing_sl if trailing_activated else sl):
                if trailing_activated and trailing_sl > entry:
                    return 1.0, j - idx
                return -1.0, j - idx

        else:  # Short
            if lows[j] < best_price:
                best_price = lows[j]
                current_profit = entry - best_price

                if current_profit >= tp_distance * trailing_start:
                    trailing_activated = True
                    new_trailing_sl = entry - (current_profit * 0.5)
                    if new_trailing_sl < trailing_sl:
                        trailing_sl = new_trailing_sl

            if lows[j] <= tp:
                return 1.0, j - idx
            if highs[j] >= (trailing_sl if trailing_activated else sl):
                if trailing_activated and trailing_sl < entry:
                    return 1.0, j - idx
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
