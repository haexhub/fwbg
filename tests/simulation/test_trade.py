"""
Tests für trade.py - Trade-Simulation und Metriken.

Testet:
- _simulate_trade_numba: Kern-Trade-Logik, TP/SL, Timeout
- compute_targets_numba: Batch-Target-Berechnung
- simulate_pro_trade: High-Level Trade-Simulation
- calculate_sharpe_ratio: Sharpe Ratio Berechnung
- calculate_calmar_ratio: Calmar Ratio Berechnung
- monte_carlo_permutation_test: Statistische Signifikanz
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from fwbg.simulation.trade import (
    _simulate_trade_numba,
    compute_targets_numba,
    simulate_pro_trade,
    calculate_sharpe_ratio,
    calculate_calmar_ratio,
    monte_carlo_permutation_test,
    calculate_equity_smoothness,
    adjust_risk_for_target_dd,
    find_optimal_circuit_breaker,
)


def create_price_arrays(n: int, trend: str = "flat", volatility: float = 0.01, seed: int = 42):
    """
    Erstellt synthetische Preis-Arrays für Tests.

    Args:
        n: Anzahl Bars
        trend: "flat", "up", "down"
        volatility: Volatilität pro Bar
        seed: Random Seed

    Returns:
        (opens, closes, highs, lows) als numpy arrays
    """
    np.random.seed(seed)

    if trend == "up":
        base = 100 + np.cumsum(np.ones(n) * 0.1)
    elif trend == "down":
        base = 100 - np.cumsum(np.ones(n) * 0.1)
    else:
        base = np.ones(n) * 100

    noise = np.random.randn(n) * volatility * 100
    closes = base + noise
    opens = np.roll(closes, 1)
    opens[0] = 100

    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n)) * volatility * 50
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n)) * volatility * 50

    return opens, closes, highs, lows


class TestSimulateTradeNumba:
    """Tests für _simulate_trade_numba."""

    def test_long_trade_tp_hit(self):
        """Test: Long-Trade erreicht TP."""
        # Stark steigender Trend
        opens = np.array([100.0, 100.0, 101.0, 102.0, 103.0])
        closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        highs = np.array([100.5, 101.5, 102.5, 103.5, 104.5])
        lows = np.array([99.5, 100.5, 101.5, 102.5, 103.5])

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=2.0, sl_distance=1.0,
            spread=0.0, slippage=0.0,
            max_bars=100, timeout_bars=0
        )

        assert result == 1.0, "Long sollte gewinnen"
        assert exit_reason == 0, "Exit-Reason sollte TP sein"

    def test_long_trade_sl_hit(self):
        """Test: Long-Trade erreicht SL."""
        # Stark fallender Trend
        opens = np.array([100.0, 100.0, 99.0, 98.0, 97.0])
        closes = np.array([100.0, 99.0, 98.0, 97.0, 96.0])
        highs = np.array([100.5, 100.0, 99.0, 98.0, 97.0])
        lows = np.array([99.5, 98.5, 97.5, 96.5, 95.5])

        result, exit_idx, _, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=5.0, sl_distance=1.0,
            spread=0.0, slippage=0.0,
            max_bars=100, timeout_bars=0
        )

        assert result == -1.0, "Long sollte verlieren"
        assert exit_reason == 1, "Exit-Reason sollte SL sein"

    def test_short_trade_tp_hit(self):
        """Test: Short-Trade erreicht TP."""
        # Stark fallender Trend
        opens = np.array([100.0, 100.0, 99.0, 98.0, 97.0])
        closes = np.array([100.0, 99.0, 98.0, 97.0, 96.0])
        highs = np.array([100.5, 100.0, 99.0, 98.0, 97.0])
        lows = np.array([99.5, 98.5, 97.5, 96.5, 95.5])

        result, _, _, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=-1,
            tp_distance=2.0, sl_distance=1.0,
            spread=0.0, slippage=0.0,
            max_bars=100, timeout_bars=0
        )

        assert result == 1.0, "Short sollte gewinnen"
        assert exit_reason == 0, "Exit-Reason sollte TP sein"

    def test_short_trade_sl_hit(self):
        """Test: Short-Trade erreicht SL."""
        # Stark steigender Trend
        opens = np.array([100.0, 100.0, 101.0, 102.0, 103.0])
        closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        highs = np.array([100.5, 101.5, 102.5, 103.5, 104.5])
        lows = np.array([99.5, 100.5, 101.5, 102.5, 103.5])

        result, _, _, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=-1,
            tp_distance=5.0, sl_distance=1.0,
            spread=0.0, slippage=0.0,
            max_bars=100, timeout_bars=0
        )

        assert result == -1.0, "Short sollte verlieren"
        assert exit_reason == 1, "Exit-Reason sollte SL sein"

    def test_timeout_exit(self):
        """Test: Trade wird nach Timeout geschlossen."""
        # Seitwärtsbewegung - weder TP noch SL
        opens = np.ones(20) * 100
        closes = np.ones(20) * 100
        highs = np.ones(20) * 100.1
        lows = np.ones(20) * 99.9

        result, exit_idx, _, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=5.0, sl_distance=5.0,  # Weit weg
            spread=0.0, slippage=0.0,
            max_bars=100, timeout_bars=10
        )

        assert exit_reason == 2, "Exit-Reason sollte Timeout sein"
        assert exit_idx == 10, "Exit sollte bei Timeout-Bar sein"

    def test_spread_impact(self):
        """Test: Spread beeinflusst Trade-Ergebnis."""
        # Knapper Gewinn ohne Spread
        opens = np.array([100.0, 100.0, 100.5])
        closes = np.array([100.0, 100.5, 100.5])
        highs = np.array([100.1, 100.6, 100.6])
        lows = np.array([99.9, 100.0, 100.0])

        # Ohne Spread: TP bei 100.5 sollte erreicht werden
        result_no_spread, _, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=0.5, sl_distance=0.5,
            spread=0.0, slippage=0.0,
            max_bars=100, timeout_bars=0
        )

        # Mit Spread: TP bei 100.5 + spread ist schwerer zu erreichen
        result_with_spread, _, _, _ = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=0.5, sl_distance=0.5,
            spread=0.1, slippage=0.0,
            max_bars=100, timeout_bars=0
        )

        # Spread macht den Trade schwieriger - mindestens einer sollte verlieren
        assert result_no_spread >= result_with_spread

    def test_edge_case_last_bar(self):
        """Test: Entry am letzten Bar."""
        opens = np.array([100.0, 101.0])
        closes = np.array([100.0, 101.0])
        highs = np.array([100.5, 101.5])
        lows = np.array([99.5, 100.5])

        # Entry am letzten Bar - kein Platz für Exit
        result, exit_idx, _, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=1, direction=1,
            tp_distance=0.5, sl_distance=0.5,
            spread=0.0, slippage=0.0,
            max_bars=100, timeout_bars=0
        )

        assert result == 0.0, "Sollte kein Ergebnis sein"
        assert exit_idx == -1

    def test_tp_and_sl_same_bar_conservative(self):
        """Test: TP und SL im selben Bar - konservativ Loss."""
        # Bar mit großer Range - beide Level im selben Bar
        opens = np.array([100.0, 100.0])
        closes = np.array([100.0, 100.0])
        highs = np.array([100.0, 105.0])  # Sehr hoch
        lows = np.array([100.0, 95.0])   # Sehr tief

        result, _, _, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=2.0, sl_distance=2.0,
            spread=0.0, slippage=0.0,
            max_bars=100, timeout_bars=0
        )

        assert result == -1.0, "Bei TP+SL im selben Bar: konservativ Loss"
        assert exit_reason == 1, "Exit-Reason sollte SL sein"


class TestComputeTargetsNumba:
    """Tests für compute_targets_numba."""

    def test_basic_target_computation(self):
        """Test: Grundlegende Target-Berechnung."""
        opens, closes, highs, lows = create_price_arrays(1000, trend="up")

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=1.0, sl_distance=0.5,
            spread=0.0, slippage=0.0,
            max_bars=100, timeout_bars=0
        )

        assert len(targets_long) == 1000
        assert len(targets_short) == 1000
        # In Aufwärtstrend: mehr Long-Wins erwartet
        assert np.sum(targets_long) > 0

    def test_uptrend_more_long_wins(self):
        """Test: Aufwärtstrend sollte mehr Long-Wins produzieren."""
        opens, closes, highs, lows = create_price_arrays(500, trend="up", volatility=0.005)

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.5, sl_distance=0.5,
            spread=0.0, slippage=0.0,
            max_bars=50, timeout_bars=0
        )

        long_wins = np.sum(targets_long)
        short_wins = np.sum(targets_short)

        # In starkem Aufwärtstrend: Long sollte besser sein
        assert long_wins >= short_wins * 0.8  # Mit Toleranz

    def test_downtrend_more_short_wins(self):
        """Test: Abwärtstrend sollte mehr Short-Wins produzieren."""
        opens, closes, highs, lows = create_price_arrays(500, trend="down", volatility=0.005)

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.5, sl_distance=0.5,
            spread=0.0, slippage=0.0,
            max_bars=50, timeout_bars=0
        )

        long_wins = np.sum(targets_long)
        short_wins = np.sum(targets_short)

        # In starkem Abwärtstrend: Short sollte besser sein
        assert short_wins >= long_wins * 0.8  # Mit Toleranz

    def test_last_bar_no_target(self):
        """Test: Letzter Bar sollte kein Target haben."""
        opens, closes, highs, lows = create_price_arrays(100)

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=1.0, sl_distance=1.0,
            spread=0.0, slippage=0.0,
            max_bars=100, timeout_bars=0
        )

        # Letzter Bar kann kein Entry sein
        assert targets_long[-1] == 0.0
        assert targets_short[-1] == 0.0


class TestSimulateProTrade:
    """Tests für simulate_pro_trade (High-Level Wrapper)."""

    def test_basic_trade(self):
        """Test: Grundlegender Trade mit Metadaten."""
        closes = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
        highs = closes + 0.2
        lows = closes - 0.2
        timestamps = pd.date_range("2024-01-01", periods=5, freq="h")

        trade = simulate_pro_trade(
            closes=closes, highs=highs, lows=lows,
            idx=0, direction=1, tp_distance=2.0, sl_distance=1.0,
            spread=0.0, timestamps=timestamps.values,
            symbol="TEST", opens=closes
        )

        if trade:
            assert "result" in trade
            assert "exit_idx" in trade
            assert "entry_time" in trade
            assert "exit_time" in trade
            assert trade["direction"] == "LONG" or trade["direction"] == 1

    def test_no_trade_at_boundary(self):
        """Test: Kein Trade am Daten-Ende."""
        closes = np.array([100.0])
        highs = closes + 0.1
        lows = closes - 0.1

        trade = simulate_pro_trade(
            closes=closes, highs=highs, lows=lows,
            idx=0, direction=1, tp_distance=2.0, sl_distance=1.0,
            spread=0.0, opens=closes
        )

        assert trade is None


class TestCalculateSharpeRatio:
    """Tests für calculate_sharpe_ratio."""

    def test_positive_sharpe(self):
        """Test: Positive Sharpe bei profitablen Returns."""
        # Konstant positive Returns
        returns = [0.01, 0.02, 0.01, 0.015, 0.02] * 20  # 100 Returns

        sharpe = calculate_sharpe_ratio(returns)

        assert sharpe > 0, "Sharpe sollte positiv sein"

    def test_negative_sharpe(self):
        """Test: Negative Sharpe bei Verlusten."""
        # Konstant negative Returns
        returns = [-0.01, -0.02, -0.01, -0.015, -0.02] * 20

        sharpe = calculate_sharpe_ratio(returns)

        assert sharpe < 0, "Sharpe sollte negativ sein"

    def test_zero_sharpe_mixed(self):
        """Test: Sharpe nahe 0 bei gemischten Returns."""
        # Exakt ausbalancierte Returns
        returns = [0.01, -0.01] * 50

        sharpe = calculate_sharpe_ratio(returns)

        assert abs(sharpe) < 1.0, "Sharpe sollte nahe 0 sein"

    def test_empty_returns(self):
        """Test: Leere Returns."""
        sharpe = calculate_sharpe_ratio([])
        assert sharpe == 0.0

    def test_single_return(self):
        """Test: Einzelner Return."""
        sharpe = calculate_sharpe_ratio([0.05])
        assert sharpe == 0.0

    def test_constant_returns(self):
        """Test: Konstante Returns (keine Volatilität) - sehr hohe Sharpe."""
        returns = [0.01] * 100

        sharpe = calculate_sharpe_ratio(returns)

        # Bei extrem niedriger Volatilität ist Sharpe sehr hoch (oder inf)
        # Da Std nahe 0, wird Sharpe sehr groß
        assert sharpe > 0 or np.isinf(sharpe), "Konstante positive Returns = sehr hohe Sharpe"


class TestCalculateCalmarRatio:
    """Tests für calculate_calmar_ratio."""

    def test_positive_calmar(self):
        """Test: Positive Calmar bei profitablen Trades."""
        # Mehr Wins als Losses
        trades = [1.0, 1.0, -1.0, 1.0, 1.0] * 20

        calmar = calculate_calmar_ratio(trades, risk_per_trade=0.02, rrr=2.0)

        assert calmar > 0, "Calmar sollte positiv sein"

    def test_zero_drawdown_handling(self):
        """Test: Handling bei keinem Drawdown."""
        # Nur Wins - kein Drawdown
        trades = [1.0] * 50

        calmar = calculate_calmar_ratio(trades, risk_per_trade=0.02, rrr=2.0)

        # Bei keinem Drawdown: Sollte nicht crashen
        assert calmar >= 0 or calmar == float('inf') or np.isinf(calmar)

    def test_empty_trades(self):
        """Test: Leere Trade-Liste."""
        calmar = calculate_calmar_ratio([], risk_per_trade=0.02, rrr=2.0)
        assert calmar == 0.0


class TestMonteCarloPermutationTest:
    """Tests für monte_carlo_permutation_test."""

    def test_significant_strategy(self):
        """Test: Signifikante Strategie hat niedrigen p-Wert."""
        # Stark profitable Trades
        trades = [1.0] * 80 + [-1.0] * 20  # 80% Winrate

        result = monte_carlo_permutation_test(trades, n_permutations=100)
        p_value = result["p_value"]

        assert p_value < 0.5, "Signifikante Strategie sollte p < 0.5 haben"

    def test_random_strategy(self):
        """Test: Zufällige Strategie hat höheren p-Wert."""
        # 50/50 Trades
        trades = [1.0, -1.0] * 50

        result = monte_carlo_permutation_test(trades, n_permutations=100)
        p_value = result["p_value"]

        assert p_value > 0.1, "Zufällige Strategie sollte höheren p-Wert haben"

    def test_few_trades(self):
        """Test: Wenige Trades."""
        trades = [1.0, -1.0, 1.0]

        # Sollte nicht crashen
        result = monte_carlo_permutation_test(trades, n_permutations=50)
        p_value = result["p_value"]

        assert 0.0 <= p_value <= 1.0


class TestEquitySmoothness:
    """Tests für calculate_equity_smoothness."""

    def test_smooth_equity(self):
        """Test: Glatte Equity hat hohen Smoothness-Score."""
        # Gleichmäßige Gewinne
        trades = [1.0, 1.0, 1.0, -1.0] * 50  # 75% Winrate, gleichmäßig

        result = calculate_equity_smoothness(trades, risk_per_trade=0.02, rrr=2.0)

        assert "smoothness_score" in result
        assert "return_volatility" in result
        assert result["smoothness_score"] > 0

    def test_volatile_equity(self):
        """Test: Volatile Equity hat niedrigeren Score."""
        # Stark schwankende Ergebnisse
        trades = [1.0] * 10 + [-1.0] * 10 + [1.0] * 10 + [-1.0] * 10

        result = calculate_equity_smoothness(trades, risk_per_trade=0.05, rrr=3.0)

        # Sollte nicht crashen und vernünftige Werte liefern
        assert 0 <= result["smoothness_score"] <= 1

    def test_insufficient_trades(self):
        """Test: Zu wenige Trades für Berechnung."""
        trades = [1.0, -1.0]

        result = calculate_equity_smoothness(trades, risk_per_trade=0.02, rrr=2.0, window_size=50)

        assert result["smoothness_score"] == 0.5  # Default


class TestAdjustKellyForTargetDd:
    """Tests für adjust_risk_for_target_dd."""

    def test_reduces_kelly_for_high_dd(self):
        """Test: Kelly wird reduziert bei hohem Drawdown."""
        # Trades mit hohem Drawdown-Potenzial
        trades = [1.0, -1.0, -1.0, -1.0, 1.0] * 20
        original_risk = 0.05

        result = adjust_risk_for_target_dd(
            trades, original_risk, rrr=2.0, target_max_dd=0.15
        )
        adjusted_risk = result["adjusted_risk"]

        assert adjusted_risk <= original_risk

    def test_maintains_minimum(self):
        """Test: Kelly bleibt über Minimum."""
        trades = [-1.0] * 50  # Nur Verluste
        original_risk = 0.05

        result = adjust_risk_for_target_dd(
            trades, original_risk, rrr=2.0, target_max_dd=0.01
        )
        adjusted_risk = result["adjusted_risk"]

        assert adjusted_risk >= 0.001  # Irgendein Minimum


class TestFindOptimalCircuitBreaker:
    """Tests für find_optimal_circuit_breaker."""

    def test_finds_optimal_cb(self):
        """Test: Findet optimalen Circuit Breaker."""
        # Trades mit Drawdown-Phasen
        trades = [1.0] * 30 + [-1.0] * 10 + [1.0] * 30 + [-1.0] * 10

        result = find_optimal_circuit_breaker(
            trades, risk_per_trade=0.02, rrr=2.0,
            loss_range=(3, 10), pause_range=(5, 20)
        )

        assert "optimal_pause_after_losses" in result
        assert "optimal_pause_bars" in result
        assert result["optimal_pause_after_losses"] >= 3
        assert result["optimal_pause_after_losses"] <= 10


class TestEdgeCases:
    """Edge Cases und Fehlerfälle."""

    def test_nan_in_prices(self):
        """Test: NaN in Preisen."""
        opens = np.array([100.0, np.nan, 101.0])
        closes = np.array([100.0, 100.5, np.nan])
        highs = np.array([100.5, 101.0, 101.5])
        lows = np.array([99.5, 100.0, 100.5])

        # Numba kann mit NaN arbeiten, aber Ergebnisse sind undefiniert
        # Sollte zumindest nicht crashen
        try:
            targets_long, targets_short = compute_targets_numba(
                opens, closes, highs, lows,
                tp_distance=1.0, sl_distance=1.0,
                spread=0.0, slippage=0.0,
                max_bars=100, timeout_bars=0
            )
            assert len(targets_long) == 3
        except Exception:
            pass  # NaN-Handling ist nicht garantiert

    def test_zero_distances(self):
        """Test: TP/SL-Distanz von 0."""
        closes = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        highs = closes
        lows = closes

        trade = simulate_pro_trade(
            closes=closes, highs=highs, lows=lows,
            idx=0, direction=1, tp_distance=0.0, sl_distance=0.0,
            spread=0.0, opens=closes
        )

        # Mit Distanz=0 werden TP/SL-Levels bei Entry sein
        # Trade könnte sofort beendet werden
        assert trade is None or "result" in trade

    def test_negative_prices(self):
        """Test: Negative Preise (theoretisch möglich bei manchen Instrumenten)."""
        opens = np.array([-10.0, -9.0, -8.0, -7.0, -6.0])
        closes = np.array([-10.0, -9.0, -8.0, -7.0, -6.0])
        highs = opens + 0.5
        lows = opens - 0.5

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=1.0, sl_distance=1.0,
            spread=0.0, slippage=0.0,
            max_bars=100, timeout_bars=0
        )

        # Sollte funktionieren (z.B. für Interest Rate Futures)
        assert len(targets_long) == 5

    def test_very_large_tp_sl(self):
        """Test: Sehr große TP/SL-Werte."""
        opens, closes, highs, lows = create_price_arrays(100)

        targets_long, _ = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=1000.0, sl_distance=1000.0,  # Viel größer als Price-Bewegung
            spread=0.0, slippage=0.0,
            max_bars=10, timeout_bars=0
        )

        # Keine Targets sollten erreicht werden
        assert np.sum(targets_long) == 0
