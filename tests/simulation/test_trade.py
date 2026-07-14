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

from fwbg.simulation.numba_core import (
    _simulate_trade_numba,
    _simulate_trade_session_numba,
)
from fwbg.simulation import compute_targets_numba
from fwbg.simulation.trade import (
    simulate_pro_trade,
    compute_session_mask,
    calculate_sharpe_ratio,
    calculate_calmar_ratio,
    monte_carlo_permutation_test,
    calculate_equity_smoothness,
    adjust_risk_for_target_dd,
    find_optimal_circuit_breaker,
    pnl_to_returns,
    attach_regime_labels,
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

        n = len(opens)
        targets_long, targets_short, _, _ = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distances=np.full(n, 1.0), sl_distances=np.full(n, 0.5),
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

        n = len(opens)
        targets_long, targets_short, _, _ = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distances=np.full(n, 0.5), sl_distances=np.full(n, 0.5),
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

        n = len(opens)
        targets_long, targets_short, _, _ = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distances=np.full(n, 0.5), sl_distances=np.full(n, 0.5),
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

        n = len(opens)
        targets_long, targets_short, _, _ = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distances=np.full(n, 1.0), sl_distances=np.full(n, 1.0),
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

    def test_timeout_inside_loop_matches_numba(self):
        """Timeout must fire INSIDE the loop, matching _simulate_trade_numba.

        Scenario: Long trade in slight uptrend. TP/SL are wide enough to not
        trigger within timeout_bars (8), but SL would hit at bar 15 if the
        trade kept running. The correct behavior is timeout at bar 8, NOT
        SL at bar 15.
        """
        n = 30
        # Price drifts up slightly for bars 0-9, then drops hard for bars 10-20
        opens = np.zeros(n)
        closes = np.zeros(n)
        highs = np.zeros(n)
        lows = np.zeros(n)

        for i in range(n):
            if i <= 9:
                # Slight uptrend: +0.5 per bar
                opens[i] = 100.0 + i * 0.5
                closes[i] = opens[i] + 0.3
            else:
                # Sharp drop: -2.0 per bar after bar 9
                opens[i] = 105.0 - (i - 9) * 2.0
                closes[i] = opens[i] - 1.5
            highs[i] = max(opens[i], closes[i]) + 0.2
            lows[i] = min(opens[i], closes[i]) - 0.2

        # Entry at bar 1 (idx=0), Long
        # TP = entry + 50 (never reached)
        # SL = entry - 10 (hit around bar 14-15 when price drops below ~90)
        # timeout_bars = 8 → should exit at bar 8 close (~104.3, profitable)
        trade = simulate_pro_trade(
            closes=closes, highs=highs, lows=lows,
            idx=0, direction=1, tp_distance=50.0, sl_distance=10.0,
            spread=0.0, opens=opens, max_bars=100, timeout_bars=8,
        )

        assert trade is not None, "Trade should complete via timeout"
        assert trade.get("exit_reason") == "timeout", (
            f"Expected timeout exit, got: {trade}"
        )
        # entry_idx=1, timeout_idx = 1+8-1 = 8
        assert trade["exit_idx"] == 8, (
            f"Expected exit at bar 8 (timeout), got bar {trade['exit_idx']}"
        )

        # Cross-check: _simulate_trade_numba must produce the same result
        result_numba, exit_idx_numba, _, exit_reason_numba = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=50.0, sl_distance=10.0,
            spread=0.0, slippage=0.0,
            max_bars=100, timeout_bars=8,
        )
        assert exit_reason_numba == 2, "Numba should timeout"
        assert exit_idx_numba == 8, "Numba should exit at bar 8"
        # Both must agree on the result sign
        assert (trade["result"] > 0) == (result_numba > 0), (
            f"Result mismatch: pro_trade={trade['result']}, numba={result_numba}"
        )

    def test_timeout_prevents_later_sl(self):
        """SL that would fire after timeout must NOT trigger."""
        n = 20
        # Flat for 5 bars, then crash
        opens = np.full(n, 100.0)
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.2)
        lows = np.full(n, 99.8)

        # Bar 7+: price crashes to SL
        for i in range(7, n):
            opens[i] = 100.0 - (i - 6) * 5.0
            closes[i] = opens[i] - 3.0
            highs[i] = opens[i] + 0.5
            lows[i] = closes[i] - 0.5

        # timeout_bars=5 → exit at bar 5 (entry at bar 1)
        # SL at entry - 8 = 92.0, would be hit around bar 8-9
        trade = simulate_pro_trade(
            closes=closes, highs=highs, lows=lows,
            idx=0, direction=1, tp_distance=50.0, sl_distance=8.0,
            spread=0.0, opens=opens, max_bars=100, timeout_bars=5,
        )

        assert trade is not None
        assert trade.get("exit_reason") == "timeout"
        assert trade["exit_idx"] == 5  # entry_idx=1, 1+5-1=5

    def test_timeout_none_no_timeout(self):
        """timeout_bars=None should disable timeout (no change from original)."""
        n = 20
        opens = np.full(n, 100.0)
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.1)
        lows = np.full(n, 99.9)

        trade = simulate_pro_trade(
            closes=closes, highs=highs, lows=lows,
            idx=0, direction=1, tp_distance=50.0, sl_distance=50.0,
            spread=0.0, opens=opens, max_bars=15, timeout_bars=None,
        )

        # No TP/SL hit, no timeout → None
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
    """Smoke-Test für monte_carlo_permutation_test (Basis-Interface).

    Ausführliche Tests mit PnL-Verteilungen, Breakeven-Szenarien und
    asymmetrischen Profilen sind in tests/simulation/test_monte_carlo.py.
    """

    def test_returns_valid_result_structure(self):
        """Funktion gibt dict mit gültigem p-Wert zurück und stürzt nicht ab."""
        pnls = [200.0] * 40 + [-80.0] * 60
        result = monte_carlo_permutation_test(pnls, n_permutations=100, random_seed=42)

        assert "p_value" in result
        assert "is_significant" in result
        assert 0.0 <= result["p_value"] <= 1.0


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
            n = len(opens)
            targets_long, targets_short, _, _ = compute_targets_numba(
                opens, closes, highs, lows,
                tp_distances=np.full(n, 1.0), sl_distances=np.full(n, 1.0),
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

        n = len(opens)
        targets_long, targets_short, _, _ = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distances=np.full(n, 1.0), sl_distances=np.full(n, 1.0),
            spread=0.0, slippage=0.0,
            max_bars=100, timeout_bars=0
        )

        # Sollte funktionieren (z.B. für Interest Rate Futures)
        assert len(targets_long) == 5

    def test_very_large_tp_sl(self):
        """Test: Sehr große TP/SL-Werte."""
        opens, closes, highs, lows = create_price_arrays(100)

        n = len(opens)
        targets_long, _, _, _ = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distances=np.full(n, 1000.0), sl_distances=np.full(n, 1000.0),  # Viel größer als Price-Bewegung
            spread=0.0, slippage=0.0,
            max_bars=10, timeout_bars=0
        )

        # Keine Targets sollten erreicht werden
        assert np.sum(targets_long) == 0


class TestPnlToReturns:
    """Tests für pnl_to_returns: skaliert pnl_raw auf Kelly-Returns."""

    def test_avg_loss_return_equals_minus_fk(self):
        """Durchschnittlicher Loss-Return muss genau -fk sein."""
        pnl = [16.0, 16.0, -24.0, -24.0, 16.0]
        returns = pnl_to_returns(pnl, fk=0.02)
        losses = [r for r in returns if r < 0]
        avg_loss = sum(losses) / len(losses)
        assert abs(avg_loss + 0.02) < 1e-12

    def test_preserves_sign(self):
        """Positives pnl → positiver Return, negatives pnl → negativer Return."""
        pnl = [10.0, -5.0, 20.0]
        returns = pnl_to_returns(pnl, fk=0.01)
        assert returns[0] > 0
        assert returns[1] < 0
        assert returns[2] > 0

    def test_proportional_to_pnl(self):
        """Doppelter Win-PnL → doppelter Return."""
        pnl = [20.0, -10.0, 10.0, -10.0]
        returns = pnl_to_returns(pnl, fk=0.02)
        assert abs(returns[0] / returns[2] - 2.0) < 1e-12

    def test_all_wins_fallback(self):
        """Keine Losses → kein Crash, alle Returns positiv."""
        pnl = [10.0, 20.0, 15.0]
        returns = pnl_to_returns(pnl, fk=0.02)
        assert len(returns) == 3
        assert all(r > 0 for r in returns)

    def test_empty_returns_empty(self):
        assert pnl_to_returns([], fk=0.02) == []


class TestComputeSessionMask:
    """Tests for compute_session_mask helper."""

    def test_non_crossing_session(self):
        """Session 8-17: bars 08:00-16:59 are in-session."""
        timestamps = pd.date_range("2025-01-02 06:00", periods=24, freq="1h")
        mask = compute_session_mask(timestamps, 8, 17)
        # 06:00, 07:00 → out of session
        assert not mask[0]
        assert not mask[1]
        # 08:00-16:00 → in session
        for i in range(2, 11):  # hours 8-16
            assert mask[i], f"Hour {6+i} should be in session"
        # 17:00 → out of session
        assert not mask[11]

    def test_midnight_crossing_session(self):
        """Session 23-6: midnight-crossing (e.g. ASX200)."""
        timestamps = pd.date_range("2025-01-02 20:00", periods=14, freq="1h")
        mask = compute_session_mask(timestamps, 23, 6)
        # 20:00, 21:00, 22:00 → out of session
        assert not mask[0]
        assert not mask[1]
        assert not mask[2]
        # 23:00-05:00 → in session
        for i in range(3, 10):  # 23, 00, 01, 02, 03, 04, 05
            assert mask[i], f"Index {i} should be in session"
        # 06:00 → out of session
        assert not mask[10]

    def test_all_in_session(self):
        """All bars within session → all True."""
        timestamps = pd.date_range("2025-01-02 09:00", periods=8, freq="1h")
        mask = compute_session_mask(timestamps, 8, 17)
        assert mask.all()

    def test_all_out_of_session(self):
        """All bars outside session → all False."""
        timestamps = pd.date_range("2025-01-02 18:00", periods=4, freq="1h")
        mask = compute_session_mask(timestamps, 8, 17)
        assert not mask.any()

    def test_flat_bars_excluded(self):
        """Flat bars (O==H==L==C, e.g. weekends/holidays) excluded from mask."""
        timestamps = pd.date_range("2025-01-02 00:00", periods=8, freq="1h")
        opens  = np.array([100, 100, 101, 102, 102, 102, 103, 104], dtype=np.float64)
        highs  = np.array([101, 100, 102, 103, 102, 103, 104, 105], dtype=np.float64)
        lows   = np.array([ 99, 100, 100, 101, 102, 101, 102, 103], dtype=np.float64)
        closes = np.array([100, 100, 101, 102, 102, 102, 103, 104], dtype=np.float64)
        # Bar 1 (01:00) and bar 4 (04:00) are flat (O==H==L==C)

        mask_no_ohlc = compute_session_mask(timestamps, 0, 8)
        assert mask_no_ohlc[1]  # without ohlc, flat bar passes
        assert mask_no_ohlc[4]

        mask_with_ohlc = compute_session_mask(
            timestamps, 0, 8, ohlc=(opens, highs, lows, closes),
        )
        assert not mask_with_ohlc[1], "Flat bar at 01:00 should be excluded"
        assert not mask_with_ohlc[4], "Flat bar at 04:00 should be excluded"
        assert mask_with_ohlc[0]  # non-flat, in session
        assert mask_with_ohlc[2]  # non-flat, in session


class TestSessionAwareSimulation:
    """Tests for session-aware trade simulation (exits only during session)."""

    def _make_data(self, n=48, start="2025-01-02 20:00", freq="1h"):
        """Create synthetic OHLC data spanning session and off-session hours."""
        timestamps = pd.date_range(start, periods=n, freq=freq)
        # Steadily rising prices
        base = 100.0 + np.arange(n, dtype=np.float64) * 0.5
        opens = base.copy()
        closes = base + 0.3
        highs = base + 0.8
        lows = base - 0.2
        return timestamps, opens, closes, highs, lows

    def test_tp_only_during_session(self):
        """TP should only trigger on in-session bars, skipping off-session."""
        # Session 23-6 UTC. Data starts 20:00, entry at bar 0 (idx=-1 not possible
        # with the numba fn, so we use idx=0 → entry at bar 1 = 21:00 = off-session).
        # TP is set very tight so it would trigger immediately on next bar.
        n = 24
        timestamps, opens, closes, highs, lows = self._make_data(n, "2025-01-02 20:00")
        in_session = compute_session_mask(timestamps, 23, 6)

        # Set TP very small (0.01) so any price movement triggers it
        # Entry at bar 1 (21:00, off-session). TP should NOT trigger at 21:00 or 22:00.
        # First in-session bar is 23:00 (index 3).
        result, exit_idx, _, exit_reason = _simulate_trade_session_numba(
            opens, closes, highs, lows, 0, 1,
            0.01, 100.0, 0.0, 0.0, n, 0, in_session,
        )
        assert result == 1.0
        # Exit must be on an in-session bar (hour 23-05)
        exit_hour = timestamps[exit_idx].hour
        assert exit_hour >= 23 or exit_hour < 6, f"Exit at hour {exit_hour} is off-session"

    def test_sl_only_during_session(self):
        """SL should only trigger on in-session bars."""
        n = 24
        timestamps = pd.date_range("2025-01-02 20:00", periods=n, freq="1h")
        # Prices that drop immediately — SL would trigger on first bar after entry
        base = 100.0 - np.arange(n, dtype=np.float64) * 2.0
        opens = base.copy()
        closes = base - 1.0
        highs = base + 0.1
        lows = base - 2.0
        in_session = compute_session_mask(timestamps, 23, 6)

        # SL tight (0.01), entry at bar 1 (21:00 off-session)
        result, exit_idx, _, exit_reason = _simulate_trade_session_numba(
            opens, closes, highs, lows, 0, 1,
            100.0, 0.01, 0.0, 0.0, n, 0, in_session,
        )
        assert result == -1.0
        exit_hour = timestamps[exit_idx].hour
        assert exit_hour >= 23 or exit_hour < 6, f"SL exit at hour {exit_hour} is off-session"

    def test_timeout_counts_session_bars_only(self):
        """Timeout should count only in-session bars, not total bars."""
        n = 48
        timestamps, opens, closes, highs, lows = self._make_data(n, "2025-01-02 20:00")
        in_session = compute_session_mask(timestamps, 23, 6)

        # timeout_bars=3 → should close after 3 SESSION bars
        # Entry at bar 1 (21:00 off-session).
        # Session bars: 23:00(3), 00:00(4), 01:00(5) → 3rd session bar = 01:00
        result, exit_idx, _, exit_reason = _simulate_trade_session_numba(
            opens, closes, highs, lows, 0, 1,
            100.0, 100.0, 0.0, 0.0, n, 3, in_session,
        )
        assert exit_reason == 2  # timeout
        assert exit_idx == 5  # 01:00 UTC (3rd session bar after entry)
        exit_hour = timestamps[exit_idx].hour
        assert exit_hour >= 23 or exit_hour < 6

    def test_trade_runs_through_off_session(self):
        """Trade should survive through off-session gap without forced closure."""
        # Session 8-17. Entry during session, price stays flat through night,
        # TP hit next morning.
        n = 48
        timestamps = pd.date_range("2025-01-02 15:00", periods=n, freq="1h")
        base = np.full(n, 100.0, dtype=np.float64)
        opens = base.copy()
        closes = base.copy()
        highs = base + 0.1
        lows = base - 0.1
        # TP spike next day at 09:00 (index 18)
        highs[18] = 120.0
        closes[18] = 115.0
        in_session = compute_session_mask(timestamps, 8, 17)

        # Entry at bar 1 (16:00, in session). TP=10.0 (won't hit until bar 18).
        # Off-session bars 17:00-07:00 should be skipped.
        result, exit_idx, _, exit_reason = _simulate_trade_session_numba(
            opens, closes, highs, lows, 0, 1,
            10.0, 100.0, 0.0, 0.0, n, 0, in_session,
        )
        assert result == 1.0
        assert exit_idx == 18  # 09:00 next day
        assert exit_reason == 0  # TP hit

    def test_no_session_filter_matches_original(self):
        """With all-True mask, session-aware function matches original."""
        n = 20
        np.random.seed(42)
        opens = 100.0 + np.random.randn(n).cumsum()
        closes = opens + np.random.randn(n) * 0.3
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(n)) * 0.5
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(n)) * 0.5
        all_true = np.ones(n, dtype=np.bool_)

        for direction in [1, -1]:
            r1, e1, p1, re1 = _simulate_trade_numba(
                opens, closes, highs, lows, 5, direction,
                2.0, 1.0, 0.01, 0.005, n, 10,
            )
            r2, e2, p2, re2 = _simulate_trade_session_numba(
                opens, closes, highs, lows, 5, direction,
                2.0, 1.0, 0.01, 0.005, n, 10, all_true,
            )
            assert r1 == r2
            assert e1 == e2
            assert re1 == re2

    def test_simulate_pro_trade_session_aware(self):
        """simulate_pro_trade respects in_session mask for exits."""
        n = 24
        timestamps = pd.date_range("2025-01-02 20:00", periods=n, freq="1h")
        base = 100.0 + np.arange(n, dtype=np.float64) * 0.5
        opens = base.copy()
        closes = base + 0.3
        highs = base + 0.8
        lows = base - 0.2
        in_session = compute_session_mask(timestamps, 23, 6)

        # timeout_bars=2 → 2 session bars. Entry at bar 1 (21:00 off-session).
        # Session bars: 23:00(3), 00:00(4) → timeout at 00:00
        trade = simulate_pro_trade(
            closes, highs, lows, 0, 1, 100.0, 100.0, 0.0,
            max_bars=n, timestamps=timestamps, opens=opens,
            timeout_bars=2, in_session=in_session,
        )
        assert trade is not None
        assert trade["exit_reason"] == "timeout"
        exit_hour = timestamps[trade["exit_idx"]].hour
        assert exit_hour >= 23 or exit_hour < 6, f"Exit at hour {exit_hour} is off-session"

    def test_exit_session_wider_than_signal_session(self):
        """Exit session (23-20) is wider than signal session (23-6).

        TP at hour 15 (outside signal session, inside exit session) should fire.
        """
        n = 48
        timestamps = pd.date_range("2025-01-02 20:00", periods=n, freq="1h")
        base = np.full(n, 100.0, dtype=np.float64)
        opens = base.copy()
        closes = base.copy()
        highs = base + 0.1
        lows = base - 0.1

        # TP spike at hour 15:00 UTC (index 19: 20:00 + 19h = 15:00 next day)
        highs[19] = 120.0
        closes[19] = 115.0

        # Exit session 23-20 (full CFD hours, only 20-23 dead zone)
        exit_mask = compute_session_mask(timestamps, 23, 20)

        # Entry at bar 1 (21:00, off-session for exits).
        # TP=10.0, won't hit until bar 19 (15:00, in exit session).
        result, exit_idx, _, exit_reason = _simulate_trade_session_numba(
            opens, closes, highs, lows, 0, 1,
            10.0, 100.0, 0.0, 0.0, n, 0, exit_mask,
        )
        assert result == 1.0, "TP should hit during exit session"
        assert exit_idx == 19
        assert exit_reason == 0  # TP

        # Same trade with narrow signal session (23-6) as exit mask —
        # TP at 15:00 is off-session, trade should NOT close there.
        signal_mask = compute_session_mask(timestamps, 23, 6)
        result2, exit_idx2, _, _ = _simulate_trade_session_numba(
            opens, closes, highs, lows, 0, 1,
            10.0, 100.0, 0.0, 0.0, n, 0, signal_mask,
        )
        # With narrow session, TP at 15:00 is skipped
        assert exit_idx2 != 19, "Narrow session should skip bar 19 (15:00 UTC)"


class TestAttachRegimeLabels:
    """Tests für attach_regime_labels (Plan 010 WP5)."""

    def _df_with_precomputed_columns(self, n=30):
        return pd.DataFrame({
            "O": np.linspace(1.0, 1.3, n),
            "H": np.linspace(1.01, 1.31, n),
            "L": np.linspace(0.99, 1.29, n),
            "C": np.linspace(1.0, 1.3, n),
            "_atr": np.linspace(0.001, 0.01, n),  # rising volatility
            "adx_14": np.linspace(10.0, 50.0, n),  # rising trend strength
        })

    def test_buckets_vol_and_trend_using_existing_columns(self):
        n = 30
        df = self._df_with_precomputed_columns(n)
        trades = [
            {"entry_idx": 0},
            {"entry_idx": n // 2},
            {"entry_idx": n - 1},
        ]
        attach_regime_labels(trades, df)

        assert trades[0]["vol_regime"] == "low"
        assert trades[0]["trend_regime"] == "ranging"
        assert trades[-1]["vol_regime"] == "high"
        assert trades[-1]["trend_regime"] == "strong_trend"

    def test_empty_trades_list_is_noop(self):
        df = self._df_with_precomputed_columns()
        attach_regime_labels([], df)  # must not raise

    def test_missing_or_out_of_range_entry_idx_is_skipped(self):
        df = self._df_with_precomputed_columns()
        trades = [{"no_entry_idx": True}, {"entry_idx": 9999}, {"entry_idx": -1}]
        attach_regime_labels(trades, df)
        for t in trades:
            assert "vol_regime" not in t
            assert "trend_regime" not in t

    def test_falls_back_to_computed_atr_and_adx_when_columns_missing(self):
        n = 60
        rng = np.random.default_rng(42)
        base = 1.0 + np.cumsum(rng.normal(0, 0.001, n))
        df = pd.DataFrame({
            "O": base,
            "H": base + 0.002,
            "L": base - 0.002,
            "C": base,
        })
        trades = [{"entry_idx": 30}]
        attach_regime_labels(trades, df)
        assert trades[0]["vol_regime"] in ("low", "medium", "high")
        assert trades[0]["trend_regime"] in ("ranging", "trending", "strong_trend")
