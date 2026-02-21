"""
Tests für equity.py - Equity-Simulation und Portfolio-Filter.

Testet:
- simulate_equity: Equity-Kurven-Simulation mit Kelly-Sizing
- filter_correlated_assets: Korrelationsfilter für Portfolio-Diversifikation
"""
import pytest
import numpy as np

from fwbg.simulation.equity import simulate_equity, simulate_equity_from_pnl, filter_correlated_assets


class TestSimulateEquity:
    """Tests für simulate_equity."""

    def test_all_wins(self):
        """Test: Alle Trades gewinnen."""
        trades = [1.0] * 10
        result = simulate_equity(trades, risk_per_trade=0.02, rrr=2.0, start_equity=100.0)

        assert result["final_equity"] > 100.0, "Equity sollte gestiegen sein"
        assert result["max_drawdown"] == 0.0, "Kein Drawdown bei nur Gewinnen"
        assert len(result["equity_curve"]) == 11  # Start + 10 Trades
        assert all(e > 0 for e in result["equity_curve"])

    def test_all_losses(self):
        """Test: Alle Trades verlieren."""
        trades = [-1.0] * 10
        result = simulate_equity(trades, risk_per_trade=0.02, rrr=2.0, start_equity=100.0)

        assert result["final_equity"] < 100.0, "Equity sollte gefallen sein"
        assert result["max_drawdown"] > 0.0, "Drawdown sollte existieren"
        assert len(result["equity_curve"]) == 11

    def test_mixed_trades(self):
        """Test: Gemischte Gewinne und Verluste."""
        # 60% Winrate mit RRR 2.0 sollte profitabel sein
        trades = [1.0] * 60 + [-1.0] * 40
        result = simulate_equity(trades, risk_per_trade=0.02, rrr=2.0, start_equity=100.0)

        # Erwarteter Gewinn pro Trade: 0.6 * 2.0 - 0.4 * 1.0 = 0.8 (in R)
        assert result["final_equity"] > 100.0, "60% Winrate mit RRR 2.0 sollte profitabel sein"

    def test_bankruptcy(self):
        """Test: Bankrott bei 100% Verlust."""
        # Mit 50% Kelly-Risk und genug Verlusten sollte nahe 0 kommen
        trades = [-1.0] * 100
        result = simulate_equity(trades, risk_per_trade=0.5, rrr=1.0, start_equity=100.0)

        # Wert konvergiert gegen 0, wird aber nie exakt 0 (geometrisches Schrumpfen)
        assert result["final_equity"] < 0.01, "Sollte nahe bankrott sein"
        assert result["max_drawdown"] > 0.99, "Drawdown sollte nahe 100% sein"

    def test_compound_cap(self):
        """Test: Compound-Cap begrenzt Positionsgröße."""
        # Viele Gewinne ohne Cap
        trades_many_wins = [1.0] * 100

        # Mit niedrigem Cap
        result_capped = simulate_equity(
            trades_many_wins, risk_per_trade=0.05, rrr=3.0,
            start_equity=100.0, compound_cap=200.0
        )

        # Ohne Cap (hoher Cap)
        result_uncapped = simulate_equity(
            trades_many_wins, risk_per_trade=0.05, rrr=3.0,
            start_equity=100.0, compound_cap=1e12
        )

        # Uncapped sollte mehr wachsen durch Compounding
        assert result_uncapped["final_equity"] > result_capped["final_equity"]

    def test_drawdown_calculation(self):
        """Test: Drawdown wird korrekt berechnet."""
        # Win, Win, Loss, Loss, Win
        trades = [1.0, 1.0, -1.0, -1.0, 1.0]
        result = simulate_equity(trades, risk_per_trade=0.10, rrr=2.0, start_equity=100.0)

        # Nach 2 Wins: Equity = 100 + 20 + 24.8 = 144.8
        # Nach Loss 1: Equity = 144.8 - 14.48 = 130.32
        # Nach Loss 2: Equity = 130.32 - 13.032 = 117.288
        # Drawdown = (144.8 - 117.288) / 144.8 = 0.19 = 19%

        assert result["max_drawdown"] > 0.1, "Sollte signifikanten Drawdown haben"
        assert len(result["drawdowns"]) == 6  # Start + 5 Trades

    def test_empty_trades(self):
        """Test: Keine Trades."""
        trades = []
        result = simulate_equity(trades, risk_per_trade=0.02, rrr=2.0, start_equity=100.0)

        assert result["final_equity"] == 100.0
        assert result["max_drawdown"] == 0.0
        assert len(result["equity_curve"]) == 1

    def test_single_win(self):
        """Test: Ein Gewinn-Trade."""
        trades = [1.0]
        result = simulate_equity(trades, risk_per_trade=0.10, rrr=2.0, start_equity=100.0)

        # Gewinn = 100 * 0.10 * 2.0 = 20
        expected_equity = 120.0
        assert abs(result["final_equity"] - expected_equity) < 0.01

    def test_single_loss(self):
        """Test: Ein Verlust-Trade."""
        trades = [-1.0]
        result = simulate_equity(trades, risk_per_trade=0.10, rrr=2.0, start_equity=100.0)

        # Verlust = 100 * 0.10 = 10
        expected_equity = 90.0
        assert abs(result["final_equity"] - expected_equity) < 0.01

    def test_drawdown_peaks_correctly(self):
        """Test: Peak wird korrekt aktualisiert."""
        # Win, Loss, Loss, Win, Win (neuer Peak)
        trades = [1.0, -1.0, -1.0, 1.0, 1.0]
        result = simulate_equity(trades, risk_per_trade=0.05, rrr=2.0, start_equity=100.0)

        # Am Ende sollte ein neuer Peak erreicht werden
        # Equity-Kurve sollte steigen
        assert result["equity_curve"][-1] > result["equity_curve"][0]

    def test_large_risk_per_trade(self):
        """Test: Hoher Kelly-Wert führt zu starken Schwankungen."""
        trades = [1.0, -1.0] * 20
        result = simulate_equity(trades, risk_per_trade=0.25, rrr=1.0, start_equity=100.0)

        # Mit 50/50 und RRR 1.0 sollte es etwa breakeven sein
        # Aber hoher Kelly führt zu Schwankungen
        assert result["max_drawdown"] > 0.0

    def test_zero_kelly_no_change(self):
        """Test: Kelly = 0 bedeutet keine Positionsgröße."""
        trades = [1.0] * 10 + [-1.0] * 10
        result = simulate_equity(trades, risk_per_trade=0.0, rrr=2.0, start_equity=100.0)

        # Keine Positionsgröße = keine Änderung
        assert result["final_equity"] == 100.0


class TestFilterCorrelatedAssets:
    """Tests für filter_correlated_assets."""

    def test_no_currencies(self):
        """Test: Assets ohne Währungs-Info werden akzeptiert."""
        results = [
            {"symbol": "EURUSD", "pnl": 100.0},
            {"symbol": "GBPUSD", "pnl": 80.0},
        ]

        filtered = filter_correlated_assets(results, threshold=0.75)

        assert len(filtered) == 2
        assert filtered[0]["symbol"] == "EURUSD"  # Höchster PnL zuerst

    def test_empty_results(self):
        """Test: Leere Ergebnisliste."""
        filtered = filter_correlated_assets([], threshold=0.75)
        assert filtered == []

    def test_currency_exposure_limit(self):
        """Test: Währungs-Exposure wird limitiert."""
        # Alle haben USD
        results = [
            {"symbol": "EURUSD", "pnl": 100.0, "currencies": ["EUR", "USD"]},
            {"symbol": "GBPUSD", "pnl": 90.0, "currencies": ["GBP", "USD"]},
            {"symbol": "AUDUSD", "pnl": 80.0, "currencies": ["AUD", "USD"]},
            {"symbol": "USDJPY", "pnl": 70.0, "currencies": ["USD", "JPY"]},
            {"symbol": "USDCHF", "pnl": 60.0, "currencies": ["USD", "CHF"]},
        ]

        # Mit threshold=0.75: max_allowed = int(1 / 0.25) = 4
        filtered = filter_correlated_assets(results, threshold=0.75)

        assert len(filtered) <= 4, "USD-Exposure sollte auf 4 begrenzt sein"

    def test_diverse_currencies_all_pass(self):
        """Test: Verschiedene Währungen passieren alle."""
        results = [
            {"symbol": "EURUSD", "pnl": 100.0, "currencies": ["EUR", "USD"]},
            {"symbol": "GBPJPY", "pnl": 90.0, "currencies": ["GBP", "JPY"]},
            {"symbol": "AUDNZD", "pnl": 80.0, "currencies": ["AUD", "NZD"]},
            {"symbol": "EURCHF", "pnl": 70.0, "currencies": ["EUR", "CHF"]},
        ]

        # Alle haben unterschiedliche Währungs-Paare
        filtered = filter_correlated_assets(results, threshold=0.75)

        # EURUSD und EURCHF teilen EUR, aber das ist ok mit threshold 0.75
        assert len(filtered) >= 3

    def test_sorted_by_pnl(self):
        """Test: Ergebnisse werden nach PnL sortiert verarbeitet."""
        results = [
            {"symbol": "LOW", "pnl": 10.0, "currencies": ["USD"]},
            {"symbol": "HIGH", "pnl": 100.0, "currencies": ["USD"]},
            {"symbol": "MED", "pnl": 50.0, "currencies": ["USD"]},
        ]

        filtered = filter_correlated_assets(results, threshold=0.75)

        # HIGH sollte zuerst kommen (höchster PnL)
        if len(filtered) > 0:
            assert filtered[0]["symbol"] == "HIGH"

    def test_threshold_edge_cases(self):
        """Test: Threshold-Grenzfälle."""
        results = [
            {"symbol": "A", "pnl": 100.0, "currencies": ["USD"]},
            {"symbol": "B", "pnl": 90.0, "currencies": ["USD"]},
        ]

        # threshold=0.99 -> max_allowed = int(1/0.01) = 100
        filtered_high = filter_correlated_assets(results, threshold=0.99)
        assert len(filtered_high) == 2

        # threshold=0.5 -> max_allowed = int(1/0.5) = 2
        filtered_low = filter_correlated_assets(results, threshold=0.5)
        assert len(filtered_low) == 2

    def test_single_result(self):
        """Test: Einzelnes Ergebnis."""
        results = [{"symbol": "EURUSD", "pnl": 100.0, "currencies": ["EUR", "USD"]}]

        filtered = filter_correlated_assets(results, threshold=0.75)

        assert len(filtered) == 1
        assert filtered[0]["symbol"] == "EURUSD"


class TestEquityEdgeCases:
    """Edge Cases und Grenzfälle."""

    def test_negative_start_equity(self):
        """Test: Negative Start-Equity wird durch Bankrott-Check auf 0 gesetzt."""
        trades = [1.0]
        result = simulate_equity(trades, risk_per_trade=0.10, rrr=2.0, start_equity=-100.0)

        # Negative Equity wird als Bankrott behandelt (equity <= 0)
        # Bei negativem Start wird equity sofort auf 0 gesetzt
        assert result["final_equity"] == 0 or result["final_equity"] <= 0

    def test_very_small_kelly(self):
        """Test: Sehr kleiner Kelly-Wert."""
        trades = [1.0] * 1000
        result = simulate_equity(
            trades, risk_per_trade=0.0001, rrr=2.0, start_equity=100.0
        )

        # Sollte minimal wachsen
        assert result["final_equity"] > 100.0
        assert result["final_equity"] < 200.0  # Nicht viel

    def test_very_high_rrr(self):
        """Test: Sehr hohes Risk-Reward-Ratio."""
        trades = [1.0] * 10
        result = simulate_equity(trades, risk_per_trade=0.01, rrr=100.0, start_equity=100.0)

        # Jeder Gewinn = 1% * 100 = 100% Zuwachs
        assert result["final_equity"] > 1000.0

    def test_alternating_trades(self):
        """Test: Abwechselnde Gewinne/Verluste."""
        trades = [1.0, -1.0] * 50
        result = simulate_equity(trades, risk_per_trade=0.02, rrr=2.0, start_equity=100.0)

        # Win gibt +4%, Loss gibt -2%
        # Netto pro Paar: ~+2%
        # Sollte profitabel sein
        assert result["final_equity"] > 100.0

    def test_drawdown_never_exceeds_one(self):
        """Test: Drawdown ist immer zwischen 0 und 1."""
        trades = np.random.choice([1.0, -1.0], size=1000).tolist()
        result = simulate_equity(trades, risk_per_trade=0.05, rrr=1.5, start_equity=100.0)

        assert 0.0 <= result["max_drawdown"] <= 1.0
        assert all(0 <= dd <= 100 for dd in result["drawdowns"])

    def test_equity_curve_length(self):
        """Test: Equity-Kurve hat korrekte Länge."""
        n_trades = 42
        trades = [1.0] * n_trades
        result = simulate_equity(trades, risk_per_trade=0.02, rrr=2.0, start_equity=100.0)

        assert len(result["equity_curve"]) == n_trades + 1
        assert len(result["drawdowns"]) == n_trades + 1

    def test_rrr_zero(self):
        """Test: RRR = 0 (kein Gewinn bei Wins)."""
        trades = [1.0] * 10 + [-1.0] * 5
        result = simulate_equity(trades, risk_per_trade=0.02, rrr=0.0, start_equity=100.0)

        # Wins bringen 0, Losses kosten 2%
        assert result["final_equity"] < 100.0


class TestSimulateEquityFromPnl:
    """Tests für simulate_equity_from_pnl: Equity-Simulation mit echten PnL-Werten."""

    def test_returns_correct_structure(self):
        """Funktion gibt dict mit equity_curve, final_equity, max_drawdown zurück."""
        pnl = [16.0, -24.0, 16.0]
        result = simulate_equity_from_pnl(pnl, fk=0.02)
        assert "equity_curve" in result
        assert "final_equity" in result
        assert "max_drawdown" in result
        assert "drawdowns" in result
        assert len(result["equity_curve"]) == 4  # start + 3 trades

    def test_losing_when_realized_rrr_too_low(self):
        """WR=50%, realized_RRR=0.67 → negatives EV → Verlust."""
        # wins=16, losses=-24 → avg_win/avg_loss = 0.67
        # EV = 0.5 * (0.02 * 16/24) - 0.5 * 0.02 = 0.5*0.01333 - 0.01 = -0.00333 < 0
        pnl = [16.0, -24.0] * 50
        result = simulate_equity_from_pnl(pnl, fk=0.02)
        assert result["final_equity"] < 100.0

    def test_profitable_when_realized_rrr_high(self):
        """WR=50%, realized_RRR=2.0 → positives EV → Gewinn."""
        # wins=48, losses=-24 → avg_win/avg_loss = 2.0
        # EV = 0.5 * (0.02 * 48/24) - 0.5 * 0.02 = 0.5*0.04 - 0.01 = 0.01 > 0
        pnl = [48.0, -24.0] * 50
        result = simulate_equity_from_pnl(pnl, fk=0.02)
        assert result["final_equity"] > 100.0

    def test_no_losses_fallback_no_crash(self):
        """Nur Gewinne → kein Absturz, Equity steigt."""
        pnl = [16.0, 20.0, 18.0]
        result = simulate_equity_from_pnl(pnl, fk=0.02)
        assert result["final_equity"] > 100.0
        assert result["max_drawdown"] == 0.0

    def test_empty_trades(self):
        """Keine Trades → Equity bleibt bei Start."""
        result = simulate_equity_from_pnl([], fk=0.02)
        assert result["final_equity"] == 100.0
        assert result["max_drawdown"] == 0.0
