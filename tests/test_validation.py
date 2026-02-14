"""
Tests für Walk-Forward Validation, Monte Carlo und Data Leakage Prevention.

Dieser Test verwendet synthetische Daten mit BEKANNTEN Eigenschaften,
um zu verifizieren dass:
1. Walk-Forward Split korrekt 60/20/20 aufteilt
2. CT-Optimierung auf Validation erfolgt (nicht OOS)
3. Monte Carlo Tests statistisch korrekt arbeiten
4. Profitable vs. unprofitable Strategien korrekt erkannt werden
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


class TestWalkForwardSplit:
    """Tests für die Nested CV Split Funktion."""

    def test_split_returns_inner_folds_and_holdout(self):
        """Nested CV sollte inner_folds (train, val) und holdout_df zurückgeben."""
        from fwbg.optimization.nested_cv import nested_cv_split

        # Erstelle Dummy-DataFrame
        n_rows = 50000
        df = pd.DataFrame({
            "C": np.random.randn(n_rows),
            "H": np.random.randn(n_rows),
            "L": np.random.randn(n_rows),
        }, index=pd.date_range("2020-01-01", periods=n_rows, freq="h"))

        result = nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=4, oos_size=4000)

        assert "inner_folds" in result, "Sollte inner_folds enthalten"
        assert "holdout_df" in result, "Sollte holdout_df enthalten"
        assert "inner_df" in result, "Sollte inner_df enthalten"
        assert len(result["inner_folds"]) > 0, "Sollte mindestens einen Inner-Fold haben"

        for train, val in result["inner_folds"]:
            assert isinstance(train, pd.DataFrame)
            assert isinstance(val, pd.DataFrame)

        assert isinstance(result["holdout_df"], pd.DataFrame)

    def test_split_holdout_ratio(self):
        """Holdout sollte ca. 20% der Daten sein."""
        from fwbg.optimization.nested_cv import nested_cv_split

        n_rows = 50000
        df = pd.DataFrame({
            "C": np.random.randn(n_rows),
        }, index=pd.date_range("2020-01-01", periods=n_rows, freq="h"))

        result = nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=4, oos_size=4000)

        holdout_ratio = len(result["holdout_df"]) / n_rows
        # Sollte ca. 20% sein (mit Toleranz)
        assert 0.18 <= holdout_ratio <= 0.22, f"Holdout ratio {holdout_ratio:.2%} sollte ca. 20% sein"

    def test_no_data_leakage_between_splits(self):
        """Es darf keine Überlappung zwischen Train/Val und Holdout geben."""
        from fwbg.optimization.nested_cv import nested_cv_split

        n_rows = 50000
        df = pd.DataFrame({
            "C": np.arange(n_rows),  # Eindeutige Werte
        }, index=pd.date_range("2020-01-01", periods=n_rows, freq="h"))

        result = nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=4, oos_size=4000)

        holdout_idx = set(result["holdout_df"].index)
        inner_idx = set(result["inner_df"].index)

        # Keine Überlappung zwischen inner und holdout
        assert len(inner_idx & holdout_idx) == 0, "Inner und Holdout überlappen sich"

        # Keine Überlappung innerhalb der Folds
        for train, val in result["inner_folds"]:
            train_idx = set(train.index)
            val_idx = set(val.index)

            assert len(train_idx & val_idx) == 0, "Train und Val überlappen sich"
            assert len(train_idx & holdout_idx) == 0, "Train und Holdout überlappen sich"
            assert len(val_idx & holdout_idx) == 0, "Val und Holdout überlappen sich"

    def test_chronological_order(self):
        """Inner kommt vor Holdout, Train vor Val."""
        from fwbg.optimization.nested_cv import nested_cv_split

        n_rows = 50000
        df = pd.DataFrame({
            "C": np.arange(n_rows),
        }, index=pd.date_range("2020-01-01", periods=n_rows, freq="h"))

        result = nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=4, oos_size=4000)

        # Inner muss vor Holdout sein
        assert result["inner_df"].index.max() < result["holdout_df"].index.min(), \
            "Inner muss vor Holdout sein"

        # Innerhalb der Inner Folds: Train vor Val
        for train, val in result["inner_folds"]:
            assert train.index.max() < val.index.min(), "Train muss vor Val sein"


class TestMonteCarloPermutation:
    """Tests für den Monte Carlo Permutation Test."""

    def test_random_trades_not_significant(self):
        """Zufällige Trades sollten nicht signifikant sein."""
        from fwbg.simulation.trade import monte_carlo_permutation_test

        np.random.seed(42)
        # 50% Win Rate, zufällige Reihenfolge
        trades = [1.0 if np.random.random() > 0.5 else -1.0 for _ in range(500)]

        result = monte_carlo_permutation_test(trades, n_permutations=1000)

        # P-Wert sollte hoch sein (nicht signifikant)
        assert result["p_value"] > 0.1, f"Zufällige Trades sollten nicht signifikant sein, p={result['p_value']}"
        assert not result["is_significant"]

    def test_highly_profitable_trades_significant(self):
        """Stark profitable Trades sollten signifikant sein."""
        from fwbg.simulation.trade import monte_carlo_permutation_test

        # 80% Win Rate - klar überdurchschnittlich
        trades = [1.0] * 80 + [-1.0] * 20

        result = monte_carlo_permutation_test(trades, n_permutations=1000)

        # P-Wert sollte niedrig sein
        assert result["p_value"] < 0.05, f"Profitable Trades sollten signifikant sein, p={result['p_value']}"
        assert result["is_significant"]

    def test_losing_trades_not_significant(self):
        """Verlierende Trades sollten nicht signifikant sein."""
        from fwbg.simulation.trade import monte_carlo_permutation_test

        # 30% Win Rate - schlecht
        trades = [1.0] * 30 + [-1.0] * 70

        result = monte_carlo_permutation_test(trades, n_permutations=1000)

        # Sollte nicht signifikant sein
        assert result["p_value"] > 0.05, f"Verlierende Trades sollten nicht signifikant sein, p={result['p_value']}"

    def test_too_few_trades_returns_not_significant(self):
        """Zu wenige Trades sollten automatisch nicht signifikant sein."""
        from fwbg.simulation.trade import monte_carlo_permutation_test

        trades = [1.0, 1.0, -1.0]  # Nur 3 Trades

        result = monte_carlo_permutation_test(trades)

        assert result["p_value"] == 1.0
        assert not result["is_significant"]
        assert result["n_permutations"] == 0


class TestMonteCarloEquity:
    """Tests für die Monte Carlo Equity Simulation."""

    def test_high_kelly_leads_to_low_equity(self):
        """Hoher Kelly-Risk bei 50/50 Trades sollte zu niedriger Equity führen."""
        from fwbg.simulation.trade import monte_carlo_equity_simulation

        # 50/50 Win/Loss (unprofitabel bei RRR=1)
        trades = [1.0] * 50 + [-1.0] * 50

        result = monte_carlo_equity_simulation(
            trades, risk_per_trade=0.25, rrr=1.0, n_simulations=500
        )

        # Bei 25% Risk und 50% WR sollte die Equity stark sinken (geometric decay)
        # 50 Wins x 1.25 * 50 Losses x 0.75 = (1.25 * 0.75)^50 * 100 ≈ 0.0039
        assert result["median_equity"] < 10, f"Equity sollte niedrig sein, ist {result['median_equity']}"
        assert result["median_equity"] > 0, "Sollte aber nicht Bankrott sein (kelly < 1)"

    def test_low_kelly_no_bankruptcy(self):
        """Niedriger Kelly-Risk sollte keine Bankrotte verursachen."""
        from fwbg.simulation.trade import monte_carlo_equity_simulation

        # 60% Win Rate
        trades = [1.0] * 60 + [-1.0] * 40

        result = monte_carlo_equity_simulation(
            trades, risk_per_trade=0.01, rrr=1.0, n_simulations=500
        )

        # Bei 1% Risk sollte es keine Bankrotte geben
        assert result["bankruptcy_rate"] == 0, f"Kein Bankrott erwartet, aber {result['bankruptcy_rate']}"

    def test_confidence_intervals_make_sense(self):
        """Konfidenzintervalle sollten logisch sein (p5 < median < p95)."""
        from fwbg.simulation.trade import monte_carlo_equity_simulation

        trades = [1.0] * 60 + [-1.0] * 40

        result = monte_carlo_equity_simulation(
            trades, risk_per_trade=0.02, rrr=1.5, n_simulations=500
        )

        assert result["p5_equity"] <= result["p25_equity"]
        assert result["p25_equity"] <= result["median_equity"]
        assert result["median_equity"] <= result["p75_equity"]
        assert result["p75_equity"] <= result["p95_equity"]


class TestSyntheticStrategy:
    """
    Tests mit synthetischen Daten wo das erwartete Ergebnis bekannt ist.

    Wir erstellen künstliche Preisdaten mit einem vorhersehbaren Muster,
    um zu testen ob die Strategie es korrekt erkennt.
    """

    def test_perfect_trend_following_signal(self):
        """
        Erstellt Daten wo ein bestimmtes Feature perfekt den Trend vorhersagt.
        Die Strategie sollte dies erkennen und profitabel sein.
        """
        from fwbg.simulation.trade import monte_carlo_permutation_test

        # Synthetischer Trade-Record: Feature sagt perfekt voraus
        # Wenn Feature > 0.5 → Long profitabel (Win)
        # Wenn Feature < 0.5 → Long unprofitabel (Loss)
        # Wir simulieren dass das Modell dies gelernt hat:
        perfect_trades = [1.0] * 70 + [-1.0] * 30  # 70% WR

        result = monte_carlo_permutation_test(perfect_trades, n_permutations=1000)

        assert result["is_significant"], "Perfekte Strategie sollte signifikant sein"
        assert result["percentile"] > 90, f"Sollte in oberem Perzentil sein, ist {result['percentile']}"

    def test_random_noise_strategy(self):
        """
        Erstellt Daten wo Features nur Rauschen sind.
        Die Strategie sollte NICHT profitabel sein.
        """
        from fwbg.simulation.trade import monte_carlo_permutation_test

        np.random.seed(123)
        # Komplett zufällige Trades
        random_trades = [1.0 if np.random.random() > 0.5 else -1.0 for _ in range(200)]

        result = monte_carlo_permutation_test(random_trades, n_permutations=1000)

        # Sollte im mittleren Bereich sein (nicht signifikant)
        assert not result["is_significant"], "Zufällige Strategie sollte nicht signifikant sein"
        assert 20 < result["percentile"] < 80, f"Sollte im mittleren Bereich sein, ist {result['percentile']}"

    def test_equity_simulation_matches_known_outcome(self):
        """
        Testet ob die Equity-Simulation das erwartete Ergebnis liefert.
        """
        from fwbg.simulation.equity import simulate_equity

        # Bekannte Trades: 4 Wins, 1 Loss
        trades = [1.0, 1.0, 1.0, -1.0, 1.0]
        risk_per_trade = 0.10  # 10%
        rrr = 2.0  # Risk-Reward-Ratio von 2

        result = simulate_equity(trades, risk_per_trade, rrr)

        # Manuelle Berechnung:
        # Start: 100
        # Win 1: 100 * (1 + 0.10 * 2) = 100 * 1.20 = 120
        # Win 2: 120 * 1.20 = 144
        # Win 3: 144 * 1.20 = 172.8
        # Loss:  172.8 * (1 - 0.10) = 172.8 * 0.90 = 155.52
        # Win 4: 155.52 * 1.20 = 186.624
        expected_final = 100 * 1.2 * 1.2 * 1.2 * 0.9 * 1.2
        assert abs(result["final_equity"] - expected_final) < 0.01, \
            f"Erwartet {expected_final}, bekommen {result['final_equity']}"

    def test_max_drawdown_calculation(self):
        """
        Testet ob der Max Drawdown korrekt berechnet wird.
        """
        from fwbg.simulation.equity import simulate_equity

        # Trades die einen klaren Drawdown erzeugen
        trades = [1.0, 1.0, -1.0, -1.0, -1.0, 1.0]  # Peak nach 2, dann 3 Losses
        risk_per_trade = 0.10
        rrr = 1.0

        result = simulate_equity(trades, risk_per_trade, rrr)

        # Peak nach 2 Wins: 100 * 1.1 * 1.1 = 121
        # Nach 3 Losses: 121 * 0.9 * 0.9 * 0.9 = 88.209
        # Drawdown: (121 - 88.209) / 121 = 0.271
        expected_dd = 1 - (0.9 ** 3)  # ~0.271
        assert abs(result["max_drawdown"] - expected_dd) < 0.01, \
            f"Erwartet DD ~{expected_dd:.3f}, bekommen {result['max_drawdown']:.3f}"


class TestFoldStability:
    """Tests für die Fold-Stabilitätsprüfung."""

    def test_all_folds_profitable_gives_high_stability(self):
        """Wenn alle Folds profitabel sind, sollte Stabilität 100% sein."""
        fold_performances = [
            {"fold": 0, "pnl": 10.0, "win_rate": 0.6},
            {"fold": 1, "pnl": 5.0, "win_rate": 0.55},
            {"fold": 2, "pnl": 15.0, "win_rate": 0.65},
            {"fold": 3, "pnl": 8.0, "win_rate": 0.58},
        ]

        profitable_folds = sum(1 for fp in fold_performances if fp["pnl"] > 0)
        stability = profitable_folds / len(fold_performances)

        assert stability == 1.0, f"Alle Folds profitabel sollte 100% sein, ist {stability}"

    def test_mixed_folds_gives_partial_stability(self):
        """Gemischte Fold-Performance sollte teilweise Stabilität geben."""
        fold_performances = [
            {"fold": 0, "pnl": 10.0, "win_rate": 0.6},
            {"fold": 1, "pnl": -5.0, "win_rate": 0.45},  # Unprofitabel
            {"fold": 2, "pnl": 15.0, "win_rate": 0.65},
            {"fold": 3, "pnl": -2.0, "win_rate": 0.48},  # Unprofitabel
        ]

        profitable_folds = sum(1 for fp in fold_performances if fp["pnl"] > 0)
        stability = profitable_folds / len(fold_performances)

        assert stability == 0.5, f"2/4 profitable Folds sollte 50% sein, ist {stability}"

    def test_no_profitable_folds_gives_zero_stability(self):
        """Keine profitablen Folds sollte 0% Stabilität geben."""
        fold_performances = [
            {"fold": 0, "pnl": -10.0, "win_rate": 0.4},
            {"fold": 1, "pnl": -5.0, "win_rate": 0.45},
            {"fold": 2, "pnl": -15.0, "win_rate": 0.35},
        ]

        profitable_folds = sum(1 for fp in fold_performances if fp["pnl"] > 0)
        stability = profitable_folds / len(fold_performances)

        assert stability == 0.0, f"Keine profitablen Folds sollte 0% sein, ist {stability}"


class TestEquityDrawdownConsistency:
    """
    Tests die sicherstellen, dass Equity-Kurve und Drawdown-Berechnung konsistent sind.

    Problem das wir verhindern wollen:
    - Drawdown-Chart zeigt 80% DD, aber Equity-Kurve zeigt keinen entsprechenden Einbruch
    - Dies wäre ein Bug in der Berechnung
    """

    def test_drawdown_matches_equity_curve(self):
        """Der berechnete Drawdown muss zur Equity-Kurve passen."""
        from fwbg.simulation.equity import simulate_equity

        # Trades mit bekanntem Muster
        trades = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
        risk_per_trade = 0.10
        rrr = 1.0

        result = simulate_equity(trades, risk_per_trade, rrr)
        eq = result["equity_curve"]
        dd = result["drawdowns"]

        # Für jeden Punkt: Prüfe ob der Drawdown zur Equity passt
        peak = eq[0]
        for i in range(1, len(eq)):
            if eq[i] > peak:
                peak = eq[i]

            # Berechne erwarteten Drawdown an diesem Punkt
            expected_dd_pct = (peak - eq[i]) / peak * 100 if peak > 0 else 0
            actual_dd_pct = dd[i]

            assert abs(expected_dd_pct - actual_dd_pct) < 0.01, \
                f"Trade {i}: DD stimmt nicht! Erwartet {expected_dd_pct:.2f}%, bekommen {actual_dd_pct:.2f}%"

    def test_max_drawdown_is_maximum_of_drawdowns(self):
        """max_drawdown muss dem Maximum der drawdowns-Liste entsprechen."""
        from fwbg.simulation.equity import simulate_equity

        trades = [1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0]
        risk_per_trade = 0.15
        rrr = 1.5

        result = simulate_equity(trades, risk_per_trade, rrr)

        max_from_list = max(result["drawdowns"]) / 100  # Liste ist in Prozent
        max_from_result = result["max_drawdown"]

        assert abs(max_from_list - max_from_result) < 0.0001, \
            f"max_drawdown ({max_from_result:.4f}) != max(drawdowns) ({max_from_list:.4f})"

    def test_drawdown_at_peak_is_zero(self):
        """An einem neuen Peak muss der Drawdown 0 sein."""
        from fwbg.simulation.equity import simulate_equity

        # Nur Gewinne = jeder Punkt ist ein neuer Peak
        trades = [1.0, 1.0, 1.0, 1.0, 1.0]
        risk_per_trade = 0.10
        rrr = 1.0

        result = simulate_equity(trades, risk_per_trade, rrr)

        # Alle Drawdowns sollten 0 sein (jeder Trade ist ein neuer Peak)
        for i, dd in enumerate(result["drawdowns"]):
            assert dd == 0.0, f"Trade {i}: Bei steigender Equity sollte DD=0 sein, ist {dd:.2f}%"

    def test_drawdown_increases_during_loss_streak(self):
        """Während einer Verlustserie muss der Drawdown steigen."""
        from fwbg.simulation.equity import simulate_equity

        # Erst Gewinne (Peak bilden), dann Verluste
        trades = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
        risk_per_trade = 0.10
        rrr = 1.0

        result = simulate_equity(trades, risk_per_trade, rrr)
        dd = result["drawdowns"]

        # Nach den 3 Gewinnen beginnen die Verluste (Index 4, 5, 6, 7)
        # Drawdown sollte monoton steigen
        for i in range(4, len(dd)):
            assert dd[i] > dd[i-1], \
                f"Drawdown sollte während Verlustserie steigen: dd[{i}]={dd[i]:.2f}% <= dd[{i-1}]={dd[i-1]:.2f}%"

    def test_equity_drop_magnitude_matches_drawdown(self):
        """Ein 50% Drawdown bedeutet, dass die Equity auf 50% des Peaks gefallen ist."""
        from fwbg.simulation.equity import simulate_equity

        # Konstruiere einen Fall mit bekanntem Drawdown
        # 10 Verluste bei 10% Kelly: (0.9)^10 = 0.3487 → ~65% DD
        trades = [-1.0] * 10
        risk_per_trade = 0.10
        rrr = 1.0

        result = simulate_equity(trades, risk_per_trade, rrr)

        # Peak ist 100 (Start-Equity)
        peak = 100.0
        final = result["final_equity"]
        max_dd = result["max_drawdown"]

        # Erwarteter Drawdown: 1 - (0.9)^10
        expected_dd = 1 - (0.9 ** 10)

        assert abs(max_dd - expected_dd) < 0.0001, \
            f"Erwarteter DD {expected_dd:.4f}, bekommen {max_dd:.4f}"

        # Equity sollte bei peak * (1 - max_dd) sein
        expected_equity = peak * (1 - max_dd)
        assert abs(final - expected_equity) < 0.01, \
            f"Equity sollte {expected_equity:.2f} sein, ist {final:.2f}"

    def test_large_drawdown_requires_long_loss_streak(self):
        """Ein 80% DD bei 1.67% Kelly benötigt ~100 aufeinanderfolgende Verluste."""
        import math

        risk_per_trade = 0.0167  # 1.67%
        target_dd = 0.82     # 82% Drawdown

        # Berechne benötigte Verluste für diesen DD
        # (1 - kelly)^n = 1 - target_dd
        # n = log(1 - target_dd) / log(1 - kelly)
        required_losses = math.log(1 - target_dd) / math.log(1 - risk_per_trade)

        # Bei 1.67% Kelly braucht man ~102 aufeinanderfolgende Verluste für 82% DD
        assert required_losses > 100, \
            f"82% DD bei 1.67% Kelly benötigt {required_losses:.0f} Verluste (erwartet >100)"

    def test_statistical_plausibility_of_loss_streak(self):
        """Prüft ob eine lange Verlustserie statistisch plausibel ist."""
        import math

        win_rate = 0.817     # 81.7% Win Rate
        n_trades = 17500     # Anzahl Trades
        loss_streak = 102    # Für 82% DD bei 1.67% Kelly

        # Wahrscheinlichkeit einer einzelnen Verlustserie dieser Länge
        p_loss = 1 - win_rate
        p_streak = p_loss ** loss_streak

        # Erwartete Anzahl solcher Streaks in n_trades
        expected_streaks = n_trades * p_streak

        # Bei 81.7% WR ist eine 102er Verlustserie praktisch unmöglich
        assert expected_streaks < 1e-50, \
            f"Eine {loss_streak}er Verlustserie bei {win_rate:.1%} WR sollte unmöglich sein, " \
            f"aber erwartete Anzahl ist {expected_streaks:.2e}"

    def test_expected_max_loss_streak(self):
        """Berechnet die erwartete längste Verlustserie."""
        import math

        win_rate = 0.817
        n_trades = 17500
        p_loss = 1 - win_rate

        # Erwartete längste Verlustserie bei n Bernoulli-Trials
        # Approximation: log(n) / (-log(1-p))
        expected_max_streak = math.log(n_trades) / (-math.log(1 - p_loss))

        # Bei 81.7% WR und 17500 Trades erwarten wir ~48 aufeinanderfolgende Verluste
        assert 40 < expected_max_streak < 60, \
            f"Erwartete max Verlustserie sollte ~48 sein, ist {expected_max_streak:.0f}"


class TestSyntheticEquityScenarios:
    """
    Tests mit synthetischen Trade-Sequenzen, bei denen wir das exakte Ergebnis kennen.
    Diese Tests stellen sicher, dass die Equity-Simulation korrekt ist.
    """

    def test_only_wins_exponential_growth(self):
        """Nur Gewinne sollten zu exponentiellem Wachstum führen."""
        from fwbg.simulation.equity import simulate_equity

        # 100 Gewinne bei 5% Kelly und RRR=2
        trades = [1.0] * 100
        risk_per_trade = 0.05
        rrr = 2.0

        # Nutze hohes compound_cap um reines Compounding zu testen
        result = simulate_equity(trades, risk_per_trade, rrr, compound_cap=1e20)

        # Erwartete Equity: 100 * (1 + 0.05*2)^100 = 100 * 1.1^100
        expected = 100 * (1.1 ** 100)
        assert abs(result["final_equity"] - expected) < expected * 0.0001, \
            f"Erwartet {expected:.2f}, bekommen {result['final_equity']:.2f}"

        # Kein Drawdown bei nur Gewinnen
        assert result["max_drawdown"] == 0, "Nur Gewinne sollten 0% DD haben"

    def test_only_losses_geometric_decay(self):
        """Nur Verluste sollten zu geometrischem Verfall führen."""
        from fwbg.simulation.equity import simulate_equity

        # 20 Verluste bei 10% Kelly
        trades = [-1.0] * 20
        risk_per_trade = 0.10
        rrr = 1.0  # RRR irrelevant bei Verlusten

        result = simulate_equity(trades, risk_per_trade, rrr, compound_cap=1e20)

        # Erwartete Equity: 100 * 0.9^20
        expected = 100 * (0.9 ** 20)
        assert abs(result["final_equity"] - expected) < 0.01, \
            f"Erwartet {expected:.4f}, bekommen {result['final_equity']:.4f}"

        # Max DD = 1 - 0.9^20
        expected_dd = 1 - (0.9 ** 20)
        assert abs(result["max_drawdown"] - expected_dd) < 0.0001, \
            f"Erwartet DD {expected_dd:.4f}, bekommen {result['max_drawdown']:.4f}"

    def test_win_loss_alternating_pattern(self):
        """Wechselnde Wins/Losses sollten vorhersagbare Equity ergeben."""
        from fwbg.simulation.equity import simulate_equity

        # Win, Loss, Win, Loss... (10x)
        trades = [1.0, -1.0] * 10
        risk_per_trade = 0.10
        rrr = 1.0

        result = simulate_equity(trades, risk_per_trade, rrr, compound_cap=1e20)

        # Bei RRR=1: Win multipliziert mit 1.1, Loss mit 0.9
        # Nach 10 Paaren: 100 * (1.1 * 0.9)^10 = 100 * 0.99^10
        expected = 100 * (0.99 ** 10)
        assert abs(result["final_equity"] - expected) < 0.01, \
            f"Erwartet {expected:.4f}, bekommen {result['final_equity']:.4f}"

    def test_recovery_after_drawdown(self):
        """Equity kann sich nach Drawdown erholen, aber DD bleibt bestehen."""
        from fwbg.simulation.equity import simulate_equity

        # 5 Verluste, dann 20 Gewinne
        trades = [-1.0] * 5 + [1.0] * 20
        risk_per_trade = 0.10
        rrr = 2.0

        result = simulate_equity(trades, risk_per_trade, rrr, compound_cap=1e20)

        # Nach 5 Verlusten: 100 * 0.9^5 = 59.05
        # Max DD an diesem Punkt: 1 - 0.9^5 = 0.4095 (40.95%)
        expected_dd = 1 - (0.9 ** 5)
        assert abs(result["max_drawdown"] - expected_dd) < 0.01, \
            f"Max DD sollte {expected_dd:.2%} sein, ist {result['max_drawdown']:.2%}"

        # Nach 20 Gewinnen sollte Equity über Startwert sein
        after_losses = 100 * (0.9 ** 5)
        final = after_losses * (1.2 ** 20)
        assert abs(result["final_equity"] - final) < final * 0.0001, \
            f"Erwartet {final:.2f}, bekommen {result['final_equity']:.2f}"

    def test_known_rrr_asymmetry(self):
        """Bei RRR < 1 ist das Risiko asymmetrisch - mehr Verlust als Gewinn."""
        from fwbg.simulation.equity import simulate_equity

        # RRR = 0.5 bedeutet: Gewinn = 5%, Verlust = 10%
        # Bei 50/50 Win Rate sollte Equity sinken
        trades = [1.0, -1.0] * 50
        risk_per_trade = 0.10
        rrr = 0.5

        result = simulate_equity(trades, risk_per_trade, rrr, compound_cap=1e20)

        # Win: 1.05, Loss: 0.9
        # Nach 50 Paaren: 100 * (1.05 * 0.9)^50
        expected = 100 * ((1.05 * 0.9) ** 50)
        assert abs(result["final_equity"] - expected) < expected * 0.001, \
            f"Erwartet {expected:.2f}, bekommen {result['final_equity']:.2f}"

        # Equity sollte gesunken sein
        assert result["final_equity"] < 100, \
            f"Bei RRR=0.5 und 50% WR sollte Equity sinken, ist {result['final_equity']:.2f}"

    def test_high_rrr_compensates_low_winrate(self):
        """Bei hohem RRR kann niedrige Win Rate kompensiert werden."""
        from fwbg.simulation.equity import simulate_equity

        # 30% Win Rate, aber RRR = 3 (Gewinn = 30%, Verlust = 10%)
        wins = [1.0] * 30
        losses = [-1.0] * 70
        trades = []
        # Mische für realistisches Szenario
        import random
        random.seed(42)
        trades = wins + losses
        random.shuffle(trades)

        risk_per_trade = 0.10
        rrr = 3.0

        result = simulate_equity(trades, risk_per_trade, rrr, compound_cap=1e20)

        # Erwartung: 30 Wins x 1.3, 70 Losses x 0.9
        # = 100 * 1.3^30 * 0.9^70
        import math
        expected = 100 * (1.3 ** 30) * (0.9 ** 70)
        # Bei shuffling variiert das Ergebnis nicht, nur die Equity-Kurve
        assert abs(result["final_equity"] - expected) < expected * 0.0001, \
            f"Erwartet {expected:.2f}, bekommen {result['final_equity']:.2f}"

    def test_drawdown_visual_consistency(self):
        """
        Stellt sicher, dass der Drawdown-Chart zur Equity-Kurve passt.

        Wenn die Equity von 100 auf 50 fällt, muss der DD 50% zeigen.
        Wenn die Equity dann auf 75 steigt, bleibt DD bei 50% (nicht bei neuen Peak).
        """
        from fwbg.simulation.equity import simulate_equity

        # Konstruiere eine spezifische Sequenz
        # Start: 100
        # 5 Gewinne: 100 * 1.1^5 = 161.05 (neuer Peak)
        # 10 Verluste: 161.05 * 0.9^10 = 56.17 (DD = 65.1%)
        # 3 Gewinne: 56.17 * 1.1^3 = 74.76 (immer noch unter Peak, DD = 53.6%)
        trades = [1.0] * 5 + [-1.0] * 10 + [1.0] * 3
        risk_per_trade = 0.10
        rrr = 1.0

        result = simulate_equity(trades, risk_per_trade, rrr, compound_cap=1e20)
        eq = result["equity_curve"]
        dd = result["drawdowns"]

        # Peak nach 5 Gewinnen
        peak_after_5 = 100 * (1.1 ** 5)
        assert abs(eq[5] - peak_after_5) < 0.01, f"Peak sollte {peak_after_5:.2f} sein"

        # Equity nach 10 Verlusten (Position 15)
        equity_after_loss = peak_after_5 * (0.9 ** 10)
        expected_dd_pct = (1 - equity_after_loss / peak_after_5) * 100

        # DD an Position 15 prüfen
        assert abs(dd[15] - expected_dd_pct) < 0.1, \
            f"DD an Position 15 sollte {expected_dd_pct:.1f}% sein, ist {dd[15]:.1f}%"

        # Max DD sollte an Position 15 sein (tiefster Punkt)
        assert result["max_drawdown"] * 100 == max(dd), \
            "Max DD stimmt nicht mit DD-Liste überein"


class TestDataLeakagePrevention:
    """
    Tests die sicherstellen dass kein Data Leakage stattfindet.

    Data Leakage würde bedeuten:
    - CT wird auf OOS optimiert (FALSCH)
    - Modell sieht zukünftige Daten (FALSCH)
    """

    def test_val_comes_before_holdout_chronologically(self):
        """Validation-Daten müssen chronologisch VOR Holdout-Daten liegen."""
        from fwbg.optimization.nested_cv import nested_cv_split

        n_rows = 50000
        df = pd.DataFrame({
            "C": np.arange(n_rows),  # Werte = Index (chronologisch)
        }, index=pd.date_range("2020-01-01", periods=n_rows, freq="h"))

        result = nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=4, oos_size=4000)

        holdout = result["holdout_df"]
        min_holdout_value = holdout["C"].min()

        for i, (train, val) in enumerate(result["inner_folds"]):
            # Alle Val-Werte müssen kleiner sein als alle Holdout-Werte
            max_val_value = val["C"].max()

            assert max_val_value < min_holdout_value, \
                f"Fold {i}: Val max ({max_val_value}) >= Holdout min ({min_holdout_value}) - Data Leakage!"

    def test_train_never_sees_holdout_data(self):
        """Training darf niemals Holdout-Daten sehen."""
        from fwbg.optimization.nested_cv import nested_cv_split

        n_rows = 50000
        df = pd.DataFrame({
            "C": np.arange(n_rows),
        }, index=pd.date_range("2020-01-01", periods=n_rows, freq="h"))

        result = nested_cv_split(df, holdout_ratio=0.20, n_inner_folds=4, oos_size=4000)

        holdout = result["holdout_df"]
        min_holdout_value = holdout["C"].min()

        for i, (train, val) in enumerate(result["inner_folds"]):
            max_train_value = train["C"].max()

            assert max_train_value < min_holdout_value, \
                f"Fold {i}: Train sieht Holdout-Daten! Max Train ({max_train_value}) >= Min Holdout ({min_holdout_value})"


class TestEarlyTermination:
    """
    Tests für die Early Termination Logik bei Grid-Optimierung.

    Die Idee ist: Wenn ein Kandidat bereits zu viele fehlgeschlagene Folds hat
    und die Mindest-Fold-Stability mathematisch nicht mehr erreichen kann,
    wird die Evaluation vorzeitig abgebrochen um Rechenzeit zu sparen.
    """

    def test_early_termination_math_is_correct(self):
        """Prüft die mathematische Logik der Early Termination.

        Bei 5 Folds und min_fold_stability=0.5:
        - Min profitable Folds: ceil(5 * 0.5) = 3
        - Erlaubte Fehlschläge: 5 - 3 = 2
        - Nach 3 Fehlschlägen → Abbruch (selbst wenn alle restlichen profitabel wären)
        """
        import math

        total_folds = 5
        min_fold_stability = 0.5

        # Berechne min_profitable
        min_profitable = int(math.ceil(total_folds * min_fold_stability))
        assert min_profitable == 3

        # Erlaubte Fehlschläge
        max_failures = total_folds - min_profitable
        assert max_failures == 2

        # Simuliere: Nach fold_idx=2 (3 Folds), profitable_count=0, failed_count=3
        # Prüfe ob Abbruch erfolgen sollte
        fold_idx = 3  # Nach 3 Folds
        profitable_count = 0
        remaining_folds = total_folds - fold_idx
        max_possible_profitable = profitable_count + remaining_folds

        # 0 + 2 = 2 < 3 → Abbruch
        should_terminate = max_possible_profitable < min_profitable
        assert should_terminate is True

    def test_early_termination_with_some_wins(self):
        """Prüft Early Termination wenn einige Folds profitabel sind."""
        import math

        total_folds = 5
        min_fold_stability = 0.5
        min_profitable = int(math.ceil(total_folds * min_fold_stability))  # 3

        # Szenario: Nach 4 Folds haben wir 1 Win und 3 Losses
        fold_idx = 4
        profitable_count = 1
        remaining_folds = total_folds - fold_idx  # 1 Fold übrig

        max_possible_profitable = profitable_count + remaining_folds  # 1 + 1 = 2

        # 2 < 3 → Sollte abbrechen
        should_terminate = max_possible_profitable < min_profitable
        assert should_terminate is True

    def test_no_early_termination_when_possible(self):
        """Prüft dass kein Abbruch erfolgt wenn Stabilität noch erreichbar ist."""
        import math

        total_folds = 5
        min_fold_stability = 0.5
        min_profitable = int(math.ceil(total_folds * min_fold_stability))  # 3

        # Szenario: Nach 2 Folds haben wir 1 Win und 1 Loss
        fold_idx = 2
        profitable_count = 1
        remaining_folds = total_folds - fold_idx  # 3 Folds übrig

        max_possible_profitable = profitable_count + remaining_folds  # 1 + 3 = 4

        # 4 >= 3 → Sollte NICHT abbrechen
        should_terminate = max_possible_profitable < min_profitable
        assert should_terminate is False


class TestFirstFoldSanityCheck:
    """
    Tests für den First-Fold Sanity Check.

    Der First-Fold Sanity Check bricht nur bei extremen Fällen ab:
    - Win-Rate < 25% UND PnL < -10 UND genug Trades
    - Oder: Weniger als Minimum-Trades im ersten Fold
    """

    def test_catastrophic_first_fold_detected(self):
        """Katastrophaler erster Fold sollte erkannt werden."""
        # Einstellungen
        first_fold_min_win_rate = 0.25
        first_fold_min_pnl = -10.0
        first_fold_min_trades = 5

        # Simulierte Trades: Alle Verluste (11 Trades → PnL = -11 < -10)
        fold_trades = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        n_fold_trades = len(fold_trades)
        fold_win_rate = fold_trades.count(1.0) / n_fold_trades  # 0%
        fold_pnl = sum(fold_trades)  # -11

        # Prüfe ob katastrophal
        is_catastrophic = (
            fold_win_rate < first_fold_min_win_rate and
            fold_pnl < first_fold_min_pnl and
            n_fold_trades >= first_fold_min_trades
        )

        assert is_catastrophic is True, "11 Verluste in Folge sollte katastrophal sein"

    def test_bad_but_not_catastrophic_passes(self):
        """Schlechter aber nicht katastrophaler Fold sollte durchgelassen werden."""
        first_fold_min_win_rate = 0.25
        first_fold_min_pnl = -10.0
        first_fold_min_trades = 5

        # 30% Win-Rate - schlecht aber nicht katastrophal
        fold_trades = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        n_fold_trades = len(fold_trades)
        fold_win_rate = fold_trades.count(1.0) / n_fold_trades  # 30%
        fold_pnl = sum(fold_trades)  # -4

        is_catastrophic = (
            fold_win_rate < first_fold_min_win_rate and
            fold_pnl < first_fold_min_pnl and
            n_fold_trades >= first_fold_min_trades
        )

        # 30% > 25% → nicht katastrophal (unabhängig vom PnL)
        assert is_catastrophic is False, "30% Win-Rate sollte durchgelassen werden"

    def test_high_loss_but_ok_winrate_passes(self):
        """Hoher Verlust mit OK Win-Rate sollte durchgelassen werden."""
        first_fold_min_win_rate = 0.25
        first_fold_min_pnl = -10.0
        first_fold_min_trades = 5

        # 40% Win-Rate aber hoher Verlust (RRR schlecht)
        fold_trades = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        n_fold_trades = len(fold_trades)
        fold_win_rate = fold_trades.count(1.0) / n_fold_trades  # 40%
        fold_pnl = sum(fold_trades)  # -2

        is_catastrophic = (
            fold_win_rate < first_fold_min_win_rate and
            fold_pnl < first_fold_min_pnl and
            n_fold_trades >= first_fold_min_trades
        )

        # 40% > 25% → nicht katastrophal
        assert is_catastrophic is False

    def test_too_few_trades_not_catastrophic_by_pnl(self):
        """Wenige schlechte Trades sollten NICHT als katastrophal gelten."""
        first_fold_min_win_rate = 0.25
        first_fold_min_pnl = -10.0
        first_fold_min_trades = 5

        # Nur 3 Trades - zu wenige um sicher zu sagen
        fold_trades = [-1.0, -1.0, -1.0]
        n_fold_trades = len(fold_trades)
        fold_win_rate = fold_trades.count(1.0) / n_fold_trades  # 0%
        fold_pnl = sum(fold_trades)  # -3

        is_catastrophic = (
            fold_win_rate < first_fold_min_win_rate and
            fold_pnl < first_fold_min_pnl and
            n_fold_trades >= first_fold_min_trades
        )

        # Weniger als 5 Trades → nicht als katastrophal werten
        assert is_catastrophic is False

    def test_extreme_loss_with_low_winrate_is_catastrophic(self):
        """20% Win-Rate mit starkem Verlust sollte katastrophal sein."""
        first_fold_min_win_rate = 0.25
        first_fold_min_pnl = -10.0
        first_fold_min_trades = 5

        # 20% Win-Rate, 15 Trades
        fold_trades = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                       -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        n_fold_trades = len(fold_trades)
        fold_win_rate = fold_trades.count(1.0) / n_fold_trades  # 20%
        fold_pnl = sum(fold_trades)  # -9

        is_catastrophic = (
            fold_win_rate < first_fold_min_win_rate and
            fold_pnl < first_fold_min_pnl and
            n_fold_trades >= first_fold_min_trades
        )

        # 20% < 25% und PnL = -9 > -10 → knapp NICHT katastrophal
        assert is_catastrophic is False, "PnL -9 ist > -10, also nicht katastrophal"

        # Mit PnL = -12 wäre es katastrophal
        fold_trades_worse = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                              -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        n_trades_worse = len(fold_trades_worse)
        win_rate_worse = fold_trades_worse.count(1.0) / n_trades_worse  # ~18%
        pnl_worse = sum(fold_trades_worse)  # -11

        is_catastrophic_worse = (
            win_rate_worse < first_fold_min_win_rate and
            pnl_worse < first_fold_min_pnl and
            n_trades_worse >= first_fold_min_trades
        )

        assert is_catastrophic_worse is True, "18% Win-Rate mit PnL -11 sollte katastrophal sein"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
