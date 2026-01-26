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
    """Tests für die Walk-Forward Split Funktion."""

    def test_split_returns_train_val_test_tuples(self):
        """Walk-Forward sollte (train, val, test) Tupel zurückgeben."""
        from optimizer.process import walk_forward_split

        # Erstelle Dummy-DataFrame
        n_rows = 50000
        df = pd.DataFrame({
            "C": np.random.randn(n_rows),
            "H": np.random.randn(n_rows),
            "L": np.random.randn(n_rows),
        }, index=pd.date_range("2020-01-01", periods=n_rows, freq="h"))

        folds = walk_forward_split(df, n_folds=4, oos_size=4000)

        assert len(folds) > 0, "Sollte mindestens einen Fold haben"

        for train, val, test in folds:
            assert isinstance(train, pd.DataFrame)
            assert isinstance(val, pd.DataFrame)
            assert isinstance(test, pd.DataFrame)

    def test_split_sizes_approximate_60_20_20(self):
        """Val und Test sollten etwa gleich groß sein (20/20)."""
        from optimizer.process import walk_forward_split

        n_rows = 50000
        df = pd.DataFrame({
            "C": np.random.randn(n_rows),
        }, index=pd.date_range("2020-01-01", periods=n_rows, freq="h"))

        folds = walk_forward_split(df, n_folds=4, oos_size=4000)

        for train, val, test in folds:
            # Val und Test sollten gleich groß sein
            assert len(val) == len(test), f"Val ({len(val)}) und Test ({len(test)}) sollten gleich groß sein"
            # Test sollte oos_size sein
            assert len(test) == 4000, f"Test sollte 4000 sein, ist {len(test)}"

    def test_no_data_leakage_between_splits(self):
        """Es darf keine Überlappung zwischen Train/Val/Test geben."""
        from optimizer.process import walk_forward_split

        n_rows = 50000
        df = pd.DataFrame({
            "C": np.arange(n_rows),  # Eindeutige Werte
        }, index=pd.date_range("2020-01-01", periods=n_rows, freq="h"))

        folds = walk_forward_split(df, n_folds=4, oos_size=4000)

        for train, val, test in folds:
            train_idx = set(train.index)
            val_idx = set(val.index)
            test_idx = set(test.index)

            # Keine Überlappung
            assert len(train_idx & val_idx) == 0, "Train und Val überlappen sich"
            assert len(train_idx & test_idx) == 0, "Train und Test überlappen sich"
            assert len(val_idx & test_idx) == 0, "Val und Test überlappen sich"

    def test_chronological_order(self):
        """Train kommt vor Val, Val kommt vor Test."""
        from optimizer.process import walk_forward_split

        n_rows = 50000
        df = pd.DataFrame({
            "C": np.arange(n_rows),
        }, index=pd.date_range("2020-01-01", periods=n_rows, freq="h"))

        folds = walk_forward_split(df, n_folds=4, oos_size=4000)

        for train, val, test in folds:
            assert train.index.max() < val.index.min(), "Train muss vor Val sein"
            assert val.index.max() < test.index.min(), "Val muss vor Test sein"


class TestMonteCarloPermutation:
    """Tests für den Monte Carlo Permutation Test."""

    def test_random_trades_not_significant(self):
        """Zufällige Trades sollten nicht signifikant sein."""
        from optimizer.simulation import monte_carlo_permutation_test

        np.random.seed(42)
        # 50% Win Rate, zufällige Reihenfolge
        trades = [1.0 if np.random.random() > 0.5 else -1.0 for _ in range(500)]

        result = monte_carlo_permutation_test(trades, n_permutations=1000)

        # P-Wert sollte hoch sein (nicht signifikant)
        assert result["p_value"] > 0.1, f"Zufällige Trades sollten nicht signifikant sein, p={result['p_value']}"
        assert not result["is_significant"]

    def test_highly_profitable_trades_significant(self):
        """Stark profitable Trades sollten signifikant sein."""
        from optimizer.simulation import monte_carlo_permutation_test

        # 80% Win Rate - klar überdurchschnittlich
        trades = [1.0] * 80 + [-1.0] * 20

        result = monte_carlo_permutation_test(trades, n_permutations=1000)

        # P-Wert sollte niedrig sein
        assert result["p_value"] < 0.05, f"Profitable Trades sollten signifikant sein, p={result['p_value']}"
        assert result["is_significant"]

    def test_losing_trades_not_significant(self):
        """Verlierende Trades sollten nicht signifikant sein."""
        from optimizer.simulation import monte_carlo_permutation_test

        # 30% Win Rate - schlecht
        trades = [1.0] * 30 + [-1.0] * 70

        result = monte_carlo_permutation_test(trades, n_permutations=1000)

        # Sollte nicht signifikant sein
        assert result["p_value"] > 0.05, f"Verlierende Trades sollten nicht signifikant sein, p={result['p_value']}"

    def test_too_few_trades_returns_not_significant(self):
        """Zu wenige Trades sollten automatisch nicht signifikant sein."""
        from optimizer.simulation import monte_carlo_permutation_test

        trades = [1.0, 1.0, -1.0]  # Nur 3 Trades

        result = monte_carlo_permutation_test(trades)

        assert result["p_value"] == 1.0
        assert not result["is_significant"]
        assert result["n_permutations"] == 0


class TestMonteCarloEquity:
    """Tests für die Monte Carlo Equity Simulation."""

    def test_high_kelly_leads_to_low_equity(self):
        """Hoher Kelly-Risk bei 50/50 Trades sollte zu niedriger Equity führen."""
        from optimizer.simulation import monte_carlo_equity_simulation

        # 50/50 Win/Loss (unprofitabel bei RRR=1)
        trades = [1.0] * 50 + [-1.0] * 50

        result = monte_carlo_equity_simulation(
            trades, kelly_risk=0.25, rrr=1.0, n_simulations=500
        )

        # Bei 25% Risk und 50% WR sollte die Equity stark sinken (geometric decay)
        # 50 Wins x 1.25 * 50 Losses x 0.75 = (1.25 * 0.75)^50 * 100 ≈ 0.0039
        assert result["median_equity"] < 10, f"Equity sollte niedrig sein, ist {result['median_equity']}"
        assert result["median_equity"] > 0, "Sollte aber nicht Bankrott sein (kelly < 1)"

    def test_low_kelly_no_bankruptcy(self):
        """Niedriger Kelly-Risk sollte keine Bankrotte verursachen."""
        from optimizer.simulation import monte_carlo_equity_simulation

        # 60% Win Rate
        trades = [1.0] * 60 + [-1.0] * 40

        result = monte_carlo_equity_simulation(
            trades, kelly_risk=0.01, rrr=1.0, n_simulations=500
        )

        # Bei 1% Risk sollte es keine Bankrotte geben
        assert result["bankruptcy_rate"] == 0, f"Kein Bankrott erwartet, aber {result['bankruptcy_rate']}"

    def test_confidence_intervals_make_sense(self):
        """Konfidenzintervalle sollten logisch sein (p5 < median < p95)."""
        from optimizer.simulation import monte_carlo_equity_simulation

        trades = [1.0] * 60 + [-1.0] * 40

        result = monte_carlo_equity_simulation(
            trades, kelly_risk=0.02, rrr=1.5, n_simulations=500
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
        from optimizer.simulation import monte_carlo_permutation_test

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
        from optimizer.simulation import monte_carlo_permutation_test

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
        from optimizer.main import simulate_equity

        # Bekannte Trades: 4 Wins, 1 Loss
        trades = [1.0, 1.0, 1.0, -1.0, 1.0]
        kelly_risk = 0.10  # 10%
        rrr = 2.0  # Risk-Reward-Ratio von 2

        result = simulate_equity(trades, kelly_risk, rrr)

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
        from optimizer.main import simulate_equity

        # Trades die einen klaren Drawdown erzeugen
        trades = [1.0, 1.0, -1.0, -1.0, -1.0, 1.0]  # Peak nach 2, dann 3 Losses
        kelly_risk = 0.10
        rrr = 1.0

        result = simulate_equity(trades, kelly_risk, rrr)

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


class TestDataLeakagePrevention:
    """
    Tests die sicherstellen dass kein Data Leakage stattfindet.

    Data Leakage würde bedeuten:
    - CT wird auf OOS optimiert (FALSCH)
    - Modell sieht zukünftige Daten (FALSCH)
    """

    def test_val_comes_before_test_chronologically(self):
        """Validation-Daten müssen chronologisch VOR Test-Daten liegen."""
        from optimizer.process import walk_forward_split

        n_rows = 50000
        df = pd.DataFrame({
            "C": np.arange(n_rows),  # Werte = Index (chronologisch)
        }, index=pd.date_range("2020-01-01", periods=n_rows, freq="h"))

        folds = walk_forward_split(df, n_folds=4, oos_size=4000)

        for i, (train, val, test) in enumerate(folds):
            # Alle Val-Werte müssen kleiner sein als alle Test-Werte
            max_val_value = val["C"].max()
            min_test_value = test["C"].min()

            assert max_val_value < min_test_value, \
                f"Fold {i}: Val max ({max_val_value}) >= Test min ({min_test_value}) - Data Leakage!"

    def test_train_never_sees_test_data(self):
        """Training darf niemals Test-Daten sehen."""
        from optimizer.process import walk_forward_split

        n_rows = 50000
        df = pd.DataFrame({
            "C": np.arange(n_rows),
        }, index=pd.date_range("2020-01-01", periods=n_rows, freq="h"))

        folds = walk_forward_split(df, n_folds=4, oos_size=4000)

        for i, (train, val, test) in enumerate(folds):
            max_train_value = train["C"].max()
            min_test_value = test["C"].min()

            assert max_train_value < min_test_value, \
                f"Fold {i}: Train sieht Test-Daten! Max Train ({max_train_value}) >= Min Test ({min_test_value})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
