"""
Tests für den PnL-basierten Monte Carlo Signifikanz-Test.

Sichert das korrekte Verhalten nach der Umstellung von binären 1/-1 Werten
auf tatsächliche PnL-Verteilungen (Sign-Permutation-Test).

Wichtigste getestete Eigenschaften:
- Breakeven-Trades (pnl=0) zählen nicht als Verlust
- Asymmetrische Verteilungen werden korrekt bewertet
- Hohe RRR mit niedriger WR kann signifikant sein, wenn Gesamtpnl positiv
- Negative Gesamtpnl → nicht signifikant, unabhängig von WR
- Der rrr-Parameter ist obsolet und wird ignoriert
"""
import pytest
import numpy as np

from fwbg.simulation.trade import monte_carlo_permutation_test


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def make_trades(wins, losses, win_size=100.0, loss_size=50.0, breakevens=0):
    """Erstellt eine Liste von PnL-Werten mit gegebener Zusammensetzung."""
    return [win_size] * wins + [-loss_size] * losses + [0.0] * breakevens


# ---------------------------------------------------------------------------
# Grundverhalten: Signifikant vs. Nicht Signifikant
# ---------------------------------------------------------------------------

class TestBasicSignificance:

    def test_clearly_profitable_strategy_is_significant(self):
        """Starker Edge (viele kleine Verluste, wenige große Gewinne mit positivem Gesamt-PnL)
        muss signifikant sein."""
        # 40% WR, aber Gewinne 3x so groß wie Verluste → stark positiver PnL
        pnls = make_trades(wins=40, losses=60, win_size=300.0, loss_size=100.0)
        result = monte_carlo_permutation_test(pnls, n_permutations=1000, random_seed=42)

        assert result["is_significant"], (
            f"Profitabler Edge (PnL={result['observed_pnl']:.0f}) muss signifikant sein, "
            f"aber p={result['p_value']:.3f}"
        )
        assert result["p_value"] < 0.05

    def test_losing_strategy_is_not_significant(self):
        """Strategie mit negativem Gesamt-PnL muss abgelehnt werden."""
        # 30% WR, Gewinne kleiner als Verluste → negativer PnL
        pnls = make_trades(wins=30, losses=70, win_size=80.0, loss_size=120.0)
        result = monte_carlo_permutation_test(pnls, n_permutations=1000, random_seed=42)

        assert not result["is_significant"], (
            f"Verlustbringende Strategie (PnL={result['observed_pnl']:.0f}) "
            f"darf nicht signifikant sein, aber p={result['p_value']:.3f}"
        )
        assert result["p_value"] > 0.05

    def test_zero_pnl_strategy_is_not_significant(self):
        """Breakeven-Strategie (PnL=0) ist nicht signifikant."""
        # Gleiche absolute Werte, aber perfekt ausgewogen
        pnls = [100.0, -100.0] * 50
        result = monte_carlo_permutation_test(pnls, n_permutations=1000, random_seed=42)

        assert not result["is_significant"]
        assert result["observed_pnl"] == pytest.approx(0.0)

    def test_fewer_than_10_trades_returns_not_significant(self):
        """Bei weniger als 10 Trades ist kein sinnvoller Test möglich."""
        pnls = [100.0, 200.0, -50.0, 150.0]
        result = monte_carlo_permutation_test(pnls, n_permutations=100, random_seed=42)

        assert not result["is_significant"]
        assert result["p_value"] == 1.0
        assert result["n_permutations"] == 0


# ---------------------------------------------------------------------------
# Breakeven-Trades (kritisch für atr_trailing)
# ---------------------------------------------------------------------------

class TestBreakevenTrades:

    def test_breakeven_trades_do_not_count_as_losses(self):
        """Breakeven-Trades (pnl=0) dürfen nicht als Verluste gezählt werden.

        Das ist der Kernbug der alten Implementierung: atr_trailing Breakeven-Closes
        hatten result=-1 aber pnl_raw=0, und wurden fälschlicherweise als Verluste
        in den MC-Test eingebracht.
        """
        # 50 Trades: 20 Gewinne (200 je), 20 Verluste (-100 je), 10 Breakeven (0)
        # Gesamt-PnL = 20*200 - 20*100 + 0 = 2000 → sollte signifikant sein
        pnls_with_be = make_trades(wins=20, losses=20, win_size=200.0, loss_size=100.0, breakevens=10)
        result_with_be = monte_carlo_permutation_test(pnls_with_be, n_permutations=1000, random_seed=42)

        # Ohne Breakeven-Trades: gleicher Edge
        pnls_without_be = make_trades(wins=20, losses=20, win_size=200.0, loss_size=100.0)
        result_without_be = monte_carlo_permutation_test(pnls_without_be, n_permutations=1000, random_seed=42)

        # Beide müssen signifikant sein
        assert result_with_be["is_significant"], (
            f"Strategie mit Breakeven-Trades fälschlicherweise abgelehnt "
            f"(p={result_with_be['p_value']:.3f})"
        )
        assert result_without_be["is_significant"]

    def test_only_breakeven_trades_not_significant(self):
        """Nur Breakeven-Trades → kein Edge, nicht signifikant."""
        pnls = [0.0] * 50
        result = monte_carlo_permutation_test(pnls, n_permutations=500, random_seed=42)

        assert not result["is_significant"]
        assert result["observed_pnl"] == pytest.approx(0.0)

    def test_breakeven_doesnt_inflate_win_rate(self):
        """Breakeven-Trades dürfen die observed_win_rate nicht aufblähen."""
        # 20 Gewinne, 20 Verluste, 60 Breakeven → WR = 20/100 = 20%
        pnls = [100.0] * 20 + [-100.0] * 20 + [0.0] * 60
        result = monte_carlo_permutation_test(pnls, n_permutations=100, random_seed=42)

        assert result["observed_win_rate"] == pytest.approx(0.20, abs=0.01)


# ---------------------------------------------------------------------------
# Asymmetrische Verteilungen (hohe RRR / niedriger RRR)
# ---------------------------------------------------------------------------

class TestAsymmetricDistributions:

    def test_high_rrr_low_winrate_significant(self):
        """Strategie mit 30% WR aber 4:1 RRR hat positiven Edge und muss signifikant sein.

        Das war ein Kernproblem des alten binary-Tests: bei breakeven_wr=0.2 und
        observed_wr=0.30 war der Test korrekt, aber die tatsächliche PnL-Verteilung
        ist robuster als der WR-basierte Test für edge cases.
        """
        # 30% WR, Gewinne 4x Verluste → E[PnL] = 0.3*400 - 0.7*100 = 50 (positiv)
        pnls = make_trades(wins=60, losses=140, win_size=400.0, loss_size=100.0)
        result = monte_carlo_permutation_test(pnls, n_permutations=1000, random_seed=42)

        assert result["observed_pnl"] > 0, "Positiver PnL erwartet"
        assert result["is_significant"], (
            f"Hohe-RRR-Strategie (30% WR, 4:1) muss signifikant sein, "
            f"aber p={result['p_value']:.3f}"
        )

    def test_high_winrate_negative_pnl_not_significant(self):
        """Hohe WR aber kleine Gewinne und große Verluste → negativer PnL, nicht signifikant.

        Ohne PnL-Verteilung würde ein rein WR-basierter Test diese Strategie
        fälschlicherweise als gut einschätzen.
        """
        # 60% WR aber Gewinne nur 50, Verluste 150 → E[PnL] = 0.6*50 - 0.4*150 = -30
        pnls = make_trades(wins=60, losses=40, win_size=50.0, loss_size=150.0)
        result = monte_carlo_permutation_test(pnls, n_permutations=1000, random_seed=42)

        assert result["observed_pnl"] < 0, "Negativer PnL erwartet"
        assert not result["is_significant"], (
            f"Strategie mit negativem PnL trotz hoher WR darf nicht signifikant sein, "
            f"aber p={result['p_value']:.3f}"
        )

    def test_many_small_losses_few_large_wins(self):
        """Typisches ORB-Profil: viele kleine Verluste, wenige große Gewinne → signifikant."""
        # 25% WR, Gewinne 5x Verluste (typisch für ORB mit RRR=5)
        pnls = make_trades(wins=25, losses=75, win_size=500.0, loss_size=100.0)
        result = monte_carlo_permutation_test(pnls, n_permutations=1000, random_seed=42)

        assert result["observed_pnl"] > 0
        assert result["is_significant"]


# ---------------------------------------------------------------------------
# Rückgabe-Interface
# ---------------------------------------------------------------------------

class TestResultInterface:

    def test_result_contains_required_keys(self):
        """Alle erwarteten Keys müssen im Ergebnis vorhanden sein."""
        pnls = make_trades(wins=40, losses=60, win_size=200.0, loss_size=100.0)
        result = monte_carlo_permutation_test(pnls, n_permutations=100, random_seed=42)

        required_keys = {
            "p_value", "observed_pnl", "observed_mean_pnl", "observed_win_rate",
            "mean_random_pnl", "std_random_pnl", "percentile", "is_significant",
            "n_permutations",
        }
        assert required_keys.issubset(result.keys()), (
            f"Fehlende Keys: {required_keys - result.keys()}"
        )

    def test_p_value_between_0_and_1(self):
        """p-Wert muss immer zwischen 0 und 1 liegen."""
        for pnls in [
            make_trades(10, 90, 300, 100),
            make_trades(50, 50, 100, 100),
            make_trades(90, 10, 100, 300),
        ]:
            result = monte_carlo_permutation_test(pnls, n_permutations=200, random_seed=42)
            assert 0.0 <= result["p_value"] <= 1.0

    def test_percentile_consistent_with_p_value(self):
        """Percentile und p-value müssen konsistent sein: percentile ≈ 100*(1-p_value)."""
        pnls = make_trades(wins=50, losses=50, win_size=200.0, loss_size=100.0)
        result = monte_carlo_permutation_test(pnls, n_permutations=1000, random_seed=42)

        # percentile + p_value ≈ 1 (mit kleiner Toleranz durch Diskretisierung)
        assert result["percentile"] == pytest.approx(
            100 * (1 - result["p_value"]), abs=2.0
        )

    def test_reproducible_with_same_seed(self):
        """Gleiches Ergebnis bei gleichem Seed."""
        pnls = make_trades(wins=35, losses=65, win_size=250.0, loss_size=100.0)
        r1 = monte_carlo_permutation_test(pnls, n_permutations=500, random_seed=7)
        r2 = monte_carlo_permutation_test(pnls, n_permutations=500, random_seed=7)

        assert r1["p_value"] == r2["p_value"]
        assert r1["observed_pnl"] == r2["observed_pnl"]

    def test_different_seeds_may_differ(self):
        """Verschiedene Seeds können unterschiedliche p-Werte liefern (stochastisch)."""
        pnls = make_trades(wins=35, losses=65, win_size=200.0, loss_size=100.0)
        r1 = monte_carlo_permutation_test(pnls, n_permutations=100, random_seed=1)
        r2 = monte_carlo_permutation_test(pnls, n_permutations=100, random_seed=999)

        # Beide Ergebnisse müssen gültig sein (kein Absturz)
        assert 0 <= r1["p_value"] <= 1
        assert 0 <= r2["p_value"] <= 1

    def test_rrr_parameter_ignored_for_backward_compat(self):
        """Der rrr-Parameter wird akzeptiert aber ignoriert (Rückwärtskompatibilität)."""
        pnls = make_trades(wins=40, losses=60, win_size=200.0, loss_size=100.0)
        result_no_rrr = monte_carlo_permutation_test(pnls, n_permutations=500, random_seed=42)
        result_with_rrr = monte_carlo_permutation_test(pnls, n_permutations=500, random_seed=42, rrr=3.0)

        assert result_no_rrr["p_value"] == result_with_rrr["p_value"]


# ---------------------------------------------------------------------------
# atr_trailing spezifische Szenarien
# ---------------------------------------------------------------------------

class TestAtrTrailingScenarios:

    def test_atr_trailing_breakeven_scenario(self):
        """Realistisches atr_trailing Szenario: Breakeven-Closes zählen nicht als Verlust.

        Bei atr_trailing: Preis erreicht 50% des TP, kehrt zurück → Close bei Entry.
        pnl_raw ≈ 0 (minus Spread). Alter Code: result=-1 → fälschlicherweise Verlust.
        Neuer Code: pnl_raw=0 → neutraler Beitrag.
        """
        # Simuliert typisches atr_trailing Portfolio:
        # - 25% echte TP-Treffer (große Gewinne)
        # - 35% Breakeven-Closes (0 PnL)
        # - 40% echte SL-Treffer (Verluste)
        wins = [350.0] * 25
        breakevens = [0.0] * 35
        losses = [-100.0] * 40
        pnls = wins + breakevens + losses

        result = monte_carlo_permutation_test(pnls, n_permutations=1000, random_seed=42)

        # Gesamt-PnL = 25*350 + 0 - 40*100 = 8750 - 4000 = 4750 (positiv)
        assert result["observed_pnl"] == pytest.approx(4750.0)
        assert result["is_significant"], (
            f"atr_trailing mit echtem Edge muss signifikant sein, "
            f"aber p={result['p_value']:.3f}"
        )

    def test_atr_trailing_breakeven_as_loss_would_fail(self):
        """Dokumentiert warum der alte binary Test falsch war.

        Mit dem alten Test: Breakevens zählen als -1 → WR sinkt deutlich,
        test schlägt fehl oder ist nicht signifikant. Mit echten PnL-Werten
        ist die Strategie korrekt als signifikant erkannt.

        Dieser Test stellt sicher dass der neue Test das korrekte Ergebnis liefert.
        """
        # Gleiche Trades wie test_atr_trailing_breakeven_scenario
        wins = [350.0] * 25
        breakevens = [0.0] * 35
        losses = [-100.0] * 40
        pnls = wins + breakevens + losses

        result = monte_carlo_permutation_test(pnls, n_permutations=1000, random_seed=42)

        # Mit echten PnL-Werten: Gesamt-PnL ist positiv → signifikant
        assert result["observed_pnl"] > 0
        assert result["is_significant"]

        # Zusätzlich: Der Test zählt WR korrekt (nur echte Gewinne)
        # 25 Gewinne / 100 Trades = 25% WR
        assert result["observed_win_rate"] == pytest.approx(0.25, abs=0.01)
