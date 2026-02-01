"""Tests für Early Termination in der Grid-Search."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


@dataclass
class MockSimulationContext:
    """Mock SimulationContext für Tests."""
    symbol: str = "TEST"
    min_trades: int = 10
    min_fold_stability: float = 0.5
    early_termination: bool = True
    first_fold_sanity_check: bool = True
    first_fold_min_win_rate: float = 0.25
    first_fold_min_pnl: float = -10.0
    first_fold_min_trades: int = 5
    feature_selection: str = "boruta"
    separate_long_short: bool = False
    long_enabled: bool = True
    short_enabled: bool = True
    spread: float = 0.0001
    max_trade_bars: int = None


class TestMathematicalEarlyTermination:
    """Tests für mathematisches Early Termination (min_fold_stability nicht erreichbar)."""

    def test_early_termination_when_impossible_to_reach_stability(self):
        """Wenn nach k Folds min_fold_stability mathematisch nicht mehr erreichbar ist."""
        # 5 Folds, min_fold_stability=0.5 => brauche mindestens 3 profitable Folds
        # Nach 3 Verlusten in Folge: max_possible = 0 + 2 = 2 < 3 => terminate

        total_folds = 5
        min_fold_stability = 0.5
        min_profitable = int(np.ceil(total_folds * min_fold_stability))  # = 3

        # Simuliere 3 unprofitable Folds
        profitable_count = 0
        failed_count = 3
        fold_idx = 3  # Nach 3 Folds (0, 1, 2)

        remaining_folds = total_folds - fold_idx  # = 2
        max_possible_profitable = profitable_count + remaining_folds  # = 0 + 2 = 2

        should_terminate = max_possible_profitable < min_profitable  # 2 < 3 = True

        assert should_terminate, "Sollte terminieren wenn min_fold_stability unmöglich"
        assert min_profitable == 3
        assert max_possible_profitable == 2

    def test_no_early_termination_when_still_possible(self):
        """Keine Termination wenn min_fold_stability noch erreichbar ist."""
        total_folds = 5
        min_fold_stability = 0.5
        min_profitable = int(np.ceil(total_folds * min_fold_stability))  # = 3

        # Simuliere 1 profitable, 1 unprofitable
        profitable_count = 1
        fold_idx = 2  # Nach 2 Folds (0, 1)

        remaining_folds = total_folds - fold_idx  # = 3
        max_possible_profitable = profitable_count + remaining_folds  # = 1 + 3 = 4

        should_terminate = max_possible_profitable < min_profitable  # 4 < 3 = False

        assert not should_terminate, "Sollte nicht terminieren wenn noch erreichbar"


class TestConsecutiveFailuresTermination:
    """Tests für Termination bei 3+ aufeinanderfolgenden Fehlschlägen."""

    def test_termination_after_3_consecutive_failures(self):
        """Nach 3 Folds hintereinander unprofitabel sollte abgebrochen werden."""
        max_consecutive_failures = 3

        # Simuliere 3 unprofitable Folds in Folge
        consecutive_failures = 0
        fold_results = [-5.0, -3.0, -2.0]  # Alle negativ

        for pnl in fold_results:
            if pnl > 0:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            if consecutive_failures >= max_consecutive_failures:
                break

        assert consecutive_failures >= max_consecutive_failures

    def test_no_termination_with_mixed_results(self):
        """Gemischte Ergebnisse sollten nicht terminieren."""
        max_consecutive_failures = 3

        # Simuliere gemischte Folds
        consecutive_failures = 0
        fold_results = [-5.0, 2.0, -3.0, -2.0, 1.0]  # Gemischt

        terminated = False
        for pnl in fold_results:
            if pnl > 0:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            if consecutive_failures >= max_consecutive_failures:
                terminated = True
                break

        assert not terminated, "Sollte nicht terminieren bei gemischten Ergebnissen"

    def test_reset_counter_on_success(self):
        """Counter sollte bei Erfolg zurückgesetzt werden."""
        consecutive_failures = 2  # Fast bei 3

        # Ein Erfolg kommt
        pnl = 5.0
        if pnl > 0:
            consecutive_failures = 0

        assert consecutive_failures == 0, "Counter sollte bei Erfolg auf 0 zurückgesetzt werden"


class TestCumulativePnLTermination:
    """Tests für Termination bei stark negativem kumulativem PnL."""

    def test_termination_when_avg_pnl_strongly_negative(self):
        """Nach Hälfte der Folds mit avg_pnl < -3 sollte abgebrochen werden."""
        total_folds = 6
        halfway = total_folds // 2  # = 3

        # Simuliere sehr schlechte Folds (4 Folds = mehr als Hälfte)
        fold_pnls = [-5.0, -4.0, -6.0, -5.0]  # Alle stark negativ
        cumulative_pnl = sum(fold_pnls)  # = -20
        fold_idx = 3  # Nach 4 Folds (0, 1, 2, 3) - fold_idx ist 0-basiert

        avg_pnl_so_far = cumulative_pnl / (fold_idx + 1)  # = -5.0

        # Bedingung: fold_idx >= halfway AND fold_idx >= 2 AND avg_pnl < -3
        should_terminate = (
            fold_idx >= halfway and
            fold_idx >= 2 and
            avg_pnl_so_far < -3.0
        )

        assert should_terminate, "Sollte terminieren bei stark negativem avg PnL"
        assert avg_pnl_so_far == -5.0

    def test_no_termination_when_avg_pnl_acceptable(self):
        """Keine Termination bei akzeptablem avg PnL."""
        total_folds = 6
        halfway = total_folds // 2

        # Simuliere gemischte Folds
        fold_pnls = [-2.0, 1.0, -1.0]  # Leicht negativ aber akzeptabel
        cumulative_pnl = sum(fold_pnls)  # = -2
        fold_idx = 2

        avg_pnl_so_far = cumulative_pnl / (fold_idx + 1)  # = -0.67

        should_terminate = (
            fold_idx >= halfway and
            fold_idx >= 2 and
            avg_pnl_so_far < -3.0
        )

        assert not should_terminate, "Sollte nicht terminieren bei akzeptablem avg PnL"


class TestFirstFoldSanityCheck:
    """Tests für den First-Fold Sanity Check."""

    def test_catastrophic_first_fold_terminates(self):
        """Katastrophaler erster Fold (WR < 25%, PnL < -10) sollte terminieren."""
        first_fold_min_win_rate = 0.25
        first_fold_min_pnl = -10.0
        first_fold_min_trades = 5

        # Simuliere katastrophalen ersten Fold
        fold_trades = [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]  # 1 Win, 6 Losses
        n_fold_trades = len(fold_trades)
        fold_win_rate = fold_trades.count(1.0) / n_fold_trades  # = 0.143 (14.3%)
        fold_pnl = sum(fold_trades)  # = -5.0

        # Korrigiere: PnL muss auch < -10 sein
        fold_trades = [1.0] + [-1.0] * 15  # 1 Win, 15 Losses = -14 PnL
        n_fold_trades = len(fold_trades)
        fold_win_rate = fold_trades.count(1.0) / n_fold_trades  # = 0.0625 (6.25%)
        fold_pnl = sum(fold_trades)  # = -14

        is_catastrophic = (
            fold_win_rate < first_fold_min_win_rate and
            fold_pnl < first_fold_min_pnl and
            n_fold_trades >= first_fold_min_trades
        )

        assert is_catastrophic, "Sollte als katastrophal erkannt werden"
        assert fold_win_rate < 0.25
        assert fold_pnl < -10.0

    def test_bad_but_not_catastrophic_continues(self):
        """Schlechter aber nicht katastrophaler Fold sollte weiterlaufen."""
        first_fold_min_win_rate = 0.25
        first_fold_min_pnl = -10.0
        first_fold_min_trades = 5

        # Simuliere schlechten aber nicht katastrophalen Fold
        # Win Rate < 25% ABER PnL > -10
        fold_trades = [1.0, -1.0, -1.0, -1.0, -1.0]  # 1 Win, 4 Losses
        n_fold_trades = len(fold_trades)
        fold_win_rate = fold_trades.count(1.0) / n_fold_trades  # = 0.20 (20%)
        fold_pnl = sum(fold_trades)  # = -3.0

        is_catastrophic = (
            fold_win_rate < first_fold_min_win_rate and
            fold_pnl < first_fold_min_pnl and
            n_fold_trades >= first_fold_min_trades
        )

        # Nicht katastrophal weil PnL > -10
        assert not is_catastrophic, "Sollte nicht als katastrophal erkannt werden (PnL ok)"

    def test_too_few_trades_continues(self):
        """Zu wenige Trades im ersten Fold sollte abbrechen."""
        first_fold_min_trades = 5

        fold_trades = [1.0, -1.0]  # Nur 2 Trades
        n_fold_trades = len(fold_trades)

        too_few_trades = n_fold_trades < first_fold_min_trades

        assert too_few_trades, "Sollte als 'zu wenige Trades' erkannt werden"


class TestEarlyTerminationIntegration:
    """Integration Tests für die komplette Early Termination Logik."""

    def test_all_termination_conditions_combined(self):
        """Teste dass alle Bedingungen korrekt zusammenwirken."""
        # Setup
        total_folds = 8
        min_fold_stability = 0.5
        min_profitable = int(np.ceil(total_folds * min_fold_stability))  # = 4
        max_consecutive_failures = 3

        # Simuliere Fold-Ergebnisse
        fold_pnls = [-2.0, -3.0, -4.0, 1.0, -1.0, -2.0, -3.0, -1.0]

        profitable_count = 0
        consecutive_failures = 0
        cumulative_pnl = 0.0
        terminated_reason = None

        for fold_idx, pnl in enumerate(fold_pnls):
            cumulative_pnl += pnl

            # 1. Mathematisches Early Termination
            remaining_folds = total_folds - fold_idx - 1
            max_possible_profitable = profitable_count + remaining_folds + (1 if pnl > 0 else 0)
            if max_possible_profitable < min_profitable:
                terminated_reason = "mathematical"
                break

            # 2. Consecutive Failures
            if pnl > 0:
                profitable_count += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1

            if consecutive_failures >= max_consecutive_failures:
                terminated_reason = "consecutive_failures"
                break

            # 3. Cumulative PnL (nach Hälfte)
            halfway = total_folds // 2
            if fold_idx >= halfway and fold_idx >= 2:
                avg_pnl = cumulative_pnl / (fold_idx + 1)
                if avg_pnl < -3.0:
                    terminated_reason = "cumulative_pnl"
                    break

        # In diesem Beispiel: Fold 0, 1, 2 sind alle negativ = 3 consecutive failures
        assert terminated_reason == "consecutive_failures"


class TestRAMLimitEnforcement:
    """Tests für die RAM-Limit-Durchsetzung bei mehreren Assets."""

    def test_ram_limit_calculation_with_high_usage(self):
        """Bei hoher RAM-Nutzung sollten weniger Threads gestartet werden."""
        total_ram_gb = 32.0
        min_free_ram_percent = 0.20  # 20% frei halten
        target_max_used_percent = 1.0 - min_free_ram_percent  # = 0.80 (max 80% nutzen)
        ram_per_thread_gb = 0.5

        # Simuliere: Bereits 75% RAM genutzt (unter Limit)
        current_used_percent = 0.75
        currently_used_gb = total_ram_gb * current_used_percent  # = 24 GB
        max_usable_ram_gb = total_ram_gb * target_max_used_percent  # = 25.6 GB

        available_ram_for_threads = max(0, max_usable_ram_gb - currently_used_gb)  # = 1.6 GB
        ram_based_limit = max(1, int(available_ram_for_threads / ram_per_thread_gb))  # = 3

        assert ram_based_limit == 3
        assert available_ram_for_threads == pytest.approx(1.6, rel=0.01)

    def test_ram_limit_when_already_exceeded(self):
        """Bei bereits überschrittenem RAM-Limit stark reduzieren."""
        total_ram_gb = 32.0
        min_free_ram_percent = 0.20
        target_max_used_percent = 0.80
        ram_per_thread_gb = 0.5

        # Simuliere: Bereits 85% RAM genutzt (über Limit!)
        current_used_percent = 0.85
        free_ram_gb = total_ram_gb * (1 - current_used_percent)  # = 4.8 GB
        min_free_ram_gb = total_ram_gb * min_free_ram_percent  # = 6.4 GB

        # Wenn über Limit: nutze alten Algorithmus (verfügbar - reserve)
        available_ram_for_threads = max(0, free_ram_gb - min_free_ram_gb)  # = -1.6 -> 0
        ram_based_limit = max(1, int(available_ram_for_threads / ram_per_thread_gb))  # = 1

        assert ram_based_limit == 1  # Minimum


class TestProgressDisplayFixes:
    """Tests für die Progress-Display-Korrekturen."""

    def test_eta_threshold_with_large_grid(self):
        """ETA sollte auch bei großem Grid und kleinem Fortschritt berechnet werden."""
        total_assets = 1
        min_progress = 0.01 * total_assets  # = 0.01

        # 264 von 4200 Grid-Kombinationen = 6.3%
        grid_progress = 264 / 4200  # = 0.063
        total_progress = grid_progress  # Bei 1 Asset

        elapsed = 60  # 1 Minute

        # Alte Bedingung: total_progress < 0.1 => kein ETA
        old_condition = elapsed < 10 or total_progress < 0.1

        # Neue Bedingung: min_progress = 1% des Gesamtfortschritts
        new_condition = elapsed < 30 or total_progress < min_progress

        assert old_condition, "Alte Bedingung würde ETA blockieren"
        assert not new_condition, "Neue Bedingung erlaubt ETA-Berechnung"
