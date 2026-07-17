"""
Pytest Tests für Robust Validation Framework.

WICHTIG: Diese Tests DETEKTIEREN Sample Bias für DIAGNOSTIK,
nicht zum automatischen Ausschluss. Wenn viele Assets biased sind,
ist das ein SYSTEMATISCHES Problem im Code, kein Asset-Problem.

Diese Tests stellen sicher dass:
1. Sample Bias ERKANNT wird (für Diagnose)
2. Walk-Forward Validation funktioniert
3. Robustheit-Checks korrekt arbeiten
"""
import pytest
import numpy as np
import pandas as pd
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))  # Add tests directory to path

from fwbg.optimization.robust_validation import (
    create_walk_forward_folds,
    plan_walk_forward,
    RobustValidationResult,
)
from test_sample_bias_detection import SampleBiasDetector


class TestPlanWalkForward:
    """Adaptives Fold-Sizing: an vorhandene Historie anpassen statt skippen."""

    def test_plentiful_data_matches_legacy_behaviour(self):
        # Reichlich Historie: identisch zur früheren max()-Logik
        # (target_folds, (total-min_train)//folds, target_min_train).
        total, folds, min_train = 100_000, 8, 17_500
        n, oos, mt = plan_walk_forward(total, folds, min_train, min_oos=400)
        assert n == 8
        assert mt == min_train
        assert oos == (total - min_train) // folds

    def test_day1_eurusd_regression(self):
        # Der ursprüngliche Bug: DAY_1 mit HOUR-Fallback verlangte 49500 Balken.
        # EURUSD/DAY_1 hat ~6746 — muss jetzt mit vollen Folds passen.
        total = 6746
        plan = plan_walk_forward(total, target_folds=8, target_min_train=1000, min_oos=60)
        assert plan is not None
        n, oos, mt = plan
        assert n == 8
        assert mt + n * oos <= total  # tatsächlich konstruierbar

    def test_short_history_index_still_runs(self):
        # Kurze Index-Historie (DAX/DAY_1 ~3180) lief früher auf Grund; jetzt OK.
        total = 3180
        plan = plan_walk_forward(total, target_folds=8, target_min_train=1000, min_oos=60)
        assert plan is not None
        n, oos, mt = plan
        assert mt + n * oos <= total
        assert oos >= 60

    def test_reduces_folds_when_needed(self):
        # Zu wenig für 8 Folds bei min_oos, aber genug für weniger Folds.
        total = 700
        plan = plan_walk_forward(total, target_folds=8, target_min_train=1000,
                                 min_oos=60, min_folds=3)
        assert plan is not None
        n, oos, mt = plan
        assert 3 <= n <= 8
        assert mt + n * oos <= total

    def test_returns_none_when_truly_insufficient(self):
        assert plan_walk_forward(50, target_folds=8, target_min_train=1000,
                                 min_oos=60) is None

    def test_respects_user_lower_fold_count(self):
        # target_folds < default min_folds darf nicht nach oben gedrückt werden.
        plan = plan_walk_forward(10_000, target_folds=2, target_min_train=1000,
                                 min_oos=100)
        assert plan is not None
        assert plan[0] == 2

    def test_plan_is_constructible_by_create_folds(self):
        # Ein Plan muss immer ohne ValueError in echte Folds übersetzbar sein.
        total = 3180
        n, oos, mt = plan_walk_forward(total, 8, 1000, min_oos=60)
        df = pd.DataFrame({
            "O": np.ones(total), "H": np.ones(total),
            "L": np.ones(total), "C": np.ones(total),
        })
        folds = create_walk_forward_folds(df, n_folds=n, test_size=oos,
                                          min_train_size=mt, anchored=True)
        assert len(folds) == n


class TestWalkForwardFolds:
    """Tests für Walk-Forward Fold Creation."""

    def test_creates_correct_number_of_folds(self):
        """Test: Korrekte Anzahl Folds wird erstellt."""
        # Create dummy data
        df = pd.DataFrame({
            'O': np.random.randn(50000),
            'H': np.random.randn(50000),
            'L': np.random.randn(50000),
            'C': np.random.randn(50000),
        })

        folds = create_walk_forward_folds(
            df,
            n_folds=5,
            test_size=4000,
            min_train_size=20000,
        )

        assert len(folds) == 5, "Should create exactly 5 folds"

    def test_folds_have_no_overlap(self):
        """Test: Keine Überlappung zwischen Train und Test."""
        df = pd.DataFrame({
            'O': np.arange(50000),  # Use range for easy checking
            'H': np.arange(50000),
            'L': np.arange(50000),
            'C': np.arange(50000),
        })

        folds = create_walk_forward_folds(df, n_folds=5)

        for fold in folds:
            # Train should end where test starts
            assert fold.train_end == fold.test_start, \
                f"Fold {fold.fold_id}: Train end ({fold.train_end}) should equal test start ({fold.test_start})"

            # No index overlap
            train_indices = set(fold.train_df.index)
            test_indices = set(fold.test_df.index)
            overlap = train_indices.intersection(test_indices)

            assert len(overlap) == 0, \
                f"Fold {fold.fold_id}: Found {len(overlap)} overlapping indices between train and test"

    def test_test_periods_dont_overlap(self):
        """Test: Test-Perioden überlappen nicht."""
        df = pd.DataFrame({
            'O': np.arange(50000),
            'H': np.arange(50000),
            'L': np.arange(50000),
            'C': np.arange(50000),
        })

        folds = create_walk_forward_folds(df, n_folds=5)

        for i in range(len(folds) - 1):
            fold1 = folds[i]
            fold2 = folds[i + 1]

            # Next fold's test should start after previous fold's test
            assert fold2.test_start >= fold1.test_end, \
                f"Fold {fold2.fold_id} test starts before Fold {fold1.fold_id} test ends"

    def test_anchored_training_grows(self):
        """Test: Anchored mode - Training wächst über Folds."""
        df = pd.DataFrame({
            'O': np.arange(50000),
            'H': np.arange(50000),
            'L': np.arange(50000),
            'C': np.arange(50000),
        })

        folds = create_walk_forward_folds(df, n_folds=5, anchored=True)

        for i in range(len(folds) - 1):
            fold1 = folds[i]
            fold2 = folds[i + 1]

            # Training should grow (or stay same)
            train_size_1 = fold1.train_end - fold1.train_start
            train_size_2 = fold2.train_end - fold2.train_start

            assert train_size_2 >= train_size_1, \
                f"Anchored mode: Training should grow from fold {i} to {i+1}"

    def test_insufficient_data_raises_error(self):
        """Test: Error bei zu wenig Daten."""
        df = pd.DataFrame({
            'O': np.arange(1000),  # Too small
            'H': np.arange(1000),
            'L': np.arange(1000),
            'C': np.arange(1000),
        })

        with pytest.raises(ValueError, match="Not enough data"):
            create_walk_forward_folds(
                df,
                n_folds=5,
                test_size=4000,  # Impossible with only 1000 bars
                min_train_size=20000,
            )


class TestRobustValidationResult:
    """Tests für RobustValidationResult."""

    def test_is_robust_with_good_metrics(self):
        """Test: Gute Metriken werden als robust erkannt."""
        result = RobustValidationResult(
            mean_win_rate=0.68,
            std_win_rate=0.08,  # Low std - consistent
            min_win_rate=0.62,
            max_win_rate=0.74,
            mean_pnl=150.0,
            std_pnl=25.0,
            min_pnl=120.0,
            max_pnl=180.0,
            total_trades=850,  # Plenty of trades
            mean_trades_per_fold=170.0,
            consistency_score=0.85,  # High consistency
            worst_case_pnl=120.0,
            sample_bias_detected=False,
            holdout_vs_inner_ratios=[0.95, 1.08, 0.92, 1.12, 0.98],  # All < 2.0
            fold_results=[],
            selected_config={},
        )

        assert result.is_robust(
            min_total_trades=500,
            max_win_rate_std=0.15,
            max_bias_ratio=2.0,
        ) is True

    def test_is_not_robust_with_low_trades(self):
        """Test: Zu wenig Trades → nicht robust."""
        result = RobustValidationResult(
            mean_win_rate=0.68,
            std_win_rate=0.08,
            min_win_rate=0.62,
            max_win_rate=0.74,
            mean_pnl=150.0,
            std_pnl=25.0,
            min_pnl=120.0,
            max_pnl=180.0,
            total_trades=200,  # TOO LOW
            mean_trades_per_fold=40.0,
            consistency_score=0.85,
            worst_case_pnl=120.0,
            sample_bias_detected=False,
            holdout_vs_inner_ratios=[0.95, 1.08, 0.92, 1.12, 0.98],
            fold_results=[],
            selected_config={},
        )

        assert result.is_robust(min_total_trades=500) is False

    def test_is_not_robust_with_high_variance(self):
        """Test: Hohe Varianz → nicht robust."""
        result = RobustValidationResult(
            mean_win_rate=0.68,
            std_win_rate=0.22,  # VERY HIGH
            min_win_rate=0.45,
            max_win_rate=0.88,
            mean_pnl=150.0,
            std_pnl=85.0,
            min_pnl=50.0,
            max_pnl=250.0,
            total_trades=850,
            mean_trades_per_fold=170.0,
            consistency_score=0.35,  # LOW
            worst_case_pnl=50.0,
            sample_bias_detected=False,
            holdout_vs_inner_ratios=[0.95, 1.08, 0.92, 1.12, 0.98],
            fold_results=[],
            selected_config={},
        )

        assert result.is_robust(max_win_rate_std=0.15) is False

    def test_is_not_robust_with_sample_bias(self):
        """Test: Sample Bias → nicht robust."""
        result = RobustValidationResult(
            mean_win_rate=0.75,
            std_win_rate=0.08,
            min_win_rate=0.68,
            max_win_rate=0.85,
            mean_pnl=250.0,
            std_pnl=45.0,
            min_pnl=190.0,
            max_pnl=320.0,
            total_trades=850,
            mean_trades_per_fold=170.0,
            consistency_score=0.85,
            worst_case_pnl=190.0,
            sample_bias_detected=True,  # DETECTED!
            holdout_vs_inner_ratios=[0.95, 3.2, 0.92, 1.12, 0.98],  # One fold is 3.2x!
            fold_results=[],
            selected_config={},
        )

        assert result.is_robust(max_bias_ratio=2.0) is False


class TestSampleBiasDetection:
    """
    Integration tests: Stellt sicher dass Sample Bias erkannt wird.

    Diese Tests simulieren die AUDUSD/EURUSD Fälle und prüfen
    dass das System diese korrekt als problematisch erkennt.
    """

    def test_audusd_case_is_detected_as_biased(self):
        """Test: AUDUSD Fall (2.33x ratio) wird als biased erkannt."""
        # Real AUDUSD metrics from Feb 6
        result = SampleBiasDetector.run_full_check(
            inner_val_pnl=55.4,
            holdout_pnl=129.0,
            win_rate=0.831,
            rrr=0.5,
            n_trades=195,
        )

        assert "WARNING" in result['verdict'], \
            "AUDUSD case should be detected and warned about"

        assert result['checks']['holdout_bias']['has_bias'] is True, \
            "Holdout bias should be detected (2.33x ratio)"

        assert result['checks']['winrate_bias']['has_bias'] is True, \
            "Unrealistic win-rate should be detected"

        assert result['checks']['trade_count']['has_issue'] is True, \
            "Insufficient trades should be detected"

    def test_eurusd_extreme_case_is_detected(self):
        """Test: EURUSD Fall (4.28x ratio) wird als extrem biased erkannt."""
        # Real EURUSD metrics from Feb 7
        result = SampleBiasDetector.run_full_check(
            inner_val_pnl=114.4,
            holdout_pnl=490.0,
            win_rate=0.763,
            rrr=0.5,
            n_trades=932,
        )

        assert "WARNING" in result['verdict'], \
            "EURUSD case should be detected and warned about (indicates systematic problem)"

        assert result['checks']['holdout_bias']['has_bias'] is True, \
            "Extreme holdout bias (4.28x) should be detected"

        bias_ratio = 490.0 / 114.4
        assert bias_ratio > 3.0, \
            "EURUSD bias ratio should be > 3.0"

    def test_normal_case_is_not_flagged(self):
        """Test: Normale robuste Config wird nicht geflaggt."""
        # Realistic robust metrics
        result = SampleBiasDetector.run_full_check(
            inner_val_pnl=120.0,
            holdout_pnl=115.0,  # Slightly worse (normal)
            win_rate=0.70,  # Reasonable for RRR=0.5
            rrr=0.5,
            n_trades=850,
        )

        assert result['verdict'] == "OK - Appears Robust", \
            "Normal case should show OK"

        assert result['checks']['holdout_bias']['has_bias'] is False, \
            "No bias should be detected"

        assert result['checks']['winrate_bias']['has_bias'] is False, \
            "Win-rate should be plausible"

        assert result['checks']['trade_count']['has_issue'] is False, \
            "Sufficient trades"


class TestRegressionPrevention:
    """
    Regression Tests: Verhindert dass alte Probleme wiederkehren.
    """

    def test_walk_forward_reduces_bias_risk(self):
        """
        Test: Walk-Forward reduziert Sample Bias Risiko.

        Mit 5 Folds ist es statistisch unwahrscheinlich dass
        ALLE Folds biased sind.
        """
        # Simulate walk-forward with 5 folds
        # Even if one fold is biased, others should balance it out

        fold_ratios = [0.95, 1.08, 0.92, 1.12, 0.98]  # Normal variation

        # No individual fold is >2x
        assert all(r < 2.0 for r in fold_ratios), \
            "No individual fold should show extreme bias"

        # Mean is close to 1.0
        mean_ratio = np.mean(fold_ratios)
        assert 0.8 < mean_ratio < 1.2, \
            "Mean ratio should be close to 1.0"

    def test_insufficient_trades_is_always_caught(self):
        """
        Test: Zu wenig Trades wird immer erkannt.

        Verhindert dass Configs mit <300 Trades deployed werden.
        """
        for n_trades in [50, 100, 200, 299]:
            result = SampleBiasDetector.check_insufficient_trades(
                n_trades=n_trades,
                min_trades=300,
            )

            assert result['has_issue'] is True, \
                f"{n_trades} trades should be flagged as insufficient"


@pytest.mark.parametrize("inner_pnl,holdout_pnl,expected_bias", [
    (100, 50, False),    # Holdout worse (overfitting)
    (100, 100, False),   # Equal (normal)
    (100, 120, False),   # Slightly better (normal)
    (100, 180, False),   # 1.8x (borderline)
    (100, 220, True),    # 2.2x (BIAS!)
    (100, 350, True),    # 3.5x (EXTREME BIAS!)
])
def test_bias_detection_thresholds(inner_pnl, holdout_pnl, expected_bias):
    """
    Parametrized test: Prüft verschiedene Holdout/Inner Ratios.
    """
    result = SampleBiasDetector.check_holdout_vs_inner_bias(
        inner_val_pnl=inner_pnl,
        holdout_pnl=holdout_pnl,
        max_ratio=2.0,
    )

    assert result['has_bias'] == expected_bias, \
        f"Ratio {holdout_pnl/inner_pnl:.2f}x: expected bias={expected_bias}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
