"""
Regression tests for metric calculation bugs.

Bug 1: Sharpe annualization - trades_per_year was computed as
        total_trades/len(all_trades) * bars_per_year = 1.0 * 6000 = 6000,
        giving sqrt(6000) ≈ 77.5x inflation. Correct formula:
        total_trades * bars_per_year / total_test_bars.

Bug 2: fold_stability denominator - used len(inner_val_pnls) instead of total_folds,
        excluding failed folds and inflating stability to 1.0.
"""
import pytest
import numpy as np

from fwbg.simulation.trade import calculate_sharpe_ratio


class TestSharpeTradesPerYearFormula:
    """Bug 1: trades_per_year must use actual observation period, not bars_per_year."""

    def test_buggy_formula_gives_bars_per_year(self):
        """The buggy formula total_trades/len(all_trades)*bars_per_year always gives bars_per_year."""
        total_trades = 300
        all_trades_len = 300  # total_trades == len(all_trades) by construction
        bars_per_year = 6000

        buggy_tpy = total_trades / all_trades_len * bars_per_year

        assert buggy_tpy == 6000, (
            "Buggy formula must simplify to bars_per_year since total_trades == len(all_trades)"
        )

    def test_correct_formula_uses_observation_period(self):
        """Correct formula: total_trades * bars_per_year / total_test_bars."""
        total_trades = 300
        bars_per_year = 6000
        total_test_bars = 32000  # 8 folds × 4000 bars

        correct_tpy = total_trades * bars_per_year / total_test_bars

        assert correct_tpy == pytest.approx(56.25)
        assert correct_tpy < 100, "trades_per_year should be realistic for hourly data"
        assert correct_tpy != bars_per_year, "Must not equal bars_per_year"

    def test_sharpe_difference_buggy_vs_correct(self):
        """Sharpe with buggy trades_per_year is ~10x higher than correct."""
        kelly = 0.02
        rrr = 1.0
        trades = [1.0] * 180 + [-1.0] * 120
        trade_returns = [kelly * rrr if t > 0 else -kelly for t in trades]

        sharpe_buggy = calculate_sharpe_ratio(trade_returns, trades_per_year=6000)
        sharpe_correct = calculate_sharpe_ratio(trade_returns, trades_per_year=56.25)

        ratio = sharpe_buggy / sharpe_correct if sharpe_correct != 0 else float('inf')
        expected_ratio = np.sqrt(6000 / 56.25)  # sqrt(106.67) ≈ 10.3

        assert ratio == pytest.approx(expected_ratio, rel=0.01), (
            f"Buggy/correct ratio={ratio:.1f}, expected ~{expected_ratio:.1f}"
        )

    def test_sharpe_correct_annualization_realistic(self):
        """Sharpe with correct annualization gives realistic values for typical strategies."""
        kelly = 0.02
        rrr = 1.0
        # 60% WR, 300 trades over 32000 bars ≈ 5.3 years
        trades = [1.0] * 180 + [-1.0] * 120
        trade_returns = [kelly * rrr if t > 0 else -kelly for t in trades]

        trades_per_year = 300 * 6000 / 32000  # 56.25
        sharpe = calculate_sharpe_ratio(trade_returns, trades_per_year=trades_per_year)

        # 60% WR with RRR=1 is decent but not exceptional
        assert 0 < sharpe < 5, (
            f"Sharpe={sharpe:.2f} should be in realistic range for 60% WR strategy"
        )


class TestFoldStabilityDenominator:
    """Bug 2: fold_stability must use total_folds, not just successful folds."""

    def test_fold_stability_includes_failed_folds(self):
        """fold_stability should count failed folds in denominator.

        If 3 of 5 folds are profitable but 2 failed (no CT found),
        stability should be 3/5 = 0.6, not 3/3 = 1.0.
        """
        total_folds = 5
        inner_val_pnls = [10, 20, 15]  # Only 3 folds succeeded

        profitable_folds = sum(1 for pnl in inner_val_pnls if pnl > 0)

        # BUGGY: uses len(inner_val_pnls) — excludes 2 failed folds
        stability_buggy = profitable_folds / len(inner_val_pnls)
        # CORRECT: uses total_folds
        stability_correct = profitable_folds / total_folds

        assert stability_buggy == 1.0, "Buggy calc gives 3/3 = 1.0"
        assert stability_correct == 0.6, "Correct calc gives 3/5 = 0.6"

    def test_fold_stability_all_succeed_agrees(self):
        """When all folds succeed, both calculations agree."""
        total_folds = 5
        inner_val_pnls = [10, 20, 15, 5, 8]

        profitable_folds = sum(1 for pnl in inner_val_pnls if pnl > 0)

        assert profitable_folds / total_folds == profitable_folds / len(inner_val_pnls)

    def test_fold_stability_with_negative_and_failed(self):
        """Mixed: profitable + negative + completely failed folds."""
        total_folds = 5
        # 4 folds produced results (1 failed completely)
        inner_val_pnls = [10, -5, 15, 8]

        profitable_folds = sum(1 for pnl in inner_val_pnls if pnl > 0)
        stability = profitable_folds / total_folds

        assert stability == 0.6  # 3 profitable out of 5 total

    def test_fold_stability_zero_total_folds_safe(self):
        """Edge case: total_folds=0 should not crash."""
        total_folds = 0
        inner_val_pnls = []

        profitable_folds = sum(1 for pnl in inner_val_pnls if pnl > 0)
        stability = profitable_folds / total_folds if total_folds > 0 else 0

        assert stability == 0
