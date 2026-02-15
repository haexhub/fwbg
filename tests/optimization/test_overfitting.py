"""Tests for Deflated Sharpe Ratio and Probability of Backtest Overfitting."""

import numpy as np
import pytest
from math import comb

from fwbg.optimization.overfitting import (
    build_performance_matrix,
    compute_overfitting_metrics,
    deflated_sharpe_ratio,
    expected_max_sr,
    probability_of_backtest_overfitting,
    sr_std,
)


# === DSR Tests ===


class TestExpectedMaxSR:
    def test_monotonic_in_n(self):
        """E[max(SR)] must increase with number of strategies."""
        vals = [expected_max_sr(n) for n in [2, 10, 50, 100, 500]]
        for a, b in zip(vals, vals[1:]):
            assert b > a

    def test_known_value_n100(self):
        """N=100 should yield ~2.5 (Bailey & López de Prado, 2014)."""
        val = expected_max_sr(100)
        assert 2.2 < val < 2.8

    def test_n1_returns_zero(self):
        """With one strategy, expected max SR is 0 (no multiple testing)."""
        assert expected_max_sr(1) == pytest.approx(0.0, abs=0.01)

    def test_large_n(self):
        """Should not overflow for large N."""
        val = expected_max_sr(10000)
        assert np.isfinite(val)
        assert val > 3.0


class TestSRStd:
    def test_normal_returns_zero_sr(self):
        """For normal returns (skew=0, kurt=3) and SR=0, σ(SR) = 1/√(T-1)."""
        n = 100
        expected = 1.0 / np.sqrt(n - 1)
        actual = sr_std(observed_sr=0.0, n_trades=n, skewness=0.0, kurtosis=3.0)
        assert actual == pytest.approx(expected, rel=0.01)

    def test_higher_kurtosis_increases_std(self):
        """Fat tails should increase SR standard deviation."""
        base = sr_std(0.5, 200, skewness=0.0, kurtosis=3.0)
        fat = sr_std(0.5, 200, skewness=0.0, kurtosis=6.0)
        assert fat > base

    def test_positive_skew_effect(self):
        """Skewness should affect the standard deviation."""
        sym = sr_std(1.0, 200, skewness=0.0, kurtosis=3.0)
        skewed = sr_std(1.0, 200, skewness=1.0, kurtosis=3.0)
        assert sym != skewed


class TestDeflatedSharpeRatio:
    def test_high_sr_few_strategies(self):
        """SR=2.0 with only 5 strategies → DSR should be high (> 0.95)."""
        result = deflated_sharpe_ratio(
            observed_sr=2.0, n_trades=500, n_strategies=5
        )
        assert result["dsr"] > 0.95
        assert result["is_significant"] is True
        assert result["n_strategies"] == 5

    def test_modest_sr_many_strategies(self):
        """SR=1.5 with 500 strategies → DSR should be low (< 0.50)."""
        result = deflated_sharpe_ratio(
            observed_sr=1.5, n_trades=300, n_strategies=500
        )
        assert result["dsr"] < 0.50
        assert result["is_significant"] is False

    def test_zero_sr(self):
        """SR=0 should always yield low DSR."""
        result = deflated_sharpe_ratio(
            observed_sr=0.0, n_trades=200, n_strategies=10
        )
        assert result["dsr"] < 0.10

    def test_result_keys(self):
        """Result dict must have all expected keys."""
        result = deflated_sharpe_ratio(
            observed_sr=1.0, n_trades=200, n_strategies=50
        )
        assert set(result.keys()) == {
            "dsr", "observed_sr", "expected_max_sr", "n_strategies", "is_significant"
        }

    def test_dsr_bounded_0_1(self):
        """DSR (a probability) must be between 0 and 1."""
        for sr, n in [(0.0, 100), (3.0, 5), (1.0, 1000)]:
            result = deflated_sharpe_ratio(
                observed_sr=sr, n_trades=200, n_strategies=n
            )
            assert 0.0 <= result["dsr"] <= 1.0


# === PBO Tests ===


class TestBuildPerformanceMatrix:
    def test_basic(self):
        """Builds matrix from grid results with common combos only."""
        grid_results_by_fold = {
            0: [
                {"tp_mult": 10, "sl_mult": 20, "timeout_bars": 0, "inner_val_pnl": 1.5},
                {"tp_mult": 15, "sl_mult": 25, "timeout_bars": 0, "inner_val_pnl": 2.0},
            ],
            1: [
                {"tp_mult": 10, "sl_mult": 20, "timeout_bars": 0, "inner_val_pnl": 1.2},
                {"tp_mult": 15, "sl_mult": 25, "timeout_bars": 0, "inner_val_pnl": 1.8},
                {"tp_mult": 99, "sl_mult": 99, "timeout_bars": 0, "inner_val_pnl": 0.5},  # only in fold 1
            ],
        }
        matrix, combo_keys = build_performance_matrix(grid_results_by_fold)
        # Only 2 combos common to both folds
        assert matrix.shape == (2, 2)
        assert len(combo_keys) == 2

    def test_missing_combos_filtered(self):
        """Combos not present in ALL folds should be excluded."""
        grid_results_by_fold = {
            0: [{"tp_mult": 10, "sl_mult": 20, "timeout_bars": 0, "inner_val_pnl": 1.0}],
            1: [{"tp_mult": 99, "sl_mult": 99, "timeout_bars": 0, "inner_val_pnl": 2.0}],
        }
        matrix, combo_keys = build_performance_matrix(grid_results_by_fold)
        # No common combos
        assert matrix.shape[0] == 0

    def test_preserves_fold_order(self):
        """Columns should correspond to fold order."""
        grid_results_by_fold = {
            0: [{"tp_mult": 10, "sl_mult": 20, "timeout_bars": 0, "inner_val_pnl": 5.0}],
            1: [{"tp_mult": 10, "sl_mult": 20, "timeout_bars": 0, "inner_val_pnl": 3.0}],
            2: [{"tp_mult": 10, "sl_mult": 20, "timeout_bars": 0, "inner_val_pnl": 7.0}],
        }
        matrix, _ = build_performance_matrix(grid_results_by_fold)
        assert matrix[0, 0] == 5.0
        assert matrix[0, 1] == 3.0
        assert matrix[0, 2] == 7.0


class TestProbabilityOfBacktestOverfitting:
    def test_consistent_strategy_low_pbo(self):
        """A strategy that dominates in all folds should have low PBO."""
        rng = np.random.default_rng(42)
        n_combos, n_folds = 20, 8
        # Combo 0 is consistently the best
        matrix = rng.normal(0, 1, (n_combos, n_folds))
        matrix[0, :] += 5.0  # Strong consistent performer
        result = probability_of_backtest_overfitting(matrix)
        assert result["pbo"] < 0.20
        assert result["is_overfit"] is False

    def test_pure_noise_high_pbo(self):
        """Pure noise should yield high PBO (> 0.30)."""
        rng = np.random.default_rng(123)
        matrix = rng.normal(0, 1, (50, 8))
        result = probability_of_backtest_overfitting(matrix)
        assert result["pbo"] > 0.30

    def test_8_folds_70_splits(self):
        """C(8,4) = 70 CSCV splits."""
        rng = np.random.default_rng(42)
        matrix = rng.normal(0, 1, (10, 8))
        result = probability_of_backtest_overfitting(matrix)
        assert result["n_cscv_splits"] == comb(8, 4)  # 70

    def test_pbo_bounded(self):
        """PBO must be between 0 and 1."""
        rng = np.random.default_rng(42)
        matrix = rng.normal(0, 1, (15, 8))
        result = probability_of_backtest_overfitting(matrix)
        assert 0.0 <= result["pbo"] <= 1.0

    def test_result_keys(self):
        """Result dict must have expected keys."""
        rng = np.random.default_rng(42)
        matrix = rng.normal(0, 1, (10, 8))
        result = probability_of_backtest_overfitting(matrix)
        assert set(result.keys()) == {
            "pbo", "n_cscv_splits", "is_overfit", "degradation", "logit_mean"
        }

    def test_too_few_folds(self):
        """With < 4 folds, PBO cannot be computed."""
        rng = np.random.default_rng(42)
        matrix = rng.normal(0, 1, (10, 3))
        result = probability_of_backtest_overfitting(matrix)
        assert result["pbo"] is None
        assert result["n_cscv_splits"] == 0


# === Combined Entry Point ===


class TestComputeOverfittingMetrics:
    def test_returns_both_dsr_and_pbo(self):
        """Entry point should return both DSR and PBO sub-dicts."""
        rng = np.random.default_rng(42)
        # Build fake grid results for 8 folds, 10 combos
        grid_results_by_fold = {}
        for fid in range(8):
            fold_results = []
            for tp in [10, 20]:
                for sl in [15, 30]:
                    for ct in [0.55, 0.60]:
                        fold_results.append({
                            "tp_mult": tp,
                            "sl_mult": sl,
                            "timeout_bars": 0,
                            "inner_val_pnl": rng.normal(1.0, 0.5),
                        })
            grid_results_by_fold[fid] = fold_results

        trade_returns = list(rng.normal(0.01, 0.1, 200))
        result = compute_overfitting_metrics(
            trade_returns=trade_returns,
            observed_sr=1.2,
            n_strategies=64,
            grid_results_by_fold=grid_results_by_fold,
            n_trades=200,
        )
        assert "dsr" in result
        assert "pbo" in result
        assert "dsr" in result["dsr"]
        assert "pbo" in result["pbo"]

    def test_empty_grid_results(self):
        """Should handle empty grid results gracefully."""
        result = compute_overfitting_metrics(
            trade_returns=[0.1, -0.05, 0.08],
            observed_sr=0.5,
            n_strategies=10,
            grid_results_by_fold={},
            n_trades=3,
        )
        assert result["pbo"]["pbo"] is None
