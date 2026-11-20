"""Tests for Deflated Sharpe Ratio and Probability of Backtest Overfitting."""

import numpy as np
import pytest
from math import comb

from fwbg.optimization import overfitting
from fwbg.optimization.overfitting import (
    build_performance_matrix,
    compute_overfitting_metrics,
    count_trials,
    deflated_sharpe_ratio,
    expected_max_sr,
    probability_of_backtest_overfitting,
    sr_std,
)
from fwbg.optimization.signal_fold import process_signal_fold
from tests.optimization.signal_harness import (
    SIGNALS,
    one_fold,
    signal_ctx,
    signal_frame,
)


# === DSR Tests ===


class TestExpectedMaxSR:
    def test_monotonic_in_n(self):
        """E[max(SR)] must increase with number of strategies."""
        vals = [expected_max_sr(n) for n in [2, 10, 50, 100, 500]]
        for a, b in zip(vals, vals[1:]):
            assert b > a

    def test_known_value_n100(self):
        """Paper formula for N=100, V[SR]=1, evaluated independently with scipy."""
        # (1-g)*Phi^-1(1-1/N) + g*Phi^-1(1-1/(N*e)), g = Euler-Mascheroni
        assert expected_max_sr(100) == pytest.approx(2.5306028932, abs=1e-9)

    def test_scales_with_the_std_of_the_trial_srs(self):
        """E[max SR] is in the unit of the SR: it scales with sqrt(V[SR])."""
        # N=500, V=0.04 (std 0.2): 0.2 * E[max] at unit variance
        assert expected_max_sr(500, sr_variance=0.04) == pytest.approx(
            0.6105055149, abs=1e-9
        )
        assert expected_max_sr(500, sr_variance=0.04) == pytest.approx(
            0.2 * expected_max_sr(500, sr_variance=1.0)
        )

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
        result = deflated_sharpe_ratio(observed_sr=2.0, n_trades=500, n_strategies=5)
        assert result["dsr"] > 0.95
        assert result["is_significant"] is True
        assert result["n_strategies"] == 5

    def test_modest_sr_many_strategies(self):
        """SR=0.5 against 500 trials whose SRs have std 0.2 → DSR is low."""
        result = deflated_sharpe_ratio(
            observed_sr=0.5, n_trades=300, n_strategies=500, sr_variance=0.04
        )
        # PSR against SR_0 = 0.6105055149: independent calculation
        assert result["dsr"] == pytest.approx(0.0358091080, abs=1e-8)
        assert result["expected_max_sr"] == pytest.approx(0.6105055149, abs=1e-9)
        assert result["is_significant"] is False

    def test_per_trade_sr_is_not_zero_or_one(self):
        """A per-trade SR of 0.1 over 1000 trades and 100 trials is neither 0 nor 1.

        The un-scaled E[max SR] (2.78 in SR units) pushed any realistic per-trade SR
        to a DSR of 0.0.
        """
        result = deflated_sharpe_ratio(observed_sr=0.1, n_trades=1000, n_strategies=100)
        # default V[SR] = 1/(T-1); SR_0 = 0.0800647323
        assert result["expected_max_sr"] == pytest.approx(0.0800647323, abs=1e-9)
        assert result["dsr"] == pytest.approx(0.7351695966, abs=1e-8)

    def test_single_trial_is_the_psr(self):
        """N=1: no deflation, DSR is the probabilistic Sharpe ratio against 0."""
        result = deflated_sharpe_ratio(observed_sr=0.1, n_trades=1000, n_strategies=1)
        assert result["expected_max_sr"] == 0.0
        assert 0.99 < result["dsr"] < 1.0

    def test_undefined_variance_gives_none_not_zero(self):
        """Skew/kurtosis that leave no positive variance: DSR is None, not a fake 0.0."""
        result = deflated_sharpe_ratio(
            observed_sr=2.0, n_trades=100, n_strategies=2, skewness=5.0, kurtosis=1.0
        )
        assert result["dsr"] is None
        assert result["is_significant"] is False

    def test_zero_sr(self):
        """SR=0 should always yield low DSR."""
        result = deflated_sharpe_ratio(observed_sr=0.0, n_trades=200, n_strategies=10)
        assert result["dsr"] < 0.10

    def test_result_keys(self):
        """Result dict must have all expected keys."""
        result = deflated_sharpe_ratio(observed_sr=1.0, n_trades=200, n_strategies=50)
        assert set(result.keys()) == {
            "dsr",
            "observed_sr",
            "expected_max_sr",
            "sr_variance",
            "n_strategies",
            "is_significant",
        }

    def test_dsr_bounded_0_1(self):
        """DSR (a probability) must be between 0 and 1."""
        for sr, n in [(0.0, 100), (3.0, 5), (1.0, 1000)]:
            result = deflated_sharpe_ratio(observed_sr=sr, n_trades=200, n_strategies=n)
            assert 0.0 <= result["dsr"] <= 1.0


class TestCountTrials:
    def test_distinct_combos_not_fold_rows(self):
        """The same combo re-evaluated in every fold is one trial."""
        rows = [
            {"tp_mult": tp, "sl_mult": 10, "timeout_bars": 0, "fold_id": fold}
            for fold in range(5)
            for tp in (5, 10, 15)
        ]
        assert len(rows) == 15
        assert count_trials(rows) == 3

    def test_variants_multiply_and_floor_is_one(self):
        rows = [{"tp_mult": 5, "sl_mult": 10}, {"tp_mult": 10, "sl_mult": 10}]
        assert count_trials(rows, n_variants=3) == 6
        assert count_trials([]) == 1

    def test_unkeyable_entry_counts_as_a_trial(self):
        assert count_trials([{"tp_mult": 5, "sl_mult": 10}, {"foo": 1}]) == 2


# === PBO Tests ===


def _reaches_every_tp(frame):
    """Two bars after each signal the high reaches every TP: each combo trades."""
    frame = frame.copy()
    frame.iloc[7::24, frame.columns.get_loc("H")] = 110.0
    return frame


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
                {
                    "tp_mult": 99,
                    "sl_mult": 99,
                    "timeout_bars": 0,
                    "inner_val_pnl": 0.5,
                },  # only in fold 1
            ],
        }
        matrix, combo_keys = build_performance_matrix(grid_results_by_fold)
        # Only 2 combos common to both folds
        assert matrix.shape == (2, 2)
        assert len(combo_keys) == 2

    def test_missing_combos_filtered(self):
        """Combos not present in ALL folds should be excluded."""
        grid_results_by_fold = {
            0: [
                {"tp_mult": 10, "sl_mult": 20, "timeout_bars": 0, "inner_val_pnl": 1.0}
            ],
            1: [
                {"tp_mult": 99, "sl_mult": 99, "timeout_bars": 0, "inner_val_pnl": 2.0}
            ],
        }
        matrix, combo_keys = build_performance_matrix(grid_results_by_fold)
        # No common combos
        assert matrix.shape[0] == 0

    def test_preserves_fold_order(self):
        """Columns should correspond to fold order."""
        grid_results_by_fold = {
            0: [
                {"tp_mult": 10, "sl_mult": 20, "timeout_bars": 0, "inner_val_pnl": 5.0}
            ],
            1: [
                {"tp_mult": 10, "sl_mult": 20, "timeout_bars": 0, "inner_val_pnl": 3.0}
            ],
            2: [
                {"tp_mult": 10, "sl_mult": 20, "timeout_bars": 0, "inner_val_pnl": 7.0}
            ],
        }
        matrix, _ = build_performance_matrix(grid_results_by_fold)
        assert matrix[0, 0] == 5.0
        assert matrix[0, 1] == 3.0
        assert matrix[0, 2] == 7.0

    def test_performance_matrix_reads_signal_grid_entries(self):
        """Signal folds' grid entries (inner metrics + fold_id) feed the PBO matrix."""
        grid_results_by_fold = {}
        for fold_id, start in enumerate(("2024-01-01", "2024-02-20")):
            train_df = _reaches_every_tp(signal_frame(960, start=start))
            test_df = signal_frame(240, start=train_df.index[-1] + train_df.index.freq)
            fold = one_fold(train_df, test_df, fold_id)
            _, grid = process_signal_fold(
                fold, train_df, test_df, SIGNALS, signal_ctx(), "TEST"
            )
            for gr in grid:
                grid_results_by_fold.setdefault(gr["fold_id"], []).append(gr)

        assert all(
            gr["inner_trades"] > 0
            for grid in grid_results_by_fold.values()
            for gr in grid
        )
        matrix, combo_keys = build_performance_matrix(grid_results_by_fold)
        # tp 2 / ct 0.5, tp 6 / ct 0.5 and tp 6 / ct 0.6: the CT pair keeps two rows.
        assert matrix.shape == (3, 2)
        assert len(set(combo_keys)) == 3

    def test_combos_differing_only_by_ct_or_hp_keep_their_rows(self):
        base = {
            "exit_strategy": "fixed",
            "tp_mult": 2,
            "sl_mult": 4,
            "timeout_bars": None,
        }
        entries = [
            {
                **base,
                "ct": 0.5,
                "model_hyperparameters": {"a": [1, 2]},
                "inner_val_pnl": 1.0,
            },
            {
                **base,
                "ct": 0.6,
                "model_hyperparameters": {"a": [1, 2]},
                "inner_val_pnl": 2.0,
            },
            {
                **base,
                "ct": 0.5,
                "model_hyperparameters": {"a": [3]},
                "inner_val_pnl": 3.0,
            },
        ]
        matrix, _ = build_performance_matrix({0: entries, 1: entries})
        assert sorted(matrix[:, 0]) == [1.0, 2.0, 3.0]

    def test_combos_differing_only_by_entry_settings_keep_their_rows(self):
        base = {
            "exit_strategy": "fixed",
            "tp_mult": 2,
            "sl_mult": 4,
            "timeout_bars": None,
        }
        entries = [
            {
                **base,
                "entry_modifier": "plain",
                "entry_modifier_params": {},
                "inner_val_pnl": 1.0,
            },
            {
                **base,
                "entry_modifier": "scale_in",
                "entry_modifier_params": {"levels": [0.5]},
                "inner_val_pnl": 2.0,
            },
        ]
        matrix, _ = build_performance_matrix({0: entries, 1: entries})
        assert sorted(matrix[:, 0]) == [1.0, 2.0]

    def test_ml_entries_match_across_folds_despite_their_chosen_ct(self):
        """An ML entry's conf_thresh is chosen per fold, not a grid dimension."""
        entry = {
            "exit_strategy": "atr_based",
            "tp_mult": 2,
            "sl_mult": 1,
            "timeout_bars": 0,
        }
        matrix, _ = build_performance_matrix(
            {
                0: [{**entry, "conf_thresh": 0.55, "inner_val_pnl": 1.0}],
                1: [{**entry, "conf_thresh": 0.65, "inner_val_pnl": 2.0}],
            }
        )
        assert matrix.tolist() == [[1.0, 2.0]]

    def test_signal_entries_without_inner_trades_are_left_out(self):
        """A combo with no inner trade is ineligible (D1): it is no PBO candidate."""
        base = {"exit_strategy": "fixed", "sl_mult": 4, "timeout_bars": None, "ct": 0.5}
        entries = [
            {**base, "tp_mult": 2, "inner_val_pnl": 0.0, "inner_trades": 0},
            {**base, "tp_mult": 6, "inner_val_pnl": 1.0, "inner_trades": 3},
        ]
        matrix, combo_keys = build_performance_matrix({0: entries, 1: entries})
        assert matrix.tolist() == [[1.0, 1.0]]
        assert [key[0] for key in combo_keys] == [6]

    def test_unhashable_grid_values_are_skipped(self):
        entries = [
            {
                "tp_mult": 2,
                "sl_mult": 4,
                "model_hyperparameters": {"a": {1}},
                "inner_val_pnl": 1.0,
            },
            {"tp_mult": 6, "sl_mult": 4, "inner_val_pnl": 2.0},
        ]
        matrix, _ = build_performance_matrix({0: entries, 1: entries})
        assert matrix.tolist() == [[2.0, 2.0]]

    def test_none_and_int_timeouts_sort_without_error(self):
        entries = [
            {"tp_mult": 2, "sl_mult": 4, "timeout_bars": None, "inner_val_pnl": 1.0},
            {"tp_mult": 2, "sl_mult": 4, "timeout_bars": 12, "inner_val_pnl": 2.0},
        ]
        matrix, combo_keys = build_performance_matrix({0: entries, 1: entries})
        assert matrix.shape == (2, 2)
        assert len(combo_keys) == 2


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
            "pbo",
            "n_cscv_splits",
            "is_overfit",
            "degradation",
            "logit_mean",
        }

    def test_single_combo_is_undefined_and_warns(self):
        """One combo: nothing to select from. pbo is None (not 0.0) and warns."""
        matrix = np.random.default_rng(1).normal(0, 1, (1, 8))
        with pytest.warns(RuntimeWarning, match="only one combo"):
            result = probability_of_backtest_overfitting(matrix)
        assert result["pbo"] is None
        assert result["is_overfit"] is None
        assert "one combo" in result["reason"]

    def test_too_few_folds(self):
        """With < 4 folds, PBO cannot be computed."""
        rng = np.random.default_rng(42)
        matrix = rng.normal(0, 1, (10, 3))
        result = probability_of_backtest_overfitting(matrix)
        assert result["pbo"] is None
        assert result["n_cscv_splits"] == 0


# === Combined Entry Point ===


class TestComputeOverfittingMetrics:
    def test_a_pbo_failure_keeps_the_dsr(self, monkeypatch):
        def broken(grid_results_by_fold):
            raise ValueError("bad grid")

        monkeypatch.setattr(overfitting, "build_performance_matrix", broken)
        result = compute_overfitting_metrics(
            trade_returns=list(np.random.default_rng(1).normal(0.01, 0.1, 50)),
            observed_sr=0.2,
            n_strategies=4,
            grid_results_by_fold={0: []},
            n_trades=50,
        )
        assert result["dsr"]["observed_sr"] == 0.2
        assert result["pbo"]["pbo"] is None
        assert result["pbo"]["error"] == "bad grid"

    def test_reports_survivor_coverage(self):
        """Combos missing in a fold drop out of the matrix; the result says how many."""
        rows = {
            fold: [
                {"tp_mult": 5, "sl_mult": 10, "inner_val_pnl": 1.0 + fold},
                {"tp_mult": 10, "sl_mult": 10, "inner_val_pnl": 2.0 - fold},
            ]
            for fold in range(4)
        }
        rows[3].append({"tp_mult": 15, "sl_mult": 10, "inner_val_pnl": 9.0})
        result = compute_overfitting_metrics(
            trade_returns=[0.1, -0.1] * 10,
            observed_sr=0.0,
            n_strategies=3,
            grid_results_by_fold=rows,
            n_trades=20,
        )
        assert result["pbo"]["n_combos_in_matrix"] == 2
        assert result["pbo"]["n_combos_evaluated"] == 3

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
                        fold_results.append(
                            {
                                "tp_mult": tp,
                                "sl_mult": sl,
                                "timeout_bars": 0,
                                "inner_val_pnl": rng.normal(1.0, 0.5),
                            }
                        )
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
