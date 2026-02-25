"""Tests for unified simulation — merge settings and re-simulate all folds."""
import pytest
import numpy as np

from fwbg.optimization.unified_simulation import (
    merge_unified_settings,
    _majority_vote_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fold_result(fold_id, tp, sl, ct, timeout_bars=None,
                      model_hyperparameters=None, exit_modifier_params=None,
                      features_long=None, features_short=None):
    """Create a minimal fold_result dict for merge testing."""
    return {
        "fold_id": fold_id,
        "test_pnl": 100.0,
        "test_win_rate": 0.55,
        "test_trades": 20,
        "inner_val_pnl": 150.0,
        "test_trades_trace": [],
        "best_config": {
            "tp": tp,
            "sl": sl,
            "ct": ct,
            "rrr": tp / sl if sl > 0 else 1.0,
            "timeout_bars": timeout_bars,
            "model_hyperparameters": model_hyperparameters,
            "exit_modifier_params": exit_modifier_params,
        },
        "selected_features_long": features_long or [],
        "selected_features_short": features_short or [],
    }


# ---------------------------------------------------------------------------
# merge_unified_settings
# ---------------------------------------------------------------------------

class TestMergeUnifiedSettings:

    def test_median_tp_sl_ct(self):
        """TP, SL, CT should be the median across folds."""
        folds = [
            _make_fold_result(0, tp=3.0, sl=1.0, ct=0.5),
            _make_fold_result(1, tp=5.0, sl=2.0, ct=0.6),
            _make_fold_result(2, tp=7.0, sl=3.0, ct=0.7),
        ]
        result = merge_unified_settings(folds, folds)
        tp, sl, ct = result["params"]
        assert tp == 5.0
        assert sl == 2.0
        assert ct == 0.6

    def test_median_even_number_of_folds(self):
        """With even folds, median interpolates."""
        folds = [
            _make_fold_result(0, tp=3.0, sl=1.0, ct=0.5),
            _make_fold_result(1, tp=5.0, sl=2.0, ct=0.7),
        ]
        result = merge_unified_settings(folds, folds)
        tp, sl, ct = result["params"]
        assert tp == 4.0
        assert sl == 1.5
        assert ct == pytest.approx(0.6)

    def test_tuple_ct(self):
        """Tuple CT (separate long/short) should have each component medianed."""
        folds = [
            _make_fold_result(0, tp=5.0, sl=2.0, ct=(0.5, 0.6)),
            _make_fold_result(1, tp=5.0, sl=2.0, ct=(0.7, 0.8)),
            _make_fold_result(2, tp=5.0, sl=2.0, ct=(0.3, 0.4)),
        ]
        result = merge_unified_settings(folds, folds)
        ct = result["params"][2]
        assert isinstance(ct, tuple)
        assert ct[0] == 0.5
        assert ct[1] == 0.6

    def test_timeout_bars_median(self):
        """timeout_bars should be the median of non-None values."""
        folds = [
            _make_fold_result(0, tp=5.0, sl=2.0, ct=0.5, timeout_bars=10),
            _make_fold_result(1, tp=5.0, sl=2.0, ct=0.5, timeout_bars=20),
            _make_fold_result(2, tp=5.0, sl=2.0, ct=0.5, timeout_bars=30),
        ]
        result = merge_unified_settings(folds, folds)
        assert result["timeout_bars"] == 20

    def test_timeout_bars_all_none(self):
        """If all folds have timeout_bars=None, result should be None."""
        folds = [
            _make_fold_result(0, tp=5.0, sl=2.0, ct=0.5, timeout_bars=None),
            _make_fold_result(1, tp=5.0, sl=2.0, ct=0.5, timeout_bars=None),
        ]
        result = merge_unified_settings(folds, folds)
        assert result["timeout_bars"] is None

    def test_timeout_bars_mixed_none(self):
        """None timeout_bars excluded from median, non-None values medianed."""
        folds = [
            _make_fold_result(0, tp=5.0, sl=2.0, ct=0.5, timeout_bars=None),
            _make_fold_result(1, tp=5.0, sl=2.0, ct=0.5, timeout_bars=10),
            _make_fold_result(2, tp=5.0, sl=2.0, ct=0.5, timeout_bars=30),
        ]
        result = merge_unified_settings(folds, folds)
        assert result["timeout_bars"] == 20

    def test_model_hyperparameters_from_first_fold(self):
        """model_hyperparameters taken from first consistent fold."""
        hp = {"signal_column_long": "rl50_pdl_retest_bull"}
        folds = [
            _make_fold_result(0, tp=5.0, sl=2.0, ct=0.5, model_hyperparameters=hp),
            _make_fold_result(1, tp=5.0, sl=2.0, ct=0.5, model_hyperparameters=hp),
        ]
        result = merge_unified_settings(folds, folds)
        assert result["model_hyperparameters"] == hp

    def test_exit_modifier_params_majority_vote(self):
        """exit_modifier_params should be the most common dict."""
        emp_a = {"breakeven_trigger": 0.5, "trail_atr_mult": 0.3}
        emp_b = {"breakeven_trigger": 0.0, "trail_atr_mult": 0.0}
        folds = [
            _make_fold_result(0, tp=5.0, sl=2.0, ct=0.5, exit_modifier_params=emp_a),
            _make_fold_result(1, tp=5.0, sl=2.0, ct=0.5, exit_modifier_params=emp_a),
            _make_fold_result(2, tp=5.0, sl=2.0, ct=0.5, exit_modifier_params=emp_b),
        ]
        result = merge_unified_settings(folds, folds)
        assert result["exit_modifier_params"] == emp_a

    def test_features_stability_threshold(self):
        """Features appearing in >= 50% of ALL folds should be selected."""
        folds = [
            _make_fold_result(0, tp=5.0, sl=2.0, ct=0.5,
                              features_long=["feat_a", "feat_b"],
                              features_short=["feat_x"]),
            _make_fold_result(1, tp=5.0, sl=2.0, ct=0.5,
                              features_long=["feat_a", "feat_c"],
                              features_short=["feat_x", "feat_y"]),
            _make_fold_result(2, tp=5.0, sl=2.0, ct=0.5,
                              features_long=["feat_a"],
                              features_short=["feat_z"]),
        ]
        # feat_a: 3/3 = 1.0 -> stable
        # feat_b: 1/3 = 0.33 -> unstable
        # feat_c: 1/3 = 0.33 -> unstable
        # feat_x: 2/3 = 0.67 -> stable
        # feat_y: 1/3 = 0.33 -> unstable
        # feat_z: 1/3 = 0.33 -> unstable
        result = merge_unified_settings(folds, folds)
        assert result["selected_features_long"] == ["feat_a"]
        assert result["selected_features_short"] == ["feat_x"]

    def test_features_stability_uses_all_folds(self):
        """Feature stability uses all_fold_results, not just consistent_folds."""
        consistent = [
            _make_fold_result(0, tp=5.0, sl=2.0, ct=0.5,
                              features_long=["feat_a"]),
        ]
        all_folds = [
            _make_fold_result(0, tp=5.0, sl=2.0, ct=0.5,
                              features_long=["feat_a"]),
            _make_fold_result(1, tp=5.0, sl=2.0, ct=0.5,
                              features_long=[]),
        ]
        # feat_a appears in 1/2 = 0.5 => stable (>= 0.5)
        result = merge_unified_settings(consistent, all_folds)
        assert "feat_a" in result["selected_features_long"]

    def test_features_below_threshold_excluded(self):
        """Features below 50% stability excluded."""
        all_folds = [
            _make_fold_result(0, tp=5.0, sl=2.0, ct=0.5,
                              features_long=["rare_feat"]),
            _make_fold_result(1, tp=5.0, sl=2.0, ct=0.5,
                              features_long=[]),
            _make_fold_result(2, tp=5.0, sl=2.0, ct=0.5,
                              features_long=[]),
        ]
        # rare_feat: 1/3 = 0.33 -> below threshold
        result = merge_unified_settings(all_folds, all_folds)
        assert result["selected_features_long"] == []

    def test_single_fold(self):
        """Edge case: single fold returns that fold's settings."""
        fold = _make_fold_result(
            0, tp=5.0, sl=2.0, ct=0.5, timeout_bars=10,
            model_hyperparameters={"key": "val"},
            exit_modifier_params={"be": 0.5},
            features_long=["f1"], features_short=["f2"],
        )
        result = merge_unified_settings([fold], [fold])
        tp, sl, ct = result["params"]
        assert tp == 5.0
        assert sl == 2.0
        assert ct == 0.5
        assert result["timeout_bars"] == 10
        assert result["model_hyperparameters"] == {"key": "val"}
        assert result["exit_modifier_params"] == {"be": 0.5}
        assert result["selected_features_long"] == ["f1"]
        assert result["selected_features_short"] == ["f2"]


# ---------------------------------------------------------------------------
# _majority_vote_dict
# ---------------------------------------------------------------------------

class TestMajorityVoteDict:

    def test_clear_winner(self):
        dicts = [{"a": 1}, {"a": 1}, {"b": 2}]
        assert _majority_vote_dict(dicts) == {"a": 1}

    def test_all_same(self):
        dicts = [{"x": 10}] * 5
        assert _majority_vote_dict(dicts) == {"x": 10}

    def test_all_none(self):
        assert _majority_vote_dict([None, None, None]) is None

    def test_none_vs_dict(self):
        dicts = [None, None, {"a": 1}]
        assert _majority_vote_dict(dicts) is None  # None is majority

    def test_empty_list(self):
        assert _majority_vote_dict([]) is None
