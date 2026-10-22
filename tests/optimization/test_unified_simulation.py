"""Tests for unified simulation — merge settings and re-simulate all folds."""
import dataclasses
from types import SimpleNamespace

import pytest
import pandas as pd

from fwbg.optimization import unified_simulation
from fwbg.optimization.unified_simulation import (
    causal_candidates,
    consistent_folds,
    hp_key,
    merge_unified_settings,
    run_unified_simulation,
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
        hp = {"signal_column_long": "hl_ses_rl50_pdl_retest_bull"}
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


# ---------------------------------------------------------------------------
# hp_key / consistent_folds
# ---------------------------------------------------------------------------

class TestConsistentFolds:

    def test_all_folds_share_hp(self):
        hp = {"a": 1}
        folds = [_make_fold_result(i, 5.0, 2.0, 0.5, model_hyperparameters=hp)
                 for i in range(3)]
        assert consistent_folds(folds) == folds

    def test_majority_hp_wins(self):
        folds = [
            _make_fold_result(0, 5.0, 2.0, 0.5, model_hyperparameters={"a": 1}),
            _make_fold_result(1, 5.0, 2.0, 0.5, model_hyperparameters={"a": 2}),
            _make_fold_result(2, 5.0, 2.0, 0.5, model_hyperparameters={"a": 1}),
        ]
        assert consistent_folds(folds) == [folds[0], folds[2]]

    def test_tie_goes_to_hp_seen_first(self):
        folds = [
            _make_fold_result(0, 5.0, 2.0, 0.5, model_hyperparameters={"a": 2}),
            _make_fold_result(1, 5.0, 2.0, 0.5, model_hyperparameters={"a": 1}),
        ]
        assert consistent_folds(folds) == [folds[0]]

    def test_hp_key_ignores_none_and_hashes_lists(self):
        assert hp_key({"a": [1, 2], "b": None}) == (("a", (1, 2)),)

    @pytest.mark.parametrize("hp", [None, {}])
    def test_hp_key_empty(self, hp):
        assert hp_key(hp) == ()


# ---------------------------------------------------------------------------
# causal_candidates
# ---------------------------------------------------------------------------

WF_FOLDS = [SimpleNamespace(fold_id=i) for i in range(4)]


class TestCausalCandidates:

    def test_prefix_median_per_fold(self):
        results = [_make_fold_result(i, tp=tp, sl=2.0, ct=0.5)
                   for i, tp in enumerate([1.0, 3.0, 5.0, 7.0])]
        candidates = causal_candidates(WF_FOLDS, results)
        assert [c["params"][0] for c in candidates] == [1.0, 2.0, 3.0, 4.0]

    def test_no_setting_before_first_successful_fold(self):
        r1 = _make_fold_result(1, tp=3.0, sl=2.0, ct=0.5)
        r3 = _make_fold_result(3, tp=7.0, sl=2.0, ct=0.5)
        candidates = causal_candidates(WF_FOLDS, [r1, r3])
        c1 = merge_unified_settings([r1], [r1])
        c13 = merge_unified_settings([r1, r3], [r1, r3])
        assert candidates == [None, c1, c1, c13]

    def test_feature_stability_on_the_prefix(self):
        results = [
            _make_fold_result(0, 5.0, 2.0, 0.5, features_long=["f"]),
            _make_fold_result(1, 5.0, 2.0, 0.5),
            _make_fold_result(2, 5.0, 2.0, 0.5),
        ]
        candidates = causal_candidates(WF_FOLDS[:3], results)
        assert [c["selected_features_long"] for c in candidates] == [["f"], ["f"], []]

    def test_hp_majority_on_the_prefix(self):
        results = [
            _make_fold_result(0, 5.0, 2.0, 0.5, model_hyperparameters={"a": 1}),
            _make_fold_result(1, 5.0, 2.0, 0.5, model_hyperparameters={"a": 2}),
            _make_fold_result(2, 5.0, 2.0, 0.5, model_hyperparameters={"a": 2}),
        ]
        candidates = causal_candidates(WF_FOLDS[:3], results)
        assert candidates[0]["model_hyperparameters"] == {"a": 1}
        assert candidates[2]["model_hyperparameters"] == {"a": 2}

    def test_prefix_order_follows_fold_id_not_arrival(self):
        results = [
            _make_fold_result(i, tp=tp, sl=2.0, ct=0.5,
                              model_hyperparameters={"a": hp}, features_long=feats)
            for i, (tp, hp, feats) in enumerate(
                [(1.0, 1, ["x"]), (3.0, 2, ["y", "x"]), (5.0, 2, ["y"]), (7.0, 1, ["z"])])
        ]
        shuffled = [results[3], results[1], results[2], results[0]]
        assert causal_candidates(WF_FOLDS, shuffled) == causal_candidates(WF_FOLDS, results)

    def test_last_candidate_is_the_full_merge(self):
        results = [
            _make_fold_result(i, tp=tp, sl=2.0, ct=0.5,
                              model_hyperparameters={"a": hp},
                              features_long=feats)
            for i, (tp, hp, feats) in enumerate(
                [(1.0, 1, ["x"]), (3.0, 2, ["x", "y"]), (5.0, 2, []), (7.0, 1, ["y"])])
        ]
        candidates = causal_candidates(WF_FOLDS, results)
        assert candidates[-1] == merge_unified_settings(consistent_folds(results), results)


# ---------------------------------------------------------------------------
# run_unified_simulation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Ctx:
    model_hyperparameters: dict = dataclasses.field(default_factory=dict)
    exit_modifier_params: dict = dataclasses.field(default_factory=dict)
    exit_strategy: str = "fixed"
    exit_params: dict = dataclasses.field(default_factory=dict)


def _candidate(tp, model_hp=None, exit_mod=None, exit_strategy=None, exit_params=None):
    return {
        "params": (tp, 2.0, 0.5),
        "timeout_bars": None,
        "selected_features_long": ["f1"],
        "selected_features_short": [],
        "model_hyperparameters": model_hp,
        "exit_modifier_params": exit_mod,
        "exit_strategy": exit_strategy,
        "exit_params": exit_params,
    }


def _prepare(fold, *args):
    frame = pd.DataFrame({"f1": [0.1, 0.2]})
    return frame, frame, ["f1"]


def _simulate(folds, candidates, prepare=_prepare):
    return run_unified_simulation(
        folds, candidates, None, None, [], _Ctx(), "EURUSD", prepare_data_fn=prepare,
    )


class TestRunUnifiedSimulation:

    def test_fold_without_setting_is_not_traded(self):
        def prepare(*args):
            pytest.fail("an untraded fold prepares no data")

        results = _simulate(WF_FOLDS[:2], [None, None], prepare)
        assert results == [
            {"trades": [], "trades_detailed": [], "pnl": 0.0, "win_rate": 0.0,
             "n_trades": 0, "dropped_features": [], "fold_id": fold.fold_id}
            for fold in WF_FOLDS[:2]
        ]

    def test_one_candidate_per_fold_required(self):
        with pytest.raises(ValueError):
            _simulate(WF_FOLDS[:2], [_candidate(1.0)])

    def test_each_fold_replayed_with_its_own_candidate(self, monkeypatch):
        calls = []

        def evaluate(test_df, train_df, candidate, ctx):
            calls.append((candidate, ctx))
            return {"trades": [], "n_trades": 0, "pnl": 0.0, "win_rate": 0.0}

        monkeypatch.setattr(unified_simulation, "evaluate_on_holdout", evaluate)
        first = _candidate(1.0, model_hp={"a": 1}, exit_mod={"be": 0.5},
                           exit_strategy="orb_based", exit_params={"col": "orb"})
        second = _candidate(3.0)
        _simulate(WF_FOLDS[:2], [first, second])

        (candidate0, ctx0), (candidate1, ctx1) = calls
        assert candidate0 == first and candidate1 == second
        assert ctx0 == _Ctx(model_hyperparameters={"a": 1}, exit_modifier_params={"be": 0.5},
                            exit_strategy="orb_based", exit_params={"col": "orb"})
        assert ctx1 == _Ctx()

    def test_candidate_exit_params_override_same_exit_strategy(self, monkeypatch):
        calls = []

        def evaluate(test_df, train_df, candidate, ctx):
            calls.append(ctx)
            return {"trades": [], "n_trades": 0, "pnl": 0.0, "win_rate": 0.0}

        monkeypatch.setattr(unified_simulation, "evaluate_on_holdout", evaluate)
        candidate = _candidate(
            1.0,
            exit_strategy="orb_based",
            exit_params={"range_column": "_orb_range_new"},
        )
        base_ctx = _Ctx(
            exit_strategy="orb_based",
            exit_params={"range_column": "_orb_range_old"},
        )
        run_unified_simulation(
            WF_FOLDS[:1],
            [candidate],
            None,
            None,
            [],
            base_ctx,
            "EURUSD",
            prepare_data_fn=_prepare,
        )

        assert calls[0].exit_strategy == "orb_based"
        assert calls[0].exit_params == {"range_column": "_orb_range_new"}

    def test_each_traded_fold_carries_its_candidates_rrr(self, monkeypatch):
        monkeypatch.setattr(
            unified_simulation, "evaluate_on_holdout",
            lambda *args: {"trades": [], "n_trades": 0, "pnl": 0.0, "win_rate": 0.0},
        )
        no_sl = {**_candidate(2.0), "params": (2.0, 0.0, 0.5)}
        results = _simulate(WF_FOLDS[:4], [None, _candidate(1.0), _candidate(3.0), no_sl])
        assert "rrr" not in results[0]
        assert [r["rrr"] for r in results[1:]] == [0.5, 1.5, 1.0]

    def test_every_result_carries_its_fold_id(self, monkeypatch):
        monkeypatch.setattr(
            unified_simulation, "evaluate_on_holdout",
            lambda *args: {"trades": [], "n_trades": 0, "pnl": 0.0, "win_rate": 0.0},
        )
        folds = [WF_FOLDS[2], WF_FOLDS[0], WF_FOLDS[3]]
        results = _simulate(folds, [_candidate(1.0), None, _candidate(2.0)])
        assert [r["fold_id"] for r in results] == [2, 0, 3]
