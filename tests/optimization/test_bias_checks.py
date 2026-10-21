"""Test/inner bias ratios when a fold's chosen setting has an inner PnL <= 0."""

import json

import pytest

from fwbg.optimization import bias_checks
from fwbg.optimization.bias_checks import (
    check_asset_bias,
    check_systematic_bias,
    fold_bias_ratios,
)
from tests.optimization import symbol_harness
from tests.optimization.symbol_harness import WINNING, patch_development, run_inline


def _fold(inner, test):
    return {"inner_val_pnl": inner, "test_pnl": test}


def _result(bias_ratios, mean_bias_ratio, without):
    return {
        "symbol": "TEST",
        "status": "ok",
        "walk_forward": {
            "n_folds": 6,
            "std_win_rate": 0.0,
            "total_trades": 1000,
            "bias_ratios": bias_ratios,
            "mean_bias_ratio": mean_bias_ratio,
            "folds_without_bias_ratio": without,
        },
    }


def test_ratio_only_where_the_inner_pnl_is_positive():
    bias = fold_bias_ratios([_fold(5.0, 10.0), _fold(0.0, 3.0), _fold(-2.0, -6.0)])
    # -6 / -2 = 3 would read as bias, 3 / 0 is undefined: both are left out.
    assert bias == {
        "bias_ratios": [2.0],
        "mean_bias_ratio": 2.0,
        "folds_without_bias_ratio": 2,
    }


def test_a_near_zero_inner_pnl_does_not_dominate_the_mean():
    """The mean weights each fold's ratio by its inner PnL: sum(test) / sum(inner)."""
    bias = fold_bias_ratios([_fold(10.0, 10.0), _fold(10.0, 10.0), _fold(0.1, 5.0)])
    assert bias["bias_ratios"] == [1.0, 1.0, 50.0]  # the fold itself still shows its ratio
    assert bias["mean_bias_ratio"] == pytest.approx(25.0 / 20.1)
    check = check_asset_bias(
        _result(bias["bias_ratios"], bias["mean_bias_ratio"], 0), verbose=False
    )
    assert check["bias_check"] == "warning"  # the >2x fold is a warning, not CRITICAL


def test_a_negative_test_pnl_keeps_its_ratio():
    assert fold_bias_ratios([_fold(4.0, -2.0)])["bias_ratios"] == [-0.5]


def test_no_positive_inner_pnl_leaves_the_mean_undefined():
    bias = fold_bias_ratios([_fold(-1.0, 4.0), _fold(0.0, 0.0)])
    assert bias == {"bias_ratios": [], "mean_bias_ratio": None, "folds_without_bias_ratio": 2}


def test_non_finite_pnl_is_excluded_from_bias_ratios():
    bias = fold_bias_ratios(
        [_fold(float("nan"), 1.0), _fold(1.0, float("inf")), _fold(2.0, 4.0)]
    )
    assert bias == {
        "bias_ratios": [2.0],
        "mean_bias_ratio": 2.0,
        "folds_without_bias_ratio": 2,
    }


def test_folds_without_a_ratio_are_a_warning_not_ok():
    check = check_asset_bias(_result([1.0], 1.0, 2), verbose=False)
    assert check["bias_check"] == "warning"
    assert "2/6 folds without bias ratio (inner PnL <= 0)" in check["warnings"]
    assert check["folds_without_bias_ratio"] == 2


def test_an_undefined_mean_is_checked_without_failing():
    check = check_asset_bias(_result([], None, 6), verbose=True)
    assert check["bias_check"] == "warning"
    assert check["mean_bias_ratio"] is None


def test_results_without_the_count_still_check_ok():
    result = _result([1.0] * 6, 1.0, 0)
    del result["walk_forward"]["folds_without_bias_ratio"]
    assert check_asset_bias(result, verbose=False)["bias_check"] == "ok"


def test_systematic_check_does_not_count_an_undefined_mean_as_ok():
    check = check_systematic_bias([_result([], None, 6)], verbose=True)
    assert check["ok_assets"] == 0
    assert check["warned_assets"] == 1


def test_systematic_bias_uses_only_measurable_assets_and_warns_on_unchecked():
    results = [_result([], 2.0, 0)] + [_result([], None, 6) for _ in range(9)]
    check = check_systematic_bias(results, verbose=False)
    assert check["status"] == "systematic_bias"
    assert check["bias_percentage"] == 1.0
    assert check["measurable_assets"] == 1
    assert check["unchecked_assets"] == 9


def test_a_run_with_inner_pnl_at_or_below_zero_records_no_ratio(monkeypatch, tmp_path):
    canned = symbol_harness.fold_result

    def losing_inner(fold):
        return {**canned(fold), "inner_val_pnl": -4.0 if fold.fold_id % 2 else 0.0}

    monkeypatch.setattr(symbol_harness, "fold_result", losing_inner)
    patch_development(monkeypatch, WINNING)
    sym_dir = run_inline(monkeypatch, tmp_path) / "grid_details" / "EURUSD"
    walk_forward = json.loads((sym_dir / "fold_results.json").read_text())["walk_forward"]
    assert walk_forward["bias_ratios"] == []
    assert walk_forward["mean_bias_ratio"] is None
    assert walk_forward["folds_without_bias_ratio"] == walk_forward["n_folds"] > 0
    assert walk_forward["sample_bias_detected"] is False


def test_an_undefined_mean_prints_n_a_on_the_ok_branch(capsys):
    check_asset_bias(_result([], None, 0), verbose=True)
    assert "No bias detected (mean=n/a" in capsys.readouterr().out


def test_fold_bias_ratios_is_exported():
    assert "fold_bias_ratios" in bias_checks.__all__
