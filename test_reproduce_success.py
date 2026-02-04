#!/usr/bin/env python3
"""
Test-Script: Reproduziert erfolgreiche Parameter aus Run 20260201_014605_c5f7f5

Testet, ob die gleichen Parameter, die im erfolgreichen Run gefunden wurden,
auch mit dem aktuellen Code noch gefunden werden.

Erfolgreiche Parameter:
- NAS100:  tp=10, sl=50, ct=0.6,  timeout=12,   feature_group=macro_vol
- DOW30:   tp=10, sl=50, ct=0.65, timeout=12,   feature_group=macro_vol
- SPX500:  tp=10, sl=30, ct=0.65, timeout=None, feature_group=macro_vol
- FTSE100: tp=10, sl=30, ct=0.65, timeout=None, feature_group=macro_vol
"""

import sys
import json
import pandas as pd
from pathlib import Path

# Füge src zum Python-Path hinzu
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fwbg.core.context import SimulationContext
from fwbg.optimization.nested_cv import nested_cv_split, run_inner_cv
from fwbg.adapters.data import DataAdapter
from fwbg.core.features import compute_all_features
from fwbg.core.indicators import filter_features_by_group

# Test-Parameter aus erfolgreichem Run
TEST_CASES = [
    {
        "symbol": "SPX500",
        "tp": 10,
        "sl": 30,
        "ct": 0.65,
        "timeout_bars": None,
        "feature_group": "macro_vol",
        "expected_inner_val_pnl": 383.6,
    },
    {
        "symbol": "NAS100",
        "tp": 10,
        "sl": 50,
        "ct": 0.6,
        "timeout_bars": 12,
        "feature_group": "macro_vol",
        "expected_inner_val_pnl": 678.0,
    },
]


def test_parameter_combination(test_case: dict, enable_new_early_termination: bool = True):
    """
    Testet eine spezifische Parameter-Kombination

    Args:
        test_case: Dict mit symbol, tp, sl, ct, timeout_bars, feature_group
        enable_new_early_termination: Wenn False, deaktiviert die neuen Early-Termination-Checks
    """
    symbol = test_case["symbol"]
    print(f"\n{'='*80}")
    print(f"Testing {symbol}")
    print(f"  Parameters: TP={test_case['tp']}, SL={test_case['sl']}, CT={test_case['ct']}, Timeout={test_case['timeout_bars']}")
    print(f"  Feature Group: {test_case['feature_group']}")
    print(f"  New Early Termination: {enable_new_early_termination}")
    print(f"{'='*80}")

    # Lade Daten
    print(f"Loading data for {symbol}...")
    df = load_data_for_symbol(symbol, "HOUR")
    if df is None or len(df) < 10000:
        print(f"❌ Insufficient data for {symbol}")
        return None

    # Erstelle SimulationContext (wie im Optimizer)
    ctx = SimulationContext(
        symbol=symbol,
        timeframe="HOUR",
        spread=0.00020,  # Standard-Spread für INDEX
        min_trades=50,
        long_enabled=True,
        short_enabled=True,
        confidence_threshold=test_case["ct"],
        max_trade_bars=200,
        regime_filter=None,
        feature_selection="boruta",
        max_features=50,

        # Early Termination Settings
        early_termination=enable_new_early_termination,  # Hier können wir testen!
        min_fold_stability=0.5,

        # First-Fold Sanity Check
        first_fold_sanity_check=True,
        first_fold_min_win_rate=0.25,
        first_fold_min_pnl=-10.0,
        first_fold_min_trades=5,

        # Exit Strategy (fixed wie im erfolgreichen Run)
        exit_strategy_mode="fixed",
        exit_params={},
    )

    # Compute Features
    print("Computing features...")
    df_with_features = compute_all_features(
        df,
        symbol=symbol,
        timeframe="HOUR",
        macro_indicators=[
            "VIX_DAY", "VVIX_DAY", "SKEW_DAY", "VXN_DAY",
            "TNX_DAY", "TYX_DAY", "FVX_DAY", "IRX_DAY",
            "DXY_DAY", "GOLD_FUT_DAY", "OIL_FUT_DAY", "SILVER_FUT_DAY",
            "SPX_DAY", "NASDAQ_DAY", "DOW_DAY", "RUSSELL_DAY",
            "NIKKEI_DAY", "HANGSENG_DAY", "FTSE_DAY", "DAX_IDX_DAY",
            "XLF_DAY", "XLE_DAY", "XLK_DAY", "XLU_DAY", "XLP_DAY",
            "TLT_DAY", "HYG_DAY", "LQD_DAY"
        ],
        lookbacks_hours=[1, 2, 4, 8, 12, 24],
        lookbacks_days=[2, 5, 10, 20, 60]
    )

    if df_with_features is None or df_with_features.empty:
        print(f"❌ Feature computation failed for {symbol}")
        return None

    # Filter Features by Group
    group_features = filter_features_by_group(df_with_features.columns, test_case["feature_group"])
    print(f"Features in group '{test_case['feature_group']}': {len(group_features)}")

    if not group_features:
        print(f"❌ No features found for group {test_case['feature_group']}")
        return None

    # Walk-Forward Split
    print("Creating walk-forward folds...")
    oos_folds = nested_cv_split(
        df_with_features,
        n_folds=8,
        oos_size=4000
    )

    # Teste nur den ersten OOS-Fold (für schnellere Tests)
    oos_idx = 0
    inner_df, oos_df, inner_folds = oos_folds[oos_idx]

    print(f"  Inner CV folds: {len(inner_folds)}")
    print(f"  Inner data: {len(inner_df)} bars")
    print(f"  OOS data: {len(oos_df)} bars")

    # Run Inner CV mit den spezifischen Parametern
    print("\nRunning Inner CV...")
    result = run_inner_cv(
        inner_folds=inner_folds,
        inner_df=inner_df,
        tp=test_case["tp"],
        sl=test_case["sl"],
        ctx=ctx,
        timeout_bars=test_case["timeout_bars"],
        group_features=group_features,
        global_grid_pos=1,
        total_grid_combos=1,
        cached_targets=None
    )

    print(f"\n{'='*80}")
    print("RESULT:")
    print(f"{'='*80}")
    print(json.dumps(result, indent=2))

    # Vergleich mit erwartetem Ergebnis
    if result.get("success", False):
        inner_val_pnl = result.get("inner_val_pnl", 0)
        expected_pnl = test_case["expected_inner_val_pnl"]

        print(f"\n✅ SUCCESS")
        print(f"  Inner Val PnL: {inner_val_pnl:.1f} (Expected: {expected_pnl:.1f})")

        # Toleranz von 20% für Vergleich (wegen möglicher Unterschiede in Daten/Random-State)
        diff_pct = abs(inner_val_pnl - expected_pnl) / expected_pnl * 100
        if diff_pct < 20:
            print(f"  ✅ Within 20% tolerance (diff: {diff_pct:.1f}%)")
        else:
            print(f"  ⚠️  Outside tolerance (diff: {diff_pct:.1f}%)")
    else:
        print(f"\n❌ FAILED")
        if result.get("early_terminated", False):
            print(f"  Reason: Early terminated after {result.get('failed_folds', 0)} failed folds")
        else:
            print(f"  Reason: {result}")

    return result


def main():
    """Führt alle Tests durch"""
    print("="*80)
    print("REPRODUKTION TEST - Erfolgreiche Parameter aus 20260201_014605_c5f7f5")
    print("="*80)

    results = {}

    for test_case in TEST_CASES:
        symbol = test_case["symbol"]

        # Test 1: Mit neuen Early Termination Checks (aktueller Code)
        print(f"\n\n### TEST 1: {symbol} - WITH new early termination ###")
        result_with_new_et = test_parameter_combination(test_case, enable_new_early_termination=True)

        # Test 2: Ohne neue Early Termination Checks (wie alter Code)
        print(f"\n\n### TEST 2: {symbol} - WITHOUT new early termination ###")
        result_without_new_et = test_parameter_combination(test_case, enable_new_early_termination=False)

        results[symbol] = {
            "with_new_et": result_with_new_et,
            "without_new_et": result_without_new_et
        }

    # Zusammenfassung
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for symbol, res in results.items():
        print(f"\n{symbol}:")

        with_et = res["with_new_et"]
        without_et = res["without_new_et"]

        if with_et and with_et.get("success"):
            print(f"  ✅ WITH new early termination: SUCCESS (PnL: {with_et.get('inner_val_pnl', 0):.1f})")
        else:
            reason = "early_terminated" if with_et and with_et.get("early_terminated") else "failed"
            print(f"  ❌ WITH new early termination: FAILED ({reason})")

        if without_et and without_et.get("success"):
            print(f"  ✅ WITHOUT new early termination: SUCCESS (PnL: {without_et.get('inner_val_pnl', 0):.1f})")
        else:
            reason = "early_terminated" if without_et and without_et.get("early_terminated") else "failed"
            print(f"  ❌ WITHOUT new early termination: FAILED ({reason})")


if __name__ == "__main__":
    main()
