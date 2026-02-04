#!/usr/bin/env python3
"""
Einfacher Test: Reproduziert erfolgreiche Parameter aus Run 20260201_014605_c5f7f5

Testet SPX500 mit: tp=10, sl=30, ct=0.65, timeout=None, feature_group=macro_vol
Expected inner_val_pnl: 383.6
"""

import sys
from pathlib import Path

# Füge src zum Python-Path hinzu
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fwbg.core.context import SimulationContext
from fwbg.optimization.nested_cv import nested_cv_split, run_inner_cv
from fwbg.data.assets import get_asset
from fwbg.data.loader import load_data_aligned
from fwbg.core.features import compute_features
from fwbg.core.indicators import get_features_by_group


def main():
    symbol = "SPX500"
    tp = 10
    sl = 30
    ct = 0.65
    timeout_bars = None
    feature_group = "macro_vol"
    expected_pnl = 383.6

    print(f"="*80)
    print(f"Testing {symbol}")
    print(f"  TP={tp}, SL={sl}, CT={ct}, Timeout={timeout_bars}")
    print(f"  Feature Group: {feature_group}")
    print(f"  Expected Inner Val PnL: {expected_pnl}")
    print(f"="*80)

    # Lade Asset-Config
    asset = get_asset(symbol)
    if not asset:
        print(f"❌ Asset {symbol} not found")
        return

    csv_path = asset.csv_path
    print(f"\nLoading data from: {csv_path}")

    # Lade Daten
    df = load_data_aligned(csv_path)
    if df is None or len(df) < 10000:
        print(f"❌ Insufficient data for {symbol}")
        return

    print(f"Loaded {len(df)} bars")

    # Erstelle SimulationContext
    ctx = SimulationContext(
        symbol=symbol,
        timeframe="HOUR",
        spread=asset.spread,
        min_trades=50,
        long_enabled=True,
        short_enabled=True,
        confidence_threshold=ct,
        max_trade_bars=200,
        regime_filter=None,
        feature_selection="boruta",
        max_features=50,

        # Early Termination Settings - TEST MIT ALTEN WERTEN
        early_termination=True,  # Basis-Check an
        min_fold_stability=0.5,

        # First-Fold Sanity Check
        first_fold_sanity_check=True,
        first_fold_min_win_rate=0.25,
        first_fold_min_pnl=-10.0,
        first_fold_min_trades=5,

        # Exit Strategy
        exit_strategy_mode="fixed",
        exit_params={},

        # XGBoost config
        xgb_n_jobs=1,  # Single-threaded für Test
    )

    # Compute Features
    print("\nComputing features...")
    df_with_features = compute_features(
        df,
        symbol=symbol,
        timeframe="HOUR",
        config={
            "macro_indicators": [
                "VIX_DAY", "VVIX_DAY", "SKEW_DAY", "VXN_DAY",
                "TNX_DAY", "TYX_DAY", "FVX_DAY", "IRX_DAY",
                "DXY_DAY", "GOLD_FUT_DAY", "OIL_FUT_DAY", "SILVER_FUT_DAY",
                "SPX_DAY", "NASDAQ_DAY", "DOW_DAY", "RUSSELL_DAY",
                "NIKKEI_DAY", "HANGSENG_DAY", "FTSE_DAY", "DAX_IDX_DAY",
                "XLF_DAY", "XLE_DAY", "XLK_DAY", "XLU_DAY", "XLP_DAY",
                "TLT_DAY", "HYG_DAY", "LQD_DAY"
            ],
            "lookbacks_hours": [1, 2, 4, 8, 12, 24],
            "lookbacks_days": [2, 5, 10, 20, 60]
        }
    )

    if df_with_features is None or df_with_features.empty:
        print(f"❌ Feature computation failed")
        return

    print(f"Computed features, df now has {len(df_with_features.columns)} columns")

    # Filter Features by Group
    group_features = get_features_by_group(df_with_features.columns, feature_group)
    print(f"Features in group '{feature_group}': {len(group_features)}")

    if not group_features:
        print(f"❌ No features found for group {feature_group}")
        return

    # Walk-Forward Split
    print("\nCreating walk-forward folds...")
    oos_folds = nested_cv_split(
        df_with_features,
        n_folds=8,
        oos_size=4000
    )

    # Teste nur den ersten OOS-Fold
    inner_df, oos_df, inner_folds = oos_folds[0]

    print(f"  Inner CV folds: {len(inner_folds)}")
    print(f"  Inner data: {len(inner_df)} bars")
    print(f"  OOS data: {len(oos_df)} bars")

    # Run Inner CV
    print("\nRunning Inner CV with OLD early termination logic...")
    result = run_inner_cv(
        inner_folds=inner_folds,
        inner_df=inner_df,
        tp=tp,
        sl=sl,
        ctx=ctx,
        timeout_bars=timeout_bars,
        group_features=group_features,
        global_grid_pos=1,
        total_grid_combos=1,
        cached_targets=None
    )

    print(f"\n{'='*80}")
    print("RESULT:")
    print(f"{'='*80}")

    if result.get("success", False):
        inner_val_pnl = result.get("inner_val_pnl", 0)
        print(f"✅ SUCCESS")
        print(f"  Inner Val PnL: {inner_val_pnl:.1f}")
        print(f"  Expected PnL: {expected_pnl:.1f}")

        diff = abs(inner_val_pnl - expected_pnl)
        diff_pct = diff / expected_pnl * 100
        print(f"  Difference: {diff:.1f} ({diff_pct:.1f}%)")

        if diff_pct < 20:
            print(f"  ✅ Within 20% tolerance")
        else:
            print(f"  ⚠️  Outside 20% tolerance")
    else:
        print(f"❌ FAILED")
        if result.get("early_terminated", False):
            print(f"  Reason: Early terminated")
            print(f"  Failed folds: {result.get('failed_folds', 0)}")
        print(f"  Full result: {result}")


if __name__ == "__main__":
    main()
