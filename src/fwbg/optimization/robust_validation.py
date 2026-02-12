"""
Robust Walk-Forward Validation Framework

WICHTIG: Dieses Framework DETEKTIERT Sample Bias für DIAGNOSTIK.
Wenn mehrere Assets biased erscheinen, ist das ein SYSTEMATISCHES
Problem im Code, nicht ein Problem einzelner Assets.

Features:
- Multiple Walk-Forward Perioden (nicht nur 1x Holdout)
- Performance-Aggregation über alle Perioden
- Robustheit-Metriken (Std-Dev, Worst-Case)
- Minimum-Trades Enforcement
- Sample-Bias Detection (für Diagnose, nicht Auto-Filter)
"""
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass

from fwbg.optimization.nested_cv import (
    nested_cv_split,
    run_inner_cv,
    evaluate_on_holdout,
)
from fwbg.optimization.targets import compute_targets
from fwbg.core.context import SimulationContext


@dataclass
class WalkForwardFold:
    """Ein einzelner Walk-Forward Fold."""
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass
class RobustValidationResult:
    """Ergebnis der robusten Validierung über multiple Folds."""
    # Aggregierte Metriken
    mean_win_rate: float
    std_win_rate: float
    min_win_rate: float
    max_win_rate: float

    mean_pnl: float
    std_pnl: float
    min_pnl: float
    max_pnl: float

    total_trades: int
    mean_trades_per_fold: float

    # Robustheit-Metriken
    consistency_score: float  # 0-1, höher = konsistenter
    worst_case_pnl: float

    # Sample Bias Detection
    sample_bias_detected: bool
    holdout_vs_inner_ratios: List[float]

    # Per-Fold Details
    fold_results: List[Dict[str, Any]]

    # Config
    selected_config: Dict[str, Any]

    def is_robust(
        self,
        min_total_trades: int = 500,
        max_win_rate_std: float = 0.15,
        max_bias_ratio: float = 2.0,
    ) -> bool:
        """
        Prüft ob die Config robust ist (für DIAGNOSTIK, nicht Auto-Filter).

        WICHTIG: Wenn viele Configs nicht robust sind, deutet das auf
        ein systematisches Problem im Code hin, nicht auf einzelne
        problematische Assets.

        Args:
            min_total_trades: Minimum Trades über alle Folds
            max_win_rate_std: Maximum Std-Dev der Win-Rate
            max_bias_ratio: Maximum Holdout/Inner Ratio
        """
        # Check 1: Genug Trades
        if self.total_trades < min_total_trades:
            return False

        # Check 2: Win-Rate konsistent
        if self.std_win_rate > max_win_rate_std:
            return False

        # Check 3: Kein extremer Sample Bias
        max_ratio = max(self.holdout_vs_inner_ratios) if self.holdout_vs_inner_ratios else 0
        if max_ratio > max_bias_ratio:
            return False

        return True


def create_walk_forward_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    test_size: int = 4000,
    min_train_size: int = 20000,
    anchored: bool = True,
) -> List[WalkForwardFold]:
    """
    Erstellt Walk-Forward Folds.

    Args:
        df: Vollständiger DataFrame
        n_folds: Anzahl der Walk-Forward Folds
        test_size: Größe jedes Test-Sets
        min_train_size: Minimum Training-Daten
        anchored: True = Training wächst, False = Rolling Window

    Returns:
        Liste von WalkForwardFold Objekten
    """
    total_len = len(df)

    # Reserve space for test folds
    available_for_train = total_len - (n_folds * test_size)

    if available_for_train < min_train_size:
        raise ValueError(
            f"Not enough data for {n_folds} folds. "
            f"Need {min_train_size + n_folds * test_size} bars, have {total_len}"
        )

    folds = []

    for fold_id in range(n_folds):
        # Test set: Move forward each fold
        test_start = available_for_train + (fold_id * test_size)
        test_end = test_start + test_size

        if anchored:
            # Anchored: Training grows from start
            train_start = 0
            train_end = test_start
        else:
            # Rolling: Training window slides
            train_size = test_start // (fold_id + 1)  # Shrink training as we go
            train_start = max(0, test_start - train_size)
            train_end = test_start

        # Ensure minimum training size
        if train_end - train_start < min_train_size:
            train_start = max(0, train_end - min_train_size)

        fold = WalkForwardFold(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_df=df.iloc[train_start:train_end].copy(),
            test_df=df.iloc[test_start:test_end].copy(),
        )
        folds.append(fold)

    return folds


def run_robust_validation(
    df: pd.DataFrame,
    strategy_config: Dict[str, Any],
    ctx: SimulationContext,
    n_walk_forward_folds: int = 5,
    verbose: bool = True,
) -> RobustValidationResult:
    """
    Führt robuste Walk-Forward Validierung durch.

    Verhindert Sample Bias durch:
    - Multiple Test-Perioden
    - Performance-Aggregation
    - Robustheit-Checks

    Args:
        df: DataFrame mit Features (bereits berechnet)
        strategy_config: Grid-Config (tp, sl, ct)
        ctx: SimulationContext
        n_walk_forward_folds: Anzahl Walk-Forward Folds
        verbose: Print Progress

    Returns:
        RobustValidationResult mit aggregierten Metriken
    """
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"ROBUST WALK-FORWARD VALIDATION")
        print(f"{'=' * 80}")
        print(f"Asset: {ctx.symbol}")
        print(f"Walk-Forward Folds: {n_walk_forward_folds}")
        print(f"Data: {len(df)} bars")

    # Create Walk-Forward Folds
    try:
        wf_folds = create_walk_forward_folds(
            df,
            n_folds=n_walk_forward_folds,
            test_size=4000,
            min_train_size=20000,
            anchored=True,
        )
    except ValueError as e:
        if verbose:
            print(f"ERROR: {e}")
        # Return empty result
        return RobustValidationResult(
            mean_win_rate=0.0,
            std_win_rate=0.0,
            min_win_rate=0.0,
            max_win_rate=0.0,
            mean_pnl=0.0,
            std_pnl=0.0,
            min_pnl=0.0,
            max_pnl=0.0,
            total_trades=0,
            mean_trades_per_fold=0.0,
            consistency_score=0.0,
            worst_case_pnl=0.0,
            sample_bias_detected=True,
            holdout_vs_inner_ratios=[],
            fold_results=[],
            selected_config={},
        )

    if verbose:
        print(f"\nCreated {len(wf_folds)} walk-forward folds:")
        for fold in wf_folds:
            print(f"  Fold {fold.fold_id}: "
                  f"Train[{fold.train_start}:{fold.train_end}] ({fold.train_end - fold.train_start} bars), "
                  f"Test[{fold.test_start}:{fold.test_end}] ({fold.test_end - fold.test_start} bars)")

    # Run optimization on each fold
    fold_results = []

    for fold in wf_folds:
        if verbose:
            print(f"\n{'-' * 80}")
            print(f"Processing Fold {fold.fold_id}...")
            print(f"{'-' * 80}")

        # Split train into inner CV folds
        cv_split = nested_cv_split(
            fold.train_df,
            holdout_ratio=0.0,  # No holdout within fold (we use test_df as holdout)
            n_inner_folds=3,  # Reduce folds for speed
            oos_size=4000,
        )

        inner_folds = cv_split["inner_folds"]

        # Run grid search (simplified - single config for now)
        # In production, iterate over strategy_config grid
        tp = strategy_config.get("tp", 10)
        sl = strategy_config.get("sl", 20)

        # Get feature columns
        feature_cols = [
            c for c in fold.train_df.columns
            if c not in ['O', 'H', 'L', 'C', 'V', '_atr', '_regime_ok']
            and not c.startswith('_')
        ]

        # Run inner CV
        inner_result = run_inner_cv(
            inner_folds=inner_folds,
            group_features=feature_cols[:50],  # Limit features for speed
            tp=tp,
            sl=sl,
            ctx=ctx,
            global_grid_pos=0,
            total_grid_combos=1,
            timeout_bars=None,
            cached_targets=None,
        )

        if not inner_result.get("success"):
            if verbose:
                print(f"  Fold {fold.fold_id}: Inner CV failed, skipping")
            continue

        inner_val_pnl = inner_result.get("avg_val_pnl", 0)
        best_ct = inner_result.get("best_ct", 0.6)
        selected_features_long = inner_result.get("selected_features_long", [])
        selected_features_short = inner_result.get("selected_features_short", [])

        # Evaluate on test fold (our "holdout")
        candidate = {
            "params": (tp, sl, best_ct),
            "timeout_bars": None,
            "selected_features_long": selected_features_long,
            "selected_features_short": selected_features_short,
        }

        holdout_result = evaluate_on_holdout(
            holdout_df=fold.test_df,
            inner_df=fold.train_df,
            candidate=candidate,
            ctx=ctx,
        )

        holdout_pnl = holdout_result.get("pnl", 0)
        holdout_win_rate = holdout_result.get("win_rate", 0)
        holdout_trades = holdout_result.get("n_trades", 0)

        # Calculate bias ratio
        bias_ratio = holdout_pnl / inner_val_pnl if inner_val_pnl > 0 else 0

        fold_result = {
            "fold_id": fold.fold_id,
            "inner_val_pnl": inner_val_pnl,
            "holdout_pnl": holdout_pnl,
            "holdout_win_rate": holdout_win_rate,
            "holdout_trades": holdout_trades,
            "bias_ratio": bias_ratio,
        }
        fold_results.append(fold_result)

        if verbose:
            print(f"  Fold {fold.fold_id} Results:")
            print(f"    Inner Val PnL: {inner_val_pnl:.1f}")
            print(f"    Holdout PnL: {holdout_pnl:.1f}")
            print(f"    Holdout Win Rate: {holdout_win_rate*100:.1f}%")
            print(f"    Holdout Trades: {holdout_trades}")
            print(f"    Bias Ratio: {bias_ratio:.2f}x")

    # Aggregate results
    if len(fold_results) == 0:
        if verbose:
            print(f"\nERROR: No successful folds")
        return RobustValidationResult(
            mean_win_rate=0.0,
            std_win_rate=0.0,
            min_win_rate=0.0,
            max_win_rate=0.0,
            mean_pnl=0.0,
            std_pnl=0.0,
            min_pnl=0.0,
            max_pnl=0.0,
            total_trades=0,
            mean_trades_per_fold=0.0,
            consistency_score=0.0,
            worst_case_pnl=0.0,
            sample_bias_detected=True,
            holdout_vs_inner_ratios=[],
            fold_results=[],
            selected_config={},
        )

    win_rates = [r["holdout_win_rate"] for r in fold_results]
    pnls = [r["holdout_pnl"] for r in fold_results]
    trades = [r["holdout_trades"] for r in fold_results]
    bias_ratios = [r["bias_ratio"] for r in fold_results if r["bias_ratio"] > 0]

    # Detect sample bias: If ANY fold has extreme bias
    sample_bias = any(ratio > 2.0 for ratio in bias_ratios)

    # Consistency score: 1 - coefficient of variation
    mean_wr = np.mean(win_rates)
    std_wr = np.std(win_rates)
    consistency = 1.0 - (std_wr / mean_wr if mean_wr > 0 else 1.0)
    consistency = max(0.0, min(1.0, consistency))  # Clip to [0, 1]

    result = RobustValidationResult(
        mean_win_rate=mean_wr,
        std_win_rate=std_wr,
        min_win_rate=min(win_rates),
        max_win_rate=max(win_rates),
        mean_pnl=np.mean(pnls),
        std_pnl=np.std(pnls),
        min_pnl=min(pnls),
        max_pnl=max(pnls),
        total_trades=sum(trades),
        mean_trades_per_fold=np.mean(trades),
        consistency_score=consistency,
        worst_case_pnl=min(pnls),
        sample_bias_detected=sample_bias,
        holdout_vs_inner_ratios=bias_ratios,
        fold_results=fold_results,
        selected_config=strategy_config,
    )

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"AGGREGATED RESULTS OVER {len(fold_results)} FOLDS")
        print(f"{'=' * 80}")
        print(f"Win Rate: {result.mean_win_rate*100:.1f}% ± {result.std_win_rate*100:.1f}%")
        print(f"  Range: [{result.min_win_rate*100:.1f}% - {result.max_win_rate*100:.1f}%]")
        print(f"PnL: {result.mean_pnl:.1f} ± {result.std_pnl:.1f}")
        print(f"  Range: [{result.min_pnl:.1f} - {result.max_pnl:.1f}]")
        print(f"Total Trades: {result.total_trades}")
        print(f"Consistency Score: {result.consistency_score:.2f} (1.0 = perfect)")
        print(f"Worst Case PnL: {result.worst_case_pnl:.1f}")
        print()

        if result.sample_bias_detected:
            print(f"⚠️  WARNING: Sample bias detected!")
            print(f"   Some folds have Holdout/Inner ratio > 2.0x")
            print(f"   Ratios: {[f'{r:.2f}x' for r in result.holdout_vs_inner_ratios]}")
        else:
            print(f"✓ No extreme sample bias detected")

        print()
        if result.is_robust(min_total_trades=500, max_win_rate_std=0.15):
            print(f"✓ ROBUST: Passes robustness checks")
        else:
            print(f"❌ NOT ROBUST:")
            if result.total_trades < 500:
                print(f"   - Not enough trades ({result.total_trades} < 500)")
            if result.std_win_rate > 0.15:
                print(f"   - Win-rate too inconsistent (std={result.std_win_rate:.2f} > 0.15)")
            if result.sample_bias_detected:
                print(f"   - Sample bias detected")
        print(f"{'=' * 80}")

    return result


__all__ = [
    "WalkForwardFold",
    "RobustValidationResult",
    "create_walk_forward_folds",
    "run_robust_validation",
]
