"""
Live Bias Detection während Optimization.

Diese Funktionen laufen NACH jedem Asset und geben sofort Feedback
wenn Sample Bias erkannt wird.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tests'))

from typing import Dict, Any, List

try:
    from test_sample_bias_detection import SampleBiasDetector
except ImportError:
    # Fallback if tests not in path
    SampleBiasDetector = None


def fold_bias_ratios(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Test/inner PnL ratio per fold, defined only where the inner PnL is positive.

    A fold whose chosen setting did not win on its inner windows (inner PnL <= 0,
    possible for ML and signal models) has no meaningful ratio: dividing by zero,
    or a sign flip that reads as "no bias" (or as bias). Such folds are left out
    of the ratios and counted; the mean is None when no fold has a ratio.

    The mean weights each fold's ratio by its inner PnL (sum of test PnL / sum of
    inner PnL over the folds with a ratio), so a fold whose inner PnL is barely
    positive keeps its large ratio in the list (a per-fold warning) but cannot
    drive the mean (the CRITICAL check) on its own. No threshold is involved.

    Scale: inner_val_pnl is the MEAN PnL per inner validation window, test_pnl the
    PnL of the whole test fold, and the two differ in length; a ratio of 1 is not
    equal per-bar performance (ML and signal models alike).
    """
    with_ratio = [r for r in fold_results if r["inner_val_pnl"] > 0]
    inner_sum = sum(r["inner_val_pnl"] for r in with_ratio)
    return {
        "bias_ratios": [r["test_pnl"] / r["inner_val_pnl"] for r in with_ratio],
        "mean_bias_ratio": sum(r["test_pnl"] for r in with_ratio) / inner_sum if with_ratio else None,
        "folds_without_bias_ratio": len(fold_results) - len(with_ratio),
    }


def check_asset_bias(result: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Prüft ein einzelnes Asset-Result auf Sample Bias.

    Args:
        result: Asset result dict from process_symbol
        verbose: Print warnings/errors

    Returns:
        Dict mit Bias-Check Ergebnissen
    """
    symbol = result.get("symbol", "UNKNOWN")

    # Nur OK assets checken
    if result.get("status") != "ok":
        return {"symbol": symbol, "status": result.get("status"), "bias_check": "skipped"}

    # Walk-Forward Results extrahieren
    wf = result.get("walk_forward", {})
    if not wf:
        if verbose:
            print(f"⚠️  {symbol}: No walk-forward results")
        return {
            "symbol": symbol,
            "bias_check": "no_walk_forward",
            "error": "Missing walk_forward section"
        }

    # Daten extrahieren
    n_folds = wf.get("n_folds", 0)
    mean_wr = wf.get("mean_win_rate", 0)
    std_wr = wf.get("std_win_rate", 0)
    bias_ratios = wf.get("bias_ratios", [])
    mean_bias = wf.get("mean_bias_ratio", 0)
    without_ratio = wf.get("folds_without_bias_ratio", 0)
    total_trades = wf.get("total_trades", 0)

    # RRR berechnen
    rrr = result.get("rrr", 0.5)

    # Checks durchführen
    issues = []
    warnings = []

    # Check 1: Genug Folds?
    if n_folds < 5:
        issues.append(f"Only {n_folds} folds (need >=5)")

    # Check 2: Mean Bias Ratio (None: no fold has a positive inner PnL)
    if mean_bias is not None and mean_bias > 1.5:
        issues.append(f"Mean bias ratio {mean_bias:.2f}x > 1.5x (systematic bias!)")
    elif mean_bias is not None and mean_bias > 1.3:
        warnings.append(f"Mean bias ratio {mean_bias:.2f}x > 1.3x (monitor)")
    if without_ratio:
        warnings.append(f"{without_ratio}/{n_folds} folds without bias ratio (inner PnL <= 0)")

    # Check 3: Einzelne Folds mit extremem Bias
    extreme_folds = [r for r in bias_ratios if r > 2.0]
    if len(extreme_folds) > 0:
        warnings.append(f"{len(extreme_folds)}/{n_folds} folds have >2.0x bias: {[f'{r:.2f}' for r in extreme_folds]}")

    # Check 4: Win-Rate Konsistenz (None: no fold traded)
    if std_wr is not None and std_wr > 0.15:
        warnings.append(f"High WR std-dev: {std_wr*100:.1f}% (regime-dependent)")

    # Check 5: Unrealistic Win-Rate (if SampleBiasDetector available)
    if SampleBiasDetector and mean_wr is not None:
        wr_check = SampleBiasDetector.check_unrealistic_winrate(
            win_rate=mean_wr,
            rrr=rrr,
            tolerance=0.15
        )
        if wr_check["has_bias"]:
            issues.append(f"Unrealistic WR: {mean_wr*100:.1f}% (breakeven={wr_check['breakeven_wr']*100:.1f}%, excess={wr_check['excess']*100:.1f}%)")

    # Check 6: Genug Trades
    if total_trades < 500:
        warnings.append(f"Low trade count: {total_trades} (need >=500)")

    # Ergebnis zusammenfassen
    if issues:
        status = "CRITICAL"
        severity = "🚨"
    elif warnings:
        status = "WARNING"
        severity = "⚠️ "
    else:
        status = "OK"
        severity = "✓"

    # Logging
    if verbose:
        if issues:
            print(f"\n{severity} {symbol}: BIAS DETECTED!")
            for issue in issues:
                print(f"    - {issue}")
            print(f"    Bias Ratios: {[f'{r:.2f}x' for r in bias_ratios]}")
        elif warnings:
            print(f"{severity} {symbol}: Warnings")
            for warn in warnings:
                print(f"    - {warn}")
        else:
            std_text = "n/a" if std_wr is None else f"{std_wr*100:.1f}%"
            mean_text = "n/a" if mean_bias is None else f"{mean_bias:.2f}x"
            print(f"{severity} {symbol}: No bias detected (mean={mean_text}, std_wr={std_text})")

    return {
        "symbol": symbol,
        "bias_check": status.lower(),
        "n_folds": n_folds,
        "mean_bias_ratio": mean_bias,
        "bias_ratios": bias_ratios,
        "folds_without_bias_ratio": without_ratio,
        "mean_win_rate": mean_wr,
        "std_win_rate": std_wr,
        "total_trades": total_trades,
        "issues": issues,
        "warnings": warnings,
    }


def check_systematic_bias(all_results: List[Dict[str, Any]], verbose: bool = True) -> Dict[str, Any]:
    """
    Prüft ob systematischer Bias über ALLE Assets vorliegt.

    Args:
        all_results: Liste aller asset results
        verbose: Print summary

    Returns:
        Dict mit System-wide Bias Check
    """
    ok_assets = [r for r in all_results if r.get("status") == "ok"]

    if len(ok_assets) == 0:
        return {"status": "no_assets", "message": "No OK assets to check"}

    # Bias-Checks sammeln
    biased_assets = []
    warned_assets = []
    ok_count = 0

    for result in ok_assets:
        wf = result.get("walk_forward", {})
        if not wf:
            continue

        mean_bias = wf.get("mean_bias_ratio", 0)
        symbol = result.get("symbol", "UNKNOWN")

        if mean_bias is None:
            # No fold had a positive inner PnL: the bias is unchecked, not OK.
            warned_assets.append({
                "symbol": symbol,
                "mean_bias": None
            })
        elif mean_bias > 1.5:
            biased_assets.append({
                "symbol": symbol,
                "mean_bias": mean_bias,
                "bias_ratios": wf.get("bias_ratios", [])
            })
        elif mean_bias > 1.3:
            warned_assets.append({
                "symbol": symbol,
                "mean_bias": mean_bias
            })
        else:
            ok_count += 1

    total = len(ok_assets)
    bias_percentage = len(biased_assets) / total if total > 0 else 0

    # System-wide Check
    if bias_percentage > 0.2:
        status = "SYSTEMATIC_BIAS"
        severity = "🚨"
        message = f"SYSTEMATIC BIAS! {len(biased_assets)}/{total} assets ({bias_percentage*100:.0f}%) have mean_bias >1.5x"
    elif bias_percentage > 0.1:
        status = "WARNING"
        severity = "⚠️ "
        message = f"Warning: {len(biased_assets)}/{total} assets ({bias_percentage*100:.0f}%) have mean_bias >1.5x"
    else:
        status = "OK"
        severity = "✓"
        message = f"System OK: Only {len(biased_assets)}/{total} assets biased (<10%)"

    # Logging
    if verbose:
        print(f"\n{'='*80}")
        print("SYSTEMATIC BIAS CHECK")
        print(f"{'='*80}")
        print(f"{severity} {message}")
        print(f"  OK: {ok_count}/{total}")
        print(f"  Warnings: {len(warned_assets)}/{total}")
        print(f"  Biased: {len(biased_assets)}/{total}")

        if biased_assets:
            print("\nBiased Assets:")
            for a in biased_assets:
                print(f"  - {a['symbol']}: {a['mean_bias']:.2f}x (ratios: {[f'{r:.2f}' for r in a['bias_ratios']]})")

        if warned_assets and verbose:
            print("\nWarning Assets:")
            for a in warned_assets:
                mean_text = "n/a" if a['mean_bias'] is None else f"{a['mean_bias']:.2f}x"
                print(f"  - {a['symbol']}: {mean_text}")

        print(f"{'='*80}\n")

    return {
        "status": status.lower(),
        "total_assets": total,
        "ok_assets": ok_count,
        "warned_assets": len(warned_assets),
        "biased_assets": len(biased_assets),
        "bias_percentage": bias_percentage,
        "biased_list": biased_assets,
        "message": message,
    }


__all__ = ["check_asset_bias", "check_systematic_bias", "fold_bias_ratios"]
