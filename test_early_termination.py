#!/usr/bin/env python3
"""
Test: Prüft, ob die neuen Early Termination Checks die erfolgreichen
Parameter aus Run 20260201_014605_c5f7f5 verhindern würden.

Simuliert die Inner-CV-Folds mit PnL-Profilen und testet:
1. Alter Code: Nur mathematische Unmöglichkeit + First-Fold Sanity Check
2. Neuer Code: + 3 consecutive failures + Halfway negative PnL check
"""
import math


def test_early_termination_old(fold_pnls, min_fold_stability=0.5):
    """
    Alte Early Termination Logic (wie in Commit b5d81a1)

    Bricht nur ab, wenn mathematisch unmöglich, min_fold_stability zu erreichen.
    """
    total_folds = len(fold_pnls)
    min_profitable = int(math.ceil(total_folds * min_fold_stability))
    profitable_count = 0
    failed_count = 0

    for fold_idx, fold_pnl in enumerate(fold_pnls):
        # Check: Kann min_fold_stability noch erreicht werden?
        remaining_folds = total_folds - fold_idx
        max_possible_profitable = profitable_count + remaining_folds
        if max_possible_profitable < min_profitable:
            return {
                "terminated": True,
                "reason": "mathematical_impossibility",
                "at_fold": fold_idx,
                "profitable_count": profitable_count,
                "failed_count": failed_count
            }

        # Fold evaluieren
        if fold_pnl is None:
            # Fold komplett fehlgeschlagen (keine Features, keine Trades, etc.)
            failed_count += 1
        elif fold_pnl > 0:
            profitable_count += 1
        else:
            failed_count += 1

    return {
        "terminated": False,
        "reason": None,
        "profitable_count": profitable_count,
        "failed_count": failed_count
    }


def test_early_termination_new(fold_pnls, min_fold_stability=0.5):
    """
    Neue Early Termination Logic (wie in Commit d5c8b50+)

    Zusätzliche Checks:
    1. 3 consecutive failures → Abbruch
    2. Nach Hälfte der Folds: Durchschnitt-PnL < -3 → Abbruch
    """
    total_folds = len(fold_pnls)
    min_profitable = int(math.ceil(total_folds * min_fold_stability))
    profitable_count = 0
    failed_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 3
    cumulative_pnl = 0.0

    for fold_idx, fold_pnl in enumerate(fold_pnls):
        # Check 1: Mathematische Unmöglichkeit (wie alter Code)
        remaining_folds = total_folds - fold_idx
        max_possible_profitable = profitable_count + remaining_folds
        if max_possible_profitable < min_profitable:
            return {
                "terminated": True,
                "reason": "mathematical_impossibility",
                "at_fold": fold_idx,
                "profitable_count": profitable_count,
                "failed_count": failed_count
            }

        # Fold evaluieren
        if fold_pnl is None:
            # Komplett fehlgeschlagen
            failed_count += 1
            consecutive_failures += 1
        elif fold_pnl > 0:
            profitable_count += 1
            consecutive_failures = 0  # Reset
            cumulative_pnl += fold_pnl
        else:
            failed_count += 1
            consecutive_failures += 1
            cumulative_pnl += fold_pnl

        # Check 2: 3 consecutive failures (NEU!)
        if consecutive_failures >= max_consecutive_failures:
            return {
                "terminated": True,
                "reason": "3_consecutive_failures",
                "at_fold": fold_idx,
                "profitable_count": profitable_count,
                "failed_count": failed_count
            }

        # Check 3: Halfway negative check (NEU!)
        halfway = total_folds // 2
        if fold_idx >= halfway and fold_idx >= 2:
            avg_pnl_so_far = cumulative_pnl / (fold_idx + 1)
            if avg_pnl_so_far < -3.0:
                return {
                    "terminated": True,
                    "reason": "halfway_negative",
                    "at_fold": fold_idx,
                    "profitable_count": profitable_count,
                    "failed_count": failed_count,
                    "avg_pnl": avg_pnl_so_far
                }

    return {
        "terminated": False,
        "reason": None,
        "profitable_count": profitable_count,
        "failed_count": failed_count
    }


# Test Cases: Simulierte Fold-PnL-Profile
# Basierend auf typischen Verläufen von Inner-CV-Folds
TEST_CASES = [
    {
        "name": "Slow Learner (Successful)",
        "description": "Erst negativ, dann positiv - typisch für ML-basierte Strategien",
        "fold_pnls": [-5, -8, 2, 15, 10, 8],  # 3 profitable von 6 = 50%
    },
    {
        "name": "Volatile Success",
        "description": "Abwechselnd positiv/negativ aber insgesamt profitabel",
        "fold_pnls": [10, -5, 12, -3, 8, 15],  # 4 profitable von 6 = 67%
    },
    {
        "name": "Rocky Start",
        "description": "3 Verluste zu Beginn, dann Erholung",
        "fold_pnls": [-2, -5, -3, 10, 15, 12],  # 3 profitable von 6 = 50%
    },
    {
        "name": "Gradual Improvement",
        "description": "Langsame Verbesserung über Folds",
        "fold_pnls": [-10, -5, -2, 1, 5, 10],  # 3 profitable von 6 = 50%
    },
    {
        "name": "Consistent Mediocrity",
        "description": "Kleine Verluste aber nie katastrophal",
        "fold_pnls": [-2, -1, -3, -2, 1, -1],  # 1 profitable von 6 = 17%
    },
]


def main():
    print("="*80)
    print("EARLY TERMINATION TEST")
    print("="*80)
    print("\nVergleicht alte vs. neue Early Termination Logic")
    print("Min Fold Stability: 50% (mindestens 3 von 5 Folds profitabel)")
    print()

    for test_case in TEST_CASES:
        print(f"\n{'='*80}")
        print(f"TEST: {test_case['name']}")
        print(f"{'='*80}")
        print(f"Description: {test_case['description']}")
        print(f"Fold PnLs: {test_case['fold_pnls']}")
        print()

        fold_pnls = test_case['fold_pnls']

        # Test mit alter Logic
        result_old = test_early_termination_old(fold_pnls)
        print(f"OLD Logic (nur mathematische Unmöglichkeit):")
        if result_old["terminated"]:
            print(f"  ❌ TERMINATED at fold {result_old['at_fold']}")
            print(f"  Reason: {result_old['reason']}")
        else:
            print(f"  ✅ COMPLETED all folds")
            print(f"  Profitable: {result_old['profitable_count']}/{len(fold_pnls)}")
        print()

        # Test mit neuer Logic
        result_new = test_early_termination_new(fold_pnls)
        print(f"NEW Logic (+ consecutive failures + halfway check):")
        if result_new["terminated"]:
            print(f"  ❌ TERMINATED at fold {result_new['at_fold']}")
            print(f"  Reason: {result_new['reason']}")
            if 'avg_pnl' in result_new:
                print(f"  Avg PnL: {result_new['avg_pnl']:.1f}")
        else:
            print(f"  ✅ COMPLETED all folds")
            print(f"  Profitable: {result_new['profitable_count']}/{len(fold_pnls)}")
        print()

        # Vergleich
        if result_old["terminated"] == result_new["terminated"]:
            print(f"  ➡️  SAME OUTCOME")
        else:
            if result_old["terminated"]:
                print(f"  ⚠️  OLD terminated, NEW did not (unexpected!)")
            else:
                print(f"  ⚠️  NEW terminated, OLD did not (MORE RESTRICTIVE!)")


    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nDie neuen Early Termination Checks sind deutlich aggressiver:")
    print("1. '3 consecutive failures' - stoppt bei 3 Verlusten hintereinander")
    print("2. 'Halfway negative' - stoppt bei durchschnittlich > 3 Verlust/Fold")
    print("\nDies kann profitable Parameter-Kombinationen verhindern, die:")
    print("- Einen schlechten Start haben, sich aber dann erholen")
    print("- Volatile Ergebnisse haben (up/down)")
    print("- Erst nach mehreren Folds zu lernen beginnen (typisch für ML!)")


if __name__ == "__main__":
    main()
