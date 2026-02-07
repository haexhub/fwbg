"""
Automatische Tests für Sample Bias Detection.

WICHTIG: Diese Tests DETEKTIEREN und WARNEN bei Sample Bias,
aber REJECTEN NICHT automatisch. Wenn mehrere Assets biased sind,
deutet das auf ein SYSTEMATISCHES Problem im Code hin, nicht auf
problematische einzelne Assets.

Diese Tests sollten WÄHREND und NACH jeder Optimierung laufen
um systematische Probleme zu DIAGNOSTIZIEREN.
"""
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any


class SampleBiasDetector:
    """
    Detek tiert Sample Bias in Backtest-Ergebnissen.

    Sample Bias liegt vor wenn:
    - Holdout deutlich besser ist als Inner Validation (lucky period)
    - Win-Rate unrealistisch hoch für gegebenes RRR
    - Zu wenig Trades für statistische Signifikanz
    """

    @staticmethod
    def check_holdout_vs_inner_bias(
        inner_val_pnl: float,
        holdout_pnl: float,
        max_ratio: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Prüft ob Holdout verdächtig viel besser ist als Inner Val.

        Args:
            inner_val_pnl: PnL auf Inner Validation
            holdout_pnl: PnL auf Holdout
            max_ratio: Maximum erlaubtes Ratio

        Returns:
            dict mit 'has_bias', 'ratio', 'message'
        """
        if inner_val_pnl <= 0:
            return {
                "has_bias": True,
                "ratio": float('inf'),
                "message": "Inner validation unprofitable, suspicious holdout profit",
                "severity": "CRITICAL"
            }

        ratio = holdout_pnl / inner_val_pnl

        if ratio > max_ratio:
            return {
                "has_bias": True,
                "ratio": ratio,
                "message": f"Holdout {ratio:.2f}x better than inner - likely sample bias",
                "severity": "HIGH"
            }
        elif ratio > 1.5:
            return {
                "has_bias": False,
                "ratio": ratio,
                "message": f"Holdout {ratio:.2f}x better than inner - monitor closely",
                "severity": "MEDIUM"
            }
        else:
            return {
                "has_bias": False,
                "ratio": ratio,
                "message": f"Holdout/Inner ratio normal ({ratio:.2f}x)",
                "severity": "NONE"
            }

    @staticmethod
    def check_unrealistic_winrate(
        win_rate: float,
        rrr: float,
        tolerance: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Prüft ob Win-Rate unrealistisch hoch für gegebenes RRR ist.

        Args:
            win_rate: Tatsächliche Win-Rate
            rrr: Risk-Reward-Ratio (TP/SL)
            tolerance: Erlaubte Abweichung über Break-Even

        Returns:
            dict mit 'has_bias', 'excess', 'message'
        """
        # Break-even win-rate für gegebenes RRR
        breakeven_wr = 1.0 / (1.0 + rrr)

        # Maximum realistische WR = breakeven + tolerance
        max_realistic_wr = breakeven_wr + tolerance

        excess = win_rate - breakeven_wr

        if win_rate > max_realistic_wr:
            return {
                "has_bias": True,
                "excess": excess,
                "breakeven_wr": breakeven_wr,
                "message": f"Win-rate {win_rate*100:.1f}% too high for RRR={rrr:.2f} "
                          f"(breakeven={breakeven_wr*100:.1f}%, excess={excess*100:.1f}%)",
                "severity": "HIGH"
            }
        elif excess > tolerance / 2:
            return {
                "has_bias": False,
                "excess": excess,
                "breakeven_wr": breakeven_wr,
                "message": f"Win-rate above breakeven but reasonable "
                          f"(excess={excess*100:.1f}%)",
                "severity": "LOW"
            }
        else:
            return {
                "has_bias": False,
                "excess": excess,
                "breakeven_wr": breakeven_wr,
                "message": f"Win-rate reasonable for RRR={rrr:.2f}",
                "severity": "NONE"
            }

    @staticmethod
    def check_insufficient_trades(
        n_trades: int,
        min_trades: int = 500,
    ) -> Dict[str, Any]:
        """
        Prüft ob genug Trades für statistische Signifikanz vorhanden sind.

        Args:
            n_trades: Anzahl Trades
            min_trades: Minimum für Signifikanz

        Returns:
            dict mit 'has_issue', 'n_trades', 'message'
        """
        if n_trades < min_trades:
            return {
                "has_issue": True,
                "n_trades": n_trades,
                "message": f"Only {n_trades} trades (need {min_trades} for significance)",
                "severity": "HIGH"
            }
        elif n_trades < min_trades * 1.5:
            return {
                "has_issue": False,
                "n_trades": n_trades,
                "message": f"{n_trades} trades - borderline significance",
                "severity": "MEDIUM"
            }
        else:
            return {
                "has_issue": False,
                "n_trades": n_trades,
                "message": f"{n_trades} trades - sufficient for significance",
                "severity": "NONE"
            }

    @staticmethod
    def check_consistency_across_folds(
        fold_win_rates: list,
        max_std: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Prüft Konsistenz der Win-Rate über mehrere Folds.

        Args:
            fold_win_rates: Liste von Win-Rates pro Fold
            max_std: Maximum Std-Dev

        Returns:
            dict mit 'has_issue', 'std', 'message'
        """
        if len(fold_win_rates) < 2:
            return {
                "has_issue": True,
                "std": 0.0,
                "message": "Need at least 2 folds to check consistency",
                "severity": "HIGH"
            }

        std = np.std(fold_win_rates)
        mean = np.mean(fold_win_rates)
        cv = std / mean if mean > 0 else float('inf')  # Coefficient of variation

        if std > max_std:
            return {
                "has_issue": True,
                "std": std,
                "cv": cv,
                "message": f"Win-rate inconsistent across folds (std={std:.2f}, CV={cv:.2f})",
                "severity": "HIGH"
            }
        elif std > max_std / 2:
            return {
                "has_issue": False,
                "std": std,
                "cv": cv,
                "message": f"Win-rate moderately consistent (std={std:.2f})",
                "severity": "MEDIUM"
            }
        else:
            return {
                "has_issue": False,
                "std": std,
                "cv": cv,
                "message": f"Win-rate highly consistent (std={std:.2f})",
                "severity": "NONE"
            }

    @classmethod
    def run_full_check(
        cls,
        inner_val_pnl: float,
        holdout_pnl: float,
        win_rate: float,
        rrr: float,
        n_trades: int,
        fold_win_rates: list = None,
    ) -> Dict[str, Any]:
        """
        Führt alle Sample Bias Checks durch.

        Returns:
            dict mit allen Ergebnissen und Overall-Verdict
        """
        results = {}

        # Check 1: Holdout vs Inner
        results["holdout_bias"] = cls.check_holdout_vs_inner_bias(
            inner_val_pnl, holdout_pnl
        )

        # Check 2: Unrealistic Win-Rate
        results["winrate_bias"] = cls.check_unrealistic_winrate(
            win_rate, rrr
        )

        # Check 3: Insufficient Trades
        results["trade_count"] = cls.check_insufficient_trades(n_trades)

        # Check 4: Fold Consistency (if available)
        if fold_win_rates is not None and len(fold_win_rates) > 1:
            results["fold_consistency"] = cls.check_consistency_across_folds(
                fold_win_rates
            )

        # Overall Verdict
        has_critical = any(
            r.get("severity") == "CRITICAL"
            for r in results.values()
        )
        has_high = any(
            r.get("severity") == "HIGH" or r.get("has_bias") or r.get("has_issue")
            for r in results.values()
        )

        if has_critical:
            verdict = "CRITICAL WARNING - Systematic Issue Suspected"
        elif has_high:
            verdict = "WARNING - High Risk of Sample Bias"
        else:
            verdict = "OK - Appears Robust"

        return {
            "verdict": verdict,
            "checks": results,
        }


# =============================================================================
# PYTEST TESTS
# =============================================================================

class TestSampleBiasDetector:
    """Unit tests für Sample Bias Detector."""

    def test_obvious_sample_bias(self):
        """Test: Offensichtlicher Sample Bias wird erkannt."""
        result = SampleBiasDetector.check_holdout_vs_inner_bias(
            inner_val_pnl=50.0,
            holdout_pnl=150.0,  # 3x better!
        )
        assert result["has_bias"] is True
        assert result["severity"] == "HIGH"

    def test_normal_holdout_performance(self):
        """Test: Normale Holdout Performance wird akzeptiert."""
        result = SampleBiasDetector.check_holdout_vs_inner_bias(
            inner_val_pnl=50.0,
            holdout_pnl=55.0,  # 1.1x better
        )
        assert result["has_bias"] is False

    def test_unrealistic_winrate(self):
        """Test: Unrealistische Win-Rate wird erkannt."""
        result = SampleBiasDetector.check_unrealistic_winrate(
            win_rate=0.83,  # 83%
            rrr=0.5,  # TP=10, SL=20
        )
        # Breakeven für RRR=0.5 ist 66.7%
        # 83% ist +16.4% über Breakeven
        # Mit tolerance=0.15 sollte das als verdächtig erkannt werden
        assert result["has_bias"] is True

    def test_realistic_winrate(self):
        """Test: Realistische Win-Rate wird akzeptiert."""
        result = SampleBiasDetector.check_unrealistic_winrate(
            win_rate=0.72,  # 72%
            rrr=0.5,
        )
        # 72% ist nur +5.3% über Breakeven (66.7%)
        assert result["has_bias"] is False

    def test_insufficient_trades(self):
        """Test: Zu wenig Trades wird erkannt."""
        result = SampleBiasDetector.check_insufficient_trades(
            n_trades=100,
            min_trades=500,
        )
        assert result["has_issue"] is True

    def test_sufficient_trades(self):
        """Test: Genug Trades wird akzeptiert."""
        result = SampleBiasDetector.check_insufficient_trades(
            n_trades=1000,
            min_trades=500,
        )
        assert result["has_issue"] is False

    def test_inconsistent_folds(self):
        """Test: Inkonsistente Fold-Ergebnisse werden erkannt."""
        result = SampleBiasDetector.check_consistency_across_folds(
            fold_win_rates=[0.45, 0.60, 0.90, 0.55],  # Sehr unterschiedlich (std > 0.15)
        )
        assert result["has_issue"] is True

    def test_consistent_folds(self):
        """Test: Konsistente Fold-Ergebnisse werden akzeptiert."""
        result = SampleBiasDetector.check_consistency_across_folds(
            fold_win_rates=[0.68, 0.70, 0.69, 0.71],  # Sehr ähnlich
        )
        assert result["has_issue"] is False

    def test_full_check_audusd_case(self):
        """Test: AUDUSD Fall (aus echten Daten) wird korrekt erkannt."""
        result = SampleBiasDetector.run_full_check(
            inner_val_pnl=55.4,
            holdout_pnl=129.0,  # 2.33x!
            win_rate=0.831,
            rrr=0.5,
            n_trades=195,
        )

        # Sollte als WARNING erkannt werden (aber nicht rejected!)
        assert "WARNING" in result["verdict"]

        # Holdout Bias sollte erkannt werden
        assert result["checks"]["holdout_bias"]["has_bias"] is True

        # Win-Rate sollte als verdächtig erkannt werden
        assert result["checks"]["winrate_bias"]["has_bias"] is True

        # Zu wenig Trades
        assert result["checks"]["trade_count"]["has_issue"] is True


if __name__ == "__main__":
    # Manuelle Tests
    print("=" * 80)
    print("SAMPLE BIAS DETECTOR - MANUAL TESTS")
    print("=" * 80)

    # Test 1: AUDUSD Case
    print("\nTest 1: AUDUSD (Real Data)")
    print("-" * 80)
    result = SampleBiasDetector.run_full_check(
        inner_val_pnl=55.4,
        holdout_pnl=129.0,
        win_rate=0.831,
        rrr=0.5,
        n_trades=195,
    )

    print(f"Verdict: {result['verdict']}")
    print("\nDetails:")
    for check_name, check_result in result['checks'].items():
        print(f"  {check_name}:")
        print(f"    {check_result['message']}")
        print(f"    Severity: {check_result['severity']}")

    # Test 2: Normal Case
    print("\n\nTest 2: Normal Robust Config")
    print("-" * 80)
    result = SampleBiasDetector.run_full_check(
        inner_val_pnl=45.0,
        holdout_pnl=42.0,  # Slightly worse (normal)
        win_rate=0.70,
        rrr=0.5,
        n_trades=850,
        fold_win_rates=[0.68, 0.71, 0.69, 0.72, 0.70],
    )

    print(f"Verdict: {result['verdict']}")
    print("\nDetails:")
    for check_name, check_result in result['checks'].items():
        print(f"  {check_name}:")
        print(f"    {check_result['message']}")
        print(f"    Severity: {check_result['severity']}")

    print("\n" + "=" * 80)
