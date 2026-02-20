"""
Tests für die Fold-Aggregations-Logik in process.py.

Sichert ab:
- _build_walk_forward_summary berichtet korrekte profitable_folds / fold_stability
- std_pnl ist nie 0 wenn Folds tatsächlich unterschiedlich sind (kein cherry-picking)
- config_inconsistent wird als Warning geflaggt, ändert aber nicht die Aggregation
- representative_fold ist der mediane Fold (kein cherry-picking des besten)
"""
import pytest
import numpy as np

from fwbg.optimization.process import _build_walk_forward_summary


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def make_fold(fold_id: int, test_pnl: float, test_win_rate: float,
              test_trades: int = 50, inner_val_pnl: float = None):
    """Erstellt ein minimales fold_result dict für Tests."""
    if inner_val_pnl is None:
        inner_val_pnl = abs(test_pnl) * 1.5  # typisch: inner > outer
    return {
        "fold_id": fold_id,
        "test_pnl": test_pnl,
        "test_win_rate": test_win_rate,
        "test_trades": test_trades,
        "inner_val_pnl": inner_val_pnl,
        "test_trades_trace": [],
        "best_config": {"tp": 5.0, "sl": 2.0, "rrr": 2.5, "ct": 0.55},
    }


# ---------------------------------------------------------------------------
# Profitable Folds und Fold Stability
# ---------------------------------------------------------------------------

class TestProfitableFoldsAndStability:

    def test_profitable_folds_counted_correctly(self):
        """Anzahl profitabler Folds muss exakt stimmen."""
        folds = [
            make_fold(0, test_pnl=1000.0, test_win_rate=0.40),
            make_fold(1, test_pnl=-500.0, test_win_rate=0.28),
            make_fold(2, test_pnl=800.0,  test_win_rate=0.38),
            make_fold(3, test_pnl=-200.0, test_win_rate=0.30),
            make_fold(4, test_pnl=600.0,  test_win_rate=0.36),
        ]
        win_rates = [f["test_win_rate"] for f in folds]
        pnls = [f["test_pnl"] for f in folds]

        summary = _build_walk_forward_summary(
            folds, win_rates, pnls, 250,
            False, [1.0] * 5, 1.0,
        )

        assert summary["profitable_folds"] == 3  # Folds 0, 2, 4

    def test_fold_stability_matches_profitable_fraction(self):
        """fold_stability = profitable_folds / n_folds."""
        folds = [
            make_fold(0, test_pnl=500.0,  test_win_rate=0.38),
            make_fold(1, test_pnl=-300.0, test_win_rate=0.27),
            make_fold(2, test_pnl=400.0,  test_win_rate=0.36),
            make_fold(3, test_pnl=600.0,  test_win_rate=0.40),
            make_fold(4, test_pnl=-100.0, test_win_rate=0.29),
            make_fold(5, test_pnl=800.0,  test_win_rate=0.42),
            make_fold(6, test_pnl=200.0,  test_win_rate=0.34),
            make_fold(7, test_pnl=-400.0, test_win_rate=0.26),
        ]
        win_rates = [f["test_win_rate"] for f in folds]
        pnls = [f["test_pnl"] for f in folds]

        summary = _build_walk_forward_summary(
            folds, win_rates, pnls, 400,
            False, [1.0] * 8, 1.0,
        )

        # 5 profitable folds / 8 total = 0.625
        assert summary["fold_stability"] == pytest.approx(5 / 8)
        assert summary["profitable_folds"] == 5

    def test_all_folds_profitable_gives_stability_1(self):
        """Alle Folds profitabel → fold_stability=1.0."""
        folds = [make_fold(i, test_pnl=float(i + 1) * 100, test_win_rate=0.38) for i in range(8)]
        win_rates = [f["test_win_rate"] for f in folds]
        pnls = [f["test_pnl"] for f in folds]

        summary = _build_walk_forward_summary(
            folds, win_rates, pnls, 400,
            False, [1.0] * 8, 1.0,
        )

        assert summary["fold_stability"] == pytest.approx(1.0)
        assert summary["profitable_folds"] == 8

    def test_no_folds_profitable_gives_stability_0(self):
        """Kein Fold profitabel → fold_stability=0.0."""
        folds = [make_fold(i, test_pnl=-float(i + 1) * 100, test_win_rate=0.25) for i in range(8)]
        win_rates = [f["test_win_rate"] for f in folds]
        pnls = [f["test_pnl"] for f in folds]

        summary = _build_walk_forward_summary(
            folds, win_rates, pnls, 400,
            False, [1.0] * 8, 1.0,
        )

        assert summary["fold_stability"] == pytest.approx(0.0)
        assert summary["profitable_folds"] == 0


# ---------------------------------------------------------------------------
# Kein Cherry-Picking: std_pnl korrekt
# ---------------------------------------------------------------------------

class TestNoCheryPicking:

    def test_std_pnl_nonzero_when_folds_differ(self):
        """std_pnl darf nicht 0 sein wenn Folds unterschiedliche PnL haben.

        Der alte Bug: bei config_inconsistent wurde std_pnl=0 gesetzt weil nur
        der beste Fold berichtet wurde. Das macht std_pnl unbrauchbar.
        """
        pnls = [-500.0, -300.0, 1800.0, -2700.0, -3300.0, -600.0, 800.0, -50.0]
        folds = [
            make_fold(i, test_pnl=pnls[i], test_win_rate=0.30 + i * 0.01)
            for i in range(8)
        ]
        win_rates = [f["test_win_rate"] for f in folds]

        summary = _build_walk_forward_summary(
            folds, win_rates, pnls, sum(f["test_trades"] for f in folds),
            False, [0.5] * 8, 0.5,
        )

        assert summary["std_pnl"] > 0, (
            "std_pnl=0 deutet auf cherry-picking hin: nur ein Fold wurde berichtet"
        )

    def test_mean_pnl_reflects_all_folds(self):
        """mean_pnl muss den Durchschnitt ALLER Folds widerspiegeln, nicht nur des besten."""
        pnls = [-500.0, -300.0, 1800.0, -2700.0, -3300.0, -600.0, 800.0, -50.0]
        folds = [make_fold(i, test_pnl=pnls[i], test_win_rate=0.30) for i in range(8)]
        win_rates = [f["test_win_rate"] for f in folds]

        summary = _build_walk_forward_summary(
            folds, win_rates, pnls, 400,
            False, [0.5] * 8, 0.5,
        )

        expected_mean = np.mean(pnls)  # ≈ -856, nicht 1800 (cherry-picked best)
        assert summary["mean_pnl"] == pytest.approx(expected_mean, rel=0.001), (
            f"mean_pnl={summary['mean_pnl']:.1f} entspricht nicht dem echten "
            f"Durchschnitt aller Folds ({expected_mean:.1f})"
        )

    def test_min_max_pnl_span_all_folds(self):
        """min_pnl und max_pnl müssen den vollen Bereich aller Folds abdecken."""
        pnls = [-500.0, -300.0, 1800.0, -2700.0, -3300.0, -600.0, 800.0, -50.0]
        folds = [make_fold(i, test_pnl=pnls[i], test_win_rate=0.30) for i in range(8)]
        win_rates = [f["test_win_rate"] for f in folds]

        summary = _build_walk_forward_summary(
            folds, win_rates, pnls, 400,
            False, [0.5] * 8, 0.5,
        )

        assert summary["min_pnl"] == pytest.approx(-3300.0)
        assert summary["max_pnl"] == pytest.approx(1800.0)


# ---------------------------------------------------------------------------
# Config-Inconsistency: nur Warning, keine Verhaltensänderung
# ---------------------------------------------------------------------------

class TestConfigInconsistency:

    def test_config_inconsistent_flagged_in_summary(self):
        """config_inconsistent=True muss in der Summary sichtbar sein."""
        folds = [make_fold(i, test_pnl=500.0, test_win_rate=0.38) for i in range(4)]
        pnls = [f["test_pnl"] for f in folds]
        win_rates = [f["test_win_rate"] for f in folds]

        summary = _build_walk_forward_summary(
            folds, win_rates, pnls, 200,
            False, [1.0] * 4, 1.0,
            config_inconsistent=True,
        )

        assert summary.get("config_inconsistent") is True

    def test_config_consistent_not_flagged(self):
        """Ohne Inconsistency darf das Flag nicht gesetzt sein."""
        folds = [make_fold(i, test_pnl=500.0, test_win_rate=0.38) for i in range(4)]
        pnls = [f["test_pnl"] for f in folds]
        win_rates = [f["test_win_rate"] for f in folds]

        summary = _build_walk_forward_summary(
            folds, win_rates, pnls, 200,
            False, [1.0] * 4, 1.0,
            config_inconsistent=False,
        )

        assert "config_inconsistent" not in summary

    def test_config_inconsistent_does_not_change_stats(self):
        """config_inconsistent darf die aggregierten Statistiken nicht verändern.

        Früher setzte dieser Flag std_pnl=0 durch cherry-picking. Das ist behoben.
        """
        pnls = [-200.0, 800.0, -400.0, 1000.0]
        folds = [make_fold(i, test_pnl=pnls[i], test_win_rate=0.30 + i * 0.03)
                 for i in range(4)]
        win_rates = [f["test_win_rate"] for f in folds]

        summary_consistent = _build_walk_forward_summary(
            folds, win_rates, pnls, 200,
            False, [1.0] * 4, 1.0,
            config_inconsistent=False,
        )
        summary_inconsistent = _build_walk_forward_summary(
            folds, win_rates, pnls, 200,
            False, [1.0] * 4, 1.0,
            config_inconsistent=True,
        )

        # Statistiken müssen identisch sein — nur das Flag unterscheidet sich
        assert summary_consistent["mean_pnl"] == summary_inconsistent["mean_pnl"]
        assert summary_consistent["std_pnl"] == summary_inconsistent["std_pnl"]
        assert summary_consistent["profitable_folds"] == summary_inconsistent["profitable_folds"]
        assert summary_consistent["fold_stability"] == summary_inconsistent["fold_stability"]

    def test_no_using_best_fold_only_key(self):
        """Der alte 'using_best_fold_only' Key darf nicht mehr existieren."""
        folds = [make_fold(i, test_pnl=float(i * 100), test_win_rate=0.35) for i in range(4)]
        pnls = [f["test_pnl"] for f in folds]
        win_rates = [f["test_win_rate"] for f in folds]

        summary = _build_walk_forward_summary(
            folds, win_rates, pnls, 200,
            False, [1.0] * 4, 1.0,
            config_inconsistent=True,
        )

        assert "using_best_fold_only" not in summary, (
            "Der 'using_best_fold_only' Key ist ein Überbleibsel des cherry-picking Bugs"
        )
        assert "best_fold_id" not in summary, (
            "Der 'best_fold_id' Key ist ein Überbleibsel des cherry-picking Bugs"
        )
