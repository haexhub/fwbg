"""Tests für die kausale Reporting-Kalibrierung (Plan: Look-Ahead-Fix)."""
import pytest

from fwbg.optimization.causal_reporting import (
    circuit_breaker_mask,
    compute_causal_reporting,
)


def _fold(binary_pnl):
    """Baut ein unified_fold_result aus (result, pnl_raw)-Tupeln."""
    return {"trades": [{"result": r, "pnl_raw": p} for r, p in binary_pnl]}


class StubRiskManager:
    """Zeichnet auf, mit welchen Trades kalibriert wurde."""

    def __init__(self, fk=0.01):
        self.fk = fk
        self.calls = []

    def compute_risk_params(self, trades, win_rate, rrr, rv_values=None, **params):
        self.calls.append(list(trades))
        return {
            "risk_per_trade": self.fk,
            "circuit_breaker": {
                "pause_after_losses": 0,
                "pause_bars": 0,
                "enabled": False,
            },
            "risk_adjustment": {"scale_factor": 1.0},
        }


class TestCircuitBreakerMask:
    def test_disabled_takes_all(self):
        assert circuit_breaker_mask([-1.0, -1.0, 1.0], 0, 5) == [True, True, True]

    def test_pause_after_losses(self):
        trades = [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0]
        mask = circuit_breaker_mask(trades, pause_after_losses=2, pause_bars=1)
        # Nach 2 Verlusten in Serie wird genau 1 Trade übersprungen.
        assert mask == [True, True, False, True, True, False, True]

    def test_matches_simulate_with_circuit_breaker(self):
        from fwbg.simulation.trade import simulate_with_circuit_breaker

        trades = [1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0]
        mask = circuit_breaker_mask(trades, 3, 2)
        taken = [t for t, keep in zip(trades, mask) if keep]
        ref = simulate_with_circuit_breaker(trades, 0.01, 2.0, 3, 2)
        assert taken == ref["filtered_trades"]


class TestComputeCausalReporting:
    def test_fold_0_excluded_and_prior_only_calibration(self):
        fold0 = _fold([(1.0, 0.003), (-1.0, -0.002)])
        fold1 = _fold([(1.0, 0.004), (-1.0, -0.002), (1.0, 0.003)])
        fold2 = _fold([(-1.0, -0.001)])
        mgr = StubRiskManager(fk=0.02)

        out = compute_causal_reporting([fold0, fold1, fold2], mgr, 2.0, {})

        # Fold 0 exkludiert, Folds 1+2 getradet
        assert out["fold_0_excluded"] is True
        assert out["excluded_trades"] == 2
        assert len(out["causal_pnl"]) == 4
        assert out["causal_trade_indices"] == [2, 3, 4, 5]
        assert out["per_fold_risk"] == [None, 0.02, 0.02]
        assert out["untraded_folds"] == []

        # Kalibrierung von Fold 1 sah NUR Fold-0-Trades,
        # Fold 2 sah Fold 0+1 — nie den eigenen Fold.
        assert mgr.calls[0] == [1.0, -1.0]
        assert mgr.calls[1] == [1.0, -1.0, 1.0, -1.0, 1.0]

    def test_returns_scaled_by_prior_avg_loss(self):
        fold0 = _fold([(1.0, 0.004), (-1.0, -0.002)])
        fold1 = _fold([(1.0, 0.006)])
        mgr = StubRiskManager(fk=0.01)

        out = compute_causal_reporting([fold0, fold1], mgr, 2.0, {})

        # Skala = avg |Loss| der Prior-Folds (0.002), NICHT des eigenen Folds
        assert out["causal_returns"][0] == pytest.approx(0.01 * 0.006 / 0.002)

    def test_no_edge_on_prior_data_means_fold_not_traded(self):
        fold0 = _fold([(-1.0, -0.002)] * 5)
        fold1 = _fold([(1.0, 0.004)] * 5)
        mgr = StubRiskManager(fk=0.0)

        out = compute_causal_reporting([fold0, fold1], mgr, 2.0, {})

        assert out["untraded_folds"] == [1]
        assert out["causal_pnl"] == []
        assert out["excluded_trades"] == 10

    def test_deterministic(self):
        folds = [
            _fold([(1.0, 0.003), (-1.0, -0.002)] * 3),
            _fold([(-1.0, -0.001), (1.0, 0.005)] * 4),
        ]
        mgr1, mgr2 = StubRiskManager(), StubRiskManager()
        out1 = compute_causal_reporting(folds, mgr1, 1.5, {})
        out2 = compute_causal_reporting(folds, mgr2, 1.5, {})
        assert out1 == out2


class TestHindsightRemoval:
    """Kern-Eigenschaft: die kausale Kalibrierung kennt spätere Loss-Cluster
    nicht — die naive (Hindsight-)Kalibrierung skaliert das Risiko anhand
    genau dieser bekannten Sequenz herunter."""

    def test_causal_risk_ignores_future_loss_cluster(self):
        from fwbg.plugins import import_plugin_module

        _kelly = import_plugin_module("fwbg-core", "risk_management", "kelly")
        if _kelly is None:
            pytest.skip("fwbg-core kelly plugin not available")
        mgr = _kelly.KellyRiskManager()

        # Fold 0: gutartig. Fold 1: harter Loss-Cluster.
        fold0_trades = ([1.0] * 6 + [-1.0] * 4) * 5
        fold1_trades = [-1.0] * 25 + [1.0] * 5
        fold0 = _fold([(r, 0.003 if r > 0 else -0.002) for r in fold0_trades])
        fold1 = _fold([(r, 0.003 if r > 0 else -0.002) for r in fold1_trades])

        out = compute_causal_reporting([fold0, fold1], mgr, 1.5, {})
        causal_fk_fold1 = out["per_fold_risk"][1]

        all_trades = fold0_trades + fold1_trades
        wr = sum(1 for t in all_trades if t > 0) / len(all_trades)
        naive_fk = mgr.compute_risk_params(all_trades, wr, 1.5)["risk_per_trade"]

        # Hindsight sieht den Cluster und drückt das Risiko — kausal nicht.
        assert causal_fk_fold1 > naive_fk
