"""Tests für die kausale Reporting-Kalibrierung (Plan: Look-Ahead-Fix)."""
import pandas as pd
import pytest

from fwbg.optimization.causal_reporting import (
    circuit_breaker_mask,
    compute_causal_reporting,
    equity_replay,
    monte_carlo_summary,
    sized_returns,
    slim_trades,
)
from fwbg.simulation.trade import (
    monte_carlo_equity_from_returns,
    monte_carlo_permutation_test,
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


class RrrRiskManager(StubRiskManager):
    """fk proportional to the RRR it is calibrated with."""

    def compute_risk_params(self, trades, win_rate, rrr, rv_values=None, **params):
        result = super().compute_risk_params(trades, win_rate, rrr, rv_values, **params)
        return {**result, "risk_per_trade": 0.01 * rrr}


class TestPerFoldRrr:
    """Fold i is sized with the RRR of the setting it was replayed with."""

    def test_each_fold_is_sized_with_its_own_rrr(self):
        folds = [
            {**_fold([(1.0, 0.003), (-1.0, -0.002)]), "rrr": 4.0},
            {**_fold([(1.0, 0.004)]), "rrr": 1.0},
            {**_fold([(-1.0, -0.001)]), "rrr": 3.0},
        ]
        out = compute_causal_reporting(folds, RrrRiskManager(), 2.0, {})
        assert out["per_fold_risk"] == [None, pytest.approx(0.01), pytest.approx(0.03)]

    def test_folds_without_rrr_fall_back_to_the_given_rrr(self):
        folds = [_fold([(1.0, 0.003), (-1.0, -0.002)]), _fold([(1.0, 0.004)])]
        out = compute_causal_reporting(folds, RrrRiskManager(), 2.0, {})
        assert out["per_fold_risk"] == [None, pytest.approx(0.02)]


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


class TestSharedHelpers:
    """Helpers shared by process_symbol and the holdout look (extracted from process.py)."""

    def test_sized_returns_apply_the_circuit_breaker_and_prior_loss_scale(self):
        trades = [{"result": r, "pnl_raw": p} for r, p in [(-1.0, -1.0), (-1.0, -2.0), (1.0, 4.0), (1.0, 5.0)]]
        cb = {"enabled": True, "pause_after_losses": 2, "pause_bars": 1}

        returns, pnl, taken = sized_returns(trades, 0.02, cb, [-2.0, -4.0, 3.0])

        # Two losses pause one trade; the scale is the prior mean loss (3).
        assert taken == [0, 1, 3]
        assert pnl == [-1.0, -2.0, 5.0]
        assert returns == pytest.approx([0.02 * -1 / 3, 0.02 * -2 / 3, 0.02 * 5 / 3])

    def test_sized_returns_ignore_a_disabled_circuit_breaker(self):
        trades = [{"result": -1.0, "pnl_raw": -1.0}] * 3
        cb = {"enabled": False, "pause_after_losses": 1, "pause_bars": 5}

        _, _, taken = sized_returns(trades, 0.01, cb, [-1.0])

        assert taken == [0, 1, 2]

    def test_monte_carlo_summary_keys(self):
        pnl = [2.0, -1.0, 2.0, 2.0, -1.0] * 6
        returns = [0.01 * p for p in pnl]

        summary = monte_carlo_summary(pnl, returns)

        perm = monte_carlo_permutation_test(pnl, n_permutations=1000)
        equity = monte_carlo_equity_from_returns(returns, n_simulations=500)
        assert summary == {
            "p_value": perm["p_value"],
            "is_significant": perm["is_significant"],
            "percentile": perm["percentile"],
            "equity_median": equity["median_equity"],
            "equity_p5": equity["p5_equity"],
            "equity_p95": equity["p95_equity"],
            "bankruptcy_rate": equity["bankruptcy_rate"],
        }

    def test_equity_replay_thins_long_curves_and_keeps_the_last_point(self):
        times = [str(t) for t in pd.date_range("2024-01-01", periods=50, freq="h")]

        full = equity_replay(times, [0.001] * 50)
        thin = equity_replay(times, [0.001] * 50, max_points=10)

        assert len(thin["equity_points"]) < len(full["equity_points"])
        assert thin["equity_points"][-1] == full["equity_points"][-1]

    def test_slim_trades_keep_the_reported_fields(self):
        trade = {"entry_time": "a", "exit_time": "b", "direction": "LONG", "result": 1.0,
                 "pnl_raw": 2.0, "exit_reason": "tp", "mae": 0.5}

        assert slim_trades([trade]) == [{k: v for k, v in trade.items() if k != "mae"}]
