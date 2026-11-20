"""
Tests für Exit-Strategy Plugin-Dispatch.

Stellt sicher, dass:
1. resolve_distances() korrekte Arrays für Fixed zurückgibt
2. _simulate_trades_core die Plugin-Distanzen verwendet (nicht hardcoded)
3. compute_targets an das Plugin delegiert
4. Neue Exit-Strategien nur resolve_distances implementieren müssen

Tests der ATR-/ORB-Exit-Strategien gehören zum separat installierten
fwbg-premium-Package.
"""

import numpy as np
import pandas as pd
import pytest

from fwbg.core.context import SimulationContext
from fwbg.plugins import import_plugin_module

_fixed = import_plugin_module("fwbg-core", "exit_strategies", "fixed")
assert _fixed is not None, "fwbg-core exit_strategies plugin must be available"

FixedExitStrategy = _fixed.FixedExitStrategy


# --- Fixtures ---


def _make_ohlc_df(n=100, base=1.1000, volatility=0.002):
    """OHLC-Daten mit kontrollierter Volatilität."""
    np.random.seed(42)
    closes = base + np.cumsum(np.random.normal(0, 0.0005, n))
    df = pd.DataFrame(
        {
            "O": closes - np.random.uniform(0, volatility / 2, n),
            "H": closes + np.abs(np.random.normal(0, volatility, n)),
            "L": closes - np.abs(np.random.normal(0, volatility, n)),
            "C": closes,
        },
        index=pd.date_range("2023-01-01", periods=n, freq="1h"),
    )
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
    return df


def _make_fixed_ctx(spread=0.0001):
    return SimulationContext(
        symbol="EURUSD",
        asset_class="forex",
        spread=spread,
        point=0.00001,
        min_trades=5,
        max_trade_bars=50,
        exit_strategy="fixed",
    )


# --- resolve_distances Tests ---


class TestResolveDistancesFixed:
    """Tests für FixedExitStrategy.resolve_distances."""

    def test_returns_constant_arrays(self):
        """Fixed-Distanzen müssen für alle Bars gleich sein."""
        df = _make_ohlc_df()
        ctx = _make_fixed_ctx(spread=0.0001)
        strategy = FixedExitStrategy()

        tp_dists, sl_dists = strategy.resolve_distances(df, tp=30, sl=20, ctx=ctx)

        assert len(tp_dists) == len(df)
        assert len(sl_dists) == len(df)
        assert np.all(tp_dists == 0.0001 * 30)  # spread * tp
        assert np.all(sl_dists == 0.0001 * 20)  # spread * sl

    def test_default_params(self):
        """Fixed liefert tp/sl als Default-Parameter."""
        defaults = FixedExitStrategy.get_default_params()
        assert "tp" in defaults
        assert "sl" in defaults

    def test_distance_proportional_to_spread(self):
        """Distanz muss proportional zum Spread sein."""
        df = _make_ohlc_df(n=10)
        strategy = FixedExitStrategy()

        ctx1 = _make_fixed_ctx(spread=0.0001)
        ctx2 = _make_fixed_ctx(spread=0.0002)

        tp1, _ = strategy.resolve_distances(df, tp=10, sl=10, ctx=ctx1)
        tp2, _ = strategy.resolve_distances(df, tp=10, sl=10, ctx=ctx2)

        assert tp2[0] == pytest.approx(tp1[0] * 2)


# --- Plugin Dispatch Tests ---


class TestPluginDispatchConsistency:
    """Stellt sicher, dass compute_targets und _simulate_trades_core
    beide über das Plugin dispatchen — keine hardcodierten Fallbacks."""

    def test_compute_targets_dispatches_to_plugin(self):
        """compute_targets muss an das Plugin delegieren."""
        from fwbg.optimization.targets import compute_targets

        df = _make_ohlc_df(n=100)
        ctx = _make_fixed_ctx(spread=0.0001)
        targets_l, targets_s, has_l, has_s = compute_targets(df, tp=30, sl=20, ctx=ctx)

        assert len(targets_l) == len(df)
        assert len(targets_s) == len(df)
        assert targets_l.sum() + targets_s.sum() > 0, (
            "Fixed-Targets sollten bei volatilen Daten Wins produzieren"
        )

    def test_targets_follow_plugin_distances(self):
        """Die Targets müssen sich mit den vom Plugin gelieferten Distanzen ändern."""
        from fwbg.optimization.targets import compute_targets

        df = _make_ohlc_df(n=200)
        ctx = _make_fixed_ctx(spread=0.0001)

        near_l, near_s, _, _ = compute_targets(df, tp=5, sl=40, ctx=ctx)
        far_l, far_s, _, _ = compute_targets(df, tp=200, sl=40, ctx=ctx)

        # A nearby TP is hit far more often than a TP of 200 spreads.
        assert near_l.sum() + near_s.sum() > far_l.sum() + far_s.sum()

    def test_simulate_trades_uses_plugin_distances(self):
        """_simulate_trades_core muss Plugin-Distanzen verwenden.

        REGRESSION GUARD: Ohne Plugin-Dispatch würde _simulate_trades_core
        feste Distanzen verwenden und tp/sl des Aufrufs ignorieren.
        """
        from fwbg.optimization.targets import _simulate_trades_core

        df = _make_ohlc_df(n=200)
        df["_regime"] = np.int8(7)

        n = len(df)
        probs_long = np.zeros((n, 2))
        probs_long[:, 1] = 0.9

        ctx = _make_fixed_ctx()
        ctx.long_enabled = True
        ctx.short_enabled = False
        narrow = _simulate_trades_core(
            df,
            probs_long,
            None,
            1,
            None,
            ct_long=0.5,
            ct_short=0.5,
            tp=10,
            sl=10,
            ctx=ctx,
        )
        wide = _simulate_trades_core(
            df,
            probs_long,
            None,
            1,
            None,
            ct_long=0.5,
            ct_short=0.5,
            tp=60,
            sl=60,
            ctx=ctx,
        )

        assert len(narrow["trades"]) > 0, "Fixed sollte Trades produzieren"
        assert len(wide["trades"]) > 0, "Fixed sollte Trades produzieren"

        narrow_avg = np.mean(np.abs([t["pnl_raw"] for t in narrow["trades"]]))
        wide_avg = np.mean(np.abs([t["pnl_raw"] for t in wide["trades"]]))
        assert narrow_avg != pytest.approx(wide_avg, rel=0.1), (
            f"PnL-Magnitudes müssen den Distanzen folgen: "
            f"narrow={narrow_avg:.6f}, wide={wide_avg:.6f}"
        )


class TestSimulateProTradeDistances:
    """Tests für simulate_pro_trade mit vorberechneten Distanzen."""

    def test_takes_distances_directly(self):
        """simulate_pro_trade muss fertige Distanzen akzeptieren."""
        from fwbg.simulation.trade import simulate_pro_trade

        closes = np.array([1.1000] * 50)
        highs = np.array([1.1010] * 50)
        lows = np.array([1.0990] * 50)

        # Starker Aufwärtstrend für sicheren TP-Hit
        for i in range(1, 50):
            closes[i] = 1.1000 + i * 0.001
            highs[i] = closes[i] + 0.001
            lows[i] = closes[i] - 0.001

        # Direkte Distanz: 5 Pips TP, 10 Pips SL
        trade = simulate_pro_trade(
            closes,
            highs,
            lows,
            idx=0,
            direction=1,
            tp_distance=0.0005,
            sl_distance=0.001,
            spread=0.0001,
            opens=closes,
        )

        assert trade is not None
        assert trade["tp_distance"] == 0.0005
        assert trade["sl_distance"] == 0.001

    def test_no_atr_parameter(self):
        """simulate_pro_trade darf keinen atrs-Parameter mehr haben."""
        from fwbg.simulation.trade import simulate_pro_trade
        import inspect

        sig = inspect.signature(simulate_pro_trade)
        param_names = list(sig.parameters.keys())

        assert "atrs" not in param_names, "atrs-Parameter muss entfernt sein"
        assert "atr_exits" not in param_names, "atr_exits-Parameter muss entfernt sein"
        assert "tp_m" not in param_names, "tp_m muss durch tp_distance ersetzt sein"
        assert "sl_m" not in param_names, "sl_m muss durch sl_distance ersetzt sein"
        assert "tp_distance" in param_names
        assert "sl_distance" in param_names


# --- Numba Compilation Smoke Tests ---


class TestNumbaCompilation:
    """Stellt sicher, dass alle @njit-Funktionen der Exit-Strategien kompilierbar sind.

    Fängt ab:
    - Stale Numba-Cache nach Signaturänderungen
    - Typ-Fehler in Numba-Argumenten
    - Fehlende/umbenannte Imports in @njit-Funktionen
    """

    def _make_numba_arrays(self, n=200):
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        opens = prices + np.random.randn(n) * 0.1
        closes = prices + np.random.randn(n) * 0.1
        highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 0.3)
        lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 0.3)
        return opens, closes, highs, lows

    def test_simulate_trade_numba_compiles(self):
        """_simulate_trade_numba muss mit korrekten Typen kompilierbar sein."""
        from fwbg.simulation import _simulate_trade_numba

        opens, closes, highs, lows = self._make_numba_arrays()
        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens,
            closes,
            highs,
            lows,
            idx=10,
            direction=1,
            tp_distance=1.0,
            sl_distance=0.5,
            spread=0.05,
            slippage=0.025,
            max_bars=100,
            timeout_bars=0,
        )
        assert result in (1.0, -1.0, 0.0)


# --- Strategy JSON End-to-End Tests ---


class TestStrategyJsonEndToEnd:
    """Für jede Strategy-JSON-Datei: exit_strategy + compute_targets funktioniert.

    Fängt ab:
    - Strategy referenziert nicht-existierendes Exit-Strategy-Plugin
    - compute_targets crasht mit den tatsächlichen Grid-Parametern
    - Numba-Kompilierung schlägt fehl bei realen Parametern
    """

    @staticmethod
    def _get_strategy_files():
        import glob

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return glob.glob(os.path.join(root, "strategies", "configs", "*.json"))

    def test_all_strategies_produce_targets(self):
        """Jede Strategy-JSON muss mit ihren exit_strategies Targets produzieren."""
        import os
        from fwbg.core import get_exit_strategy, GridParams
        from fwbg.pipeline.registry import get_registry

        get_registry().auto_discover()

        strategy_files = self._get_strategy_files()
        if not strategy_files:
            pytest.skip("Keine Strategy-Dateien gefunden")

        # Realistisches Test-DataFrame
        n = 300
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame(
            {
                "O": prices + np.random.randn(n) * 0.1,
                "H": prices + np.abs(np.random.randn(n) * 0.3),
                "L": prices - np.abs(np.random.randn(n) * 0.3),
                "C": prices + np.random.randn(n) * 0.05,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="h"),
        )
        df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
        df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))

        # ATR-Spalte vorberechnen (für ATR-Strategies)
        import ta

        df["_atr"] = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=14
        )

        failures = []

        for path in strategy_files:
            from fwbg.core.config import StrategyConfig

            strategy = StrategyConfig.from_json_file(path)
            strategy_name = os.path.basename(path)

            for es_cfg in strategy.exit_strategies:
                exit_strategy_name = es_cfg.name
                exit_params = dict(es_cfg.params)
                tp = exit_params.pop("tp_mult", 1.0)
                sl = exit_params.pop("sl_mult", 1.0)

                try:
                    exit_strategy_class = get_exit_strategy(exit_strategy_name)
                    exit_strategy_inst = exit_strategy_class()

                    ctx = SimulationContext(
                        symbol="TEST",
                        asset_class="FOREX",
                        spread=0.05,
                        point=0.01,
                        exit_params=exit_params,
                    )

                    grid_params = GridParams(
                        tp_value=float(tp),
                        sl_value=float(sl),
                        extra=exit_params,
                    )

                    result = exit_strategy_inst.compute_targets(
                        df,
                        ctx,
                        params=grid_params,
                    )
                    targets_l, targets_s = result[0], result[1]

                    assert len(targets_l) == len(df), (
                        f"{strategy_name}: targets_long hat falsche Länge"
                    )
                except Exception as e:
                    failures.append(
                        f"{strategy_name}/{exit_strategy_name} (tp={tp}, sl={sl}): {e}"
                    )

        assert not failures, f"Exit-Strategy-Fehler in Strategy-JSONs:\n" + "\n".join(
            f"  - {f}" for f in failures
        )


import os


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
