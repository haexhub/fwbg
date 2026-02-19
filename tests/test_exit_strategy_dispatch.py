"""
Tests für Exit-Strategy Plugin-Dispatch.

Stellt sicher, dass:
1. resolve_distances() korrekte Arrays für Fixed und ATR zurückgibt
2. _simulate_trades_core die Plugin-Distanzen verwendet (nicht hardcoded)
3. compute_targets an das Plugin delegiert
4. Neue Exit-Strategien nur resolve_distances implementieren müssen

Diese Tests hätten den Bug erkannt, bei dem der Optimizer mit ATR-Exits
alle Folds übersprang, weil select_features die falschen Distanzen berechnete.
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.core.context import SimulationContext
from fwbg.plugins import import_plugin_module

_fixed = import_plugin_module("fwbg-core", "exit_strategies", "fixed")
_atr = import_plugin_module("fwbg-premium", "exit_strategies", "atr_based")

if _fixed is None:
    pytest.skip("fwbg-core exit_strategies plugin not available", allow_module_level=True)
if _atr is None:
    pytest.skip("fwbg-premium exit_strategies plugin not available", allow_module_level=True)

FixedExitStrategy = _fixed.FixedExitStrategy
AtrExitStrategy = _atr.AtrExitStrategy


# --- Fixtures ---

def _make_ohlc_df(n=100, base=1.1000, volatility=0.002):
    """OHLC-Daten mit kontrollierter Volatilität."""
    np.random.seed(42)
    closes = base + np.cumsum(np.random.normal(0, 0.0005, n))
    df = pd.DataFrame({
        "O": closes - np.random.uniform(0, volatility / 2, n),
        "H": closes + np.abs(np.random.normal(0, volatility, n)),
        "L": closes - np.abs(np.random.normal(0, volatility, n)),
        "C": closes,
    }, index=pd.date_range("2023-01-01", periods=n, freq="1h"))
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
    return df


def _make_fixed_ctx(spread=0.0001):
    return SimulationContext(
        symbol="EURUSD", asset_class="forex",
        spread=spread, point=0.00001,
        min_trades=5, max_trade_bars=50,
        exit_strategy="fixed",
    )


def _make_atr_ctx(spread=0.0001):
    return SimulationContext(
        symbol="EURUSD", asset_class="forex",
        spread=spread, point=0.00001,
        min_trades=5, max_trade_bars=50,
        exit_strategy="atr_based",
        exit_params={
            "atr_period": 14,
            "min_tp_pips": 10,
            "min_sl_pips": 15,
        }
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

    def test_distance_proportional_to_spread(self):
        """Distanz muss proportional zum Spread sein."""
        df = _make_ohlc_df(n=10)
        strategy = FixedExitStrategy()

        ctx1 = _make_fixed_ctx(spread=0.0001)
        ctx2 = _make_fixed_ctx(spread=0.0002)

        tp1, _ = strategy.resolve_distances(df, tp=10, sl=10, ctx=ctx1)
        tp2, _ = strategy.resolve_distances(df, tp=10, sl=10, ctx=ctx2)

        assert tp2[0] == pytest.approx(tp1[0] * 2)


class TestResolveDistancesATR:
    """Tests für AtrExitStrategy.resolve_distances."""

    def test_returns_per_bar_arrays(self):
        """ATR-Distanzen müssen pro Bar variieren."""
        import ta
        df = _make_ohlc_df(n=100)
        df["_atr"] = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=14
        )
        ctx = _make_atr_ctx()
        strategy = AtrExitStrategy()

        tp_dists, sl_dists = strategy.resolve_distances(df, tp=2.0, sl=1.5, ctx=ctx)

        assert len(tp_dists) == len(df)
        # ATR-basiert: Distanzen müssen variieren (nicht alle gleich)
        assert not np.all(tp_dists == tp_dists[20]), "ATR-Distanzen dürfen nicht konstant sein"

    def test_respects_min_distance(self):
        """Bei niedrigem ATR müssen Mindest-Distanzen eingehalten werden."""
        n = 50
        df = pd.DataFrame({
            "O": np.full(n, 1.1000),
            "H": np.full(n, 1.1001),
            "L": np.full(n, 1.0999),
            "C": np.full(n, 1.1000),
        }, index=pd.date_range("2023-01-01", periods=n, freq="1h"))
        df["_atr"] = np.full(n, 0.000001)  # Winziger ATR

        ctx = _make_atr_ctx(spread=0.0001)
        strategy = AtrExitStrategy()

        tp_dists, sl_dists = strategy.resolve_distances(df, tp=2.0, sl=1.5, ctx=ctx)

        # min_tp_pips=10 → min_tp_distance = 0.0001 * 10 = 0.001
        # min_sl_pips=15 → min_sl_distance = 0.0001 * 15 = 0.0015
        assert np.all(tp_dists >= 0.001 - 1e-10)
        assert np.all(sl_dists >= 0.0015 - 1e-10)

    def test_uses_vol_atr_column(self):
        """Muss vol_atr verwenden wenn _atr nicht vorhanden."""
        import ta
        df = _make_ohlc_df(n=50)
        df["vol_atr"] = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=14
        )
        # Kein _atr column!
        assert "_atr" not in df.columns

        ctx = _make_atr_ctx()
        strategy = AtrExitStrategy()

        tp_dists, _ = strategy.resolve_distances(df, tp=2.0, sl=1.5, ctx=ctx)
        assert len(tp_dists) == len(df)
        # Sollte nicht alle NaN/0 sein
        assert tp_dists[20:].sum() > 0

    def test_computes_atr_if_no_column(self):
        """Muss ATR selbst berechnen wenn keine ATR-Spalte vorhanden."""
        df = _make_ohlc_df(n=50)
        assert "_atr" not in df.columns
        assert "vol_atr" not in df.columns

        ctx = _make_atr_ctx()
        strategy = AtrExitStrategy()

        tp_dists, sl_dists = strategy.resolve_distances(df, tp=2.0, sl=1.5, ctx=ctx)
        assert len(tp_dists) == len(df)


class TestFixedVsATRDistances:
    """Vergleichstests: Fixed und ATR müssen unterschiedliche Distanzen liefern."""

    def test_atr_distances_differ_from_fixed(self):
        """ATR-Distanzen müssen sich von Fixed-Distanzen unterscheiden.

        REGRESSION GUARD: Dieser Test hätte den Bug erkannt, bei dem
        ATR-Multiplikatoren (0.5) als Spread-Multiplikatoren interpretiert
        wurden, was zu 0.5-Pip-Distanzen statt ~50-Pip-Distanzen führte.
        """
        import ta
        df = _make_ohlc_df(n=100)
        df["_atr"] = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=14
        )

        fixed_ctx = _make_fixed_ctx(spread=0.0001)
        atr_ctx = _make_atr_ctx(spread=0.0001)

        fixed = FixedExitStrategy()
        atr = AtrExitStrategy()

        # ATR mult = 0.5 → mit typischem ATR ~0.002: dist ≈ 0.001 = 10 Pips
        # Fixed mult = 0.5 → dist = 0.0001 * 0.5 = 0.00005 = 0.5 Pips
        fixed_tp, _ = fixed.resolve_distances(df, tp=0.5, sl=0.5, ctx=fixed_ctx)
        atr_tp, _ = atr.resolve_distances(df, tp=0.5, sl=0.5, ctx=atr_ctx)

        # ATR-Distanzen müssen WESENTLICH größer sein als Fixed-Distanzen
        # bei gleichen Multiplikatoren (weil ATR >> spread)
        atr_mean = np.nanmean(atr_tp[20:])  # skip warmup
        fixed_val = fixed_tp[0]

        assert atr_mean > fixed_val * 5, (
            f"ATR-Distanz ({atr_mean:.6f}) muss deutlich größer sein als "
            f"Fixed-Distanz ({fixed_val:.6f}) bei gleichem Multiplikator"
        )


# --- Plugin Dispatch Tests ---

class TestPluginDispatchConsistency:
    """Stellt sicher, dass compute_targets und _simulate_trades_core
    beide über das Plugin dispatchen — keine hardcodierten Fallbacks."""

    def test_compute_targets_dispatches_to_plugin(self):
        """compute_targets muss an das Plugin delegieren."""
        from fwbg.optimization.targets import compute_targets

        import ta
        df = _make_ohlc_df(n=100)
        df["_atr"] = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=14
        )

        # Mit ATR-Context
        ctx = _make_atr_ctx(spread=0.0001)
        targets_l, targets_s, has_l, has_s = compute_targets(df, tp=2.0, sl=1.5, ctx=ctx)

        assert len(targets_l) == len(df)
        assert len(targets_s) == len(df)
        # Bei volatilen Daten mit ATR-Exits sollte es Wins geben
        assert targets_l.sum() + targets_s.sum() > 0, (
            "ATR-basierte Targets sollten Wins produzieren"
        )

    def test_atr_targets_different_from_spread_based(self):
        """ATR-basierte Targets müssen sich von Spread-basierten unterscheiden.

        REGRESSION GUARD: Dieser Test hätte den Hauptbug erkannt.
        Mit ATR-Multiplikatoren 0.5 als Spread-Multiplikatoren interpretiert,
        produziert ~0.5-Pip-Distanzen → alle Targets = 0.
        """
        from fwbg.optimization.targets import compute_targets

        import ta
        df = _make_ohlc_df(n=200)
        df["_atr"] = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=14
        )

        # Fixed context mit kleinen Multiplikatoren (wie ATR-Werte)
        fixed_ctx = _make_fixed_ctx(spread=0.0001)
        fixed_ctx.exit_strategy = "fixed"

        # ATR context mit gleichen Multiplikatoren
        atr_ctx = _make_atr_ctx(spread=0.0001)

        # Fixed: tp=0.5 → spread * 0.5 = 0.00005 (~0.5 Pips) → fast alles Win
        tgt_fixed_l, tgt_fixed_s, _, _ = compute_targets(df, tp=0.5, sl=0.5, ctx=fixed_ctx)

        # ATR: tp=0.5 → ATR * 0.5 ≈ 0.001 (~10 Pips) → realistischere Targets
        tgt_atr_l, tgt_atr_s, _, _ = compute_targets(df, tp=0.5, sl=0.5, ctx=atr_ctx)

        # Fixed mit 0.5-Pip-Distanz vs ATR mit ~10-Pip-Distanz
        # Ergebnisse MÜSSEN sich deutlich unterscheiden
        fixed_total = tgt_fixed_l.sum() + tgt_fixed_s.sum()
        atr_total = tgt_atr_l.sum() + tgt_atr_s.sum()

        assert fixed_total != atr_total, (
            f"Fixed und ATR müssen unterschiedliche Ergebnisse liefern "
            f"bei gleichen Multiplikatoren. Fixed={fixed_total}, ATR={atr_total}"
        )

        # ATR mit realistischen Distanzen sollte Wins produzieren
        assert atr_total > 0, "ATR-Targets sollten bei normaler Volatilität Wins haben"

    def test_simulate_trades_uses_plugin_distances(self):
        """_simulate_trades_core muss Plugin-Distanzen verwenden.

        REGRESSION GUARD: Ohne Plugin-Dispatch würde _simulate_trades_core
        Spread-basierte Distanzen verwenden, was bei ATR-Exits zu falschen
        PnL-Werten führt.
        """
        from fwbg.optimization.targets import _simulate_trades_core

        import ta
        df = _make_ohlc_df(n=200)
        df["_atr"] = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=14
        )
        df["_regime"] = np.int8(7)

        # Erstelle dummy probabilities (alle Bars signalisieren Long)
        n = len(df)
        probs_long = np.zeros((n, 2))
        probs_long[:, 1] = 0.9

        # Simuliere mit Fixed-Context
        fixed_ctx = _make_fixed_ctx()
        fixed_ctx.long_enabled = True
        fixed_ctx.short_enabled = False
        result_fixed = _simulate_trades_core(
            df, probs_long, None, 1, None,
            ct_long=0.5, ct_short=0.5,
            tp=30, sl=20, ctx=fixed_ctx,
        )

        # Simuliere mit ATR-Context
        atr_ctx = _make_atr_ctx()
        atr_ctx.long_enabled = True
        atr_ctx.short_enabled = False
        result_atr = _simulate_trades_core(
            df, probs_long, None, 1, None,
            ct_long=0.5, ct_short=0.5,
            tp=2.0, sl=1.5, ctx=atr_ctx,
        )

        # Beide sollten Trades produzieren
        assert len(result_fixed["trades"]) > 0, "Fixed sollte Trades produzieren"
        assert len(result_atr["trades"]) > 0, "ATR sollte Trades produzieren"

        # PnL-Werte müssen sich unterscheiden
        # (weil die TP/SL-Distanzen völlig anders sind)
        fixed_pnls = [t["pnl_raw"] for t in result_fixed["trades"]]
        atr_pnls = [t["pnl_raw"] for t in result_atr["trades"]]

        fixed_avg_abs = np.mean(np.abs(fixed_pnls))
        atr_avg_abs = np.mean(np.abs(atr_pnls))

        # ATR-Distanzen (ATR ≈ 0.002 × 2.0 = 0.004) sind viel größer als
        # Fixed-Distanzen (0.0001 × 30 = 0.003), also PnL pro Trade unterschiedlich
        # Hauptsache: ATR-Modus produziert nicht die gleichen Werte wie Fixed
        assert fixed_avg_abs != pytest.approx(atr_avg_abs, rel=0.1), (
            f"PnL-Magnitudes müssen sich unterscheiden: "
            f"Fixed={fixed_avg_abs:.6f}, ATR={atr_avg_abs:.6f}"
        )


class TestFeatureSelectionWithATR:
    """Stellt sicher, dass Feature-Selection mit ATR-Exits funktioniert.

    REGRESSION GUARD: Der Hauptbug war, dass select_features() mit
    ATR-Multiplikatoren als Spread-Multiplikatoren Targets berechnete.
    Ergebnis: alle Targets = 0 → keine Features selektiert → Fold übersprungen.
    """

    def test_atr_targets_not_all_zero(self):
        """ATR-Targets dürfen nicht alle 0 sein bei normaler Volatilität."""
        from fwbg.optimization.targets import compute_targets_cached

        import ta
        df = _make_ohlc_df(n=500)
        df["_atr"] = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=14
        )

        ctx = _make_atr_ctx(spread=0.0001)

        result = compute_targets_cached(
            df, tp=0.5, sl=0.8, ctx=ctx,
            exit_strategy_mode="atr_based",
        )
        targets_l, targets_s = result[0], result[1]

        total_wins = targets_l.sum() + targets_s.sum()
        assert total_wins > 0, (
            f"ATR-Targets mit tp=0.5, sl=0.8 sollten Wins produzieren, "
            f"bekam {total_wins} (Long: {targets_l.sum()}, Short: {targets_s.sum()})"
        )

    def test_small_atr_multipliers_still_produce_targets(self):
        """Auch kleine ATR-Multiplikatoren (0.3) müssen Targets produzieren.

        REGRESSION GUARD: Das war der exakte Fehler — ATR-Multiplikatoren
        wie 0.3 wurden als Spread-Multiplikatoren interpretiert:
        0.0001 × 0.3 = 0.00003 (~0.3 Pips) → alle Targets 0.
        Korrekt: ATR × 0.3 ≈ 0.002 × 0.3 = 0.0006 (~6 Pips) → realistische Targets.
        """
        from fwbg.optimization.targets import compute_targets_cached

        import ta
        df = _make_ohlc_df(n=500)
        df["_atr"] = ta.volatility.average_true_range(
            df["H"], df["L"], df["C"], window=14
        )

        ctx = _make_atr_ctx(spread=0.0001)

        result = compute_targets_cached(
            df, tp=0.3, sl=0.5, ctx=ctx,
            exit_strategy_mode="atr_based",
        )
        targets_l, targets_s = result[0], result[1]

        total_wins = targets_l.sum() + targets_s.sum()
        assert total_wins > 0, (
            f"Selbst kleine ATR-Multiplikatoren (tp=0.3) sollten bei "
            f"normaler Volatilität Targets produzieren, bekam {total_wins}"
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
            closes, highs, lows,
            idx=0, direction=1,
            tp_distance=0.0005, sl_distance=0.001,
            spread=0.0001, opens=closes,
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
            opens, closes, highs, lows,
            idx=10, direction=1,
            tp_distance=1.0, sl_distance=0.5,
            spread=0.05, slippage=0.025,
            max_bars=100, timeout_bars=0,
        )
        assert result in (1.0, -1.0, 0.0)

    def test_atr_compute_targets_numba_compiles(self):
        """_compute_targets_atr_numba muss kompilierbar sein."""
        opens, closes, highs, lows = self._make_numba_arrays()

        import ta
        atr_series = ta.volatility.average_true_range(
            pd.Series(highs), pd.Series(lows), pd.Series(closes), window=14
        )
        atr_values = np.nan_to_num(atr_series.values.astype(np.float64), nan=0.0)

        targets_l, targets_s = _atr._compute_targets_atr_numba(
            opens, closes, highs, lows, atr_values,
            tp_mult=2.0, sl_mult=1.5,
            spread=0.05, slippage=0.025,
            min_tp_distance=0.5, min_sl_distance=0.75,
            max_bars=200, timeout_bars=0,
        )
        assert len(targets_l) == len(closes)
        assert targets_l.sum() + targets_s.sum() > 0

    def test_atr_compute_targets_with_durations_numba_compiles(self):
        """_compute_targets_atr_with_durations_numba muss kompilierbar sein."""
        opens, closes, highs, lows = self._make_numba_arrays()

        import ta
        atr_series = ta.volatility.average_true_range(
            pd.Series(highs), pd.Series(lows), pd.Series(closes), window=14
        )
        atr_values = np.nan_to_num(atr_series.values.astype(np.float64), nan=0.0)

        targets_l, targets_s, dur_l, dur_s = _atr._compute_targets_atr_with_durations_numba(
            opens, closes, highs, lows, atr_values,
            tp_mult=2.0, sl_mult=1.5,
            spread=0.05, slippage=0.025,
            min_tp_distance=0.5, min_sl_distance=0.75,
            max_bars=200, timeout_bars=0,
        )
        assert len(dur_l) == len(closes)
        assert dur_l.dtype == np.int64

    def test_atr_adaptive_timeout_numba_compiles(self):
        """_compute_targets_atr_adaptive_timeout_numba muss kompilierbar sein."""
        opens, closes, highs, lows = self._make_numba_arrays()

        import ta
        atr_series = ta.volatility.average_true_range(
            pd.Series(highs), pd.Series(lows), pd.Series(closes), window=14
        )
        atr_values = np.nan_to_num(atr_series.values.astype(np.float64), nan=0.0)
        atr_ma = pd.Series(atr_values).rolling(50, min_periods=1).mean().values.astype(np.float64)

        targets_l, targets_s = _atr._compute_targets_atr_adaptive_timeout_numba(
            opens, closes, highs, lows,
            atr_values, atr_ma,
            tp_mult=2.0, sl_mult=1.5,
            spread=0.05, slippage=0.025,
            min_tp_distance=0.5, min_sl_distance=0.75,
            max_bars=200,
            base_timeout=48, min_timeout=12, max_timeout=96,
        )
        assert len(targets_l) == len(closes)


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
        return glob.glob(os.path.join(root, "strategies", "*.json"))

    def test_all_strategies_produce_targets(self):
        """Jede Strategy-JSON muss mit ihren Grid-Params Targets produzieren."""
        import os
        from fwbg.core import get_exit_strategy, GridParams, discover_plugins

        discover_plugins()

        strategy_files = self._get_strategy_files()
        assert strategy_files, "Keine Strategy-Dateien gefunden"

        # Realistisches Test-DataFrame
        n = 300
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "O": prices + np.random.randn(n) * 0.1,
            "H": prices + np.abs(np.random.randn(n) * 0.3),
            "L": prices - np.abs(np.random.randn(n) * 0.3),
            "C": prices + np.random.randn(n) * 0.05,
        }, index=pd.date_range("2024-01-01", periods=n, freq="h"))
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
            exit_strategy_name = strategy.exit_strategy
            exit_params = strategy.exit_params

            # Irgendein Grid-Set nehmen (alle sollten funktionieren)
            for asset_class, grid_config in strategy.grids.items():
                tp_values = grid_config.tp or [1.0]
                sl_values = grid_config.sl or [1.0]

                # Ersten TP/SL aus dem Grid testen
                tp = tp_values[0]
                sl = sl_values[0]

                try:
                    exit_strategy_class = get_exit_strategy(exit_strategy_name)
                    exit_strategy = exit_strategy_class()

                    ctx = SimulationContext(
                        symbol="TEST",
                        asset_class=asset_class,
                        spread=0.05,
                        point=0.01,
                        exit_params=exit_params,
                    )

                    grid_params = GridParams(
                        tp_value=float(tp),
                        sl_value=float(sl),
                        extra=exit_params,
                    )

                    result = exit_strategy.compute_targets(
                        df, ctx, params=grid_params,
                    )
                    targets_l, targets_s = result[0], result[1]

                    assert len(targets_l) == len(df), (
                        f"{strategy_name}/{asset_class}: targets_long hat falsche Länge"
                    )
                except Exception as e:
                    failures.append(f"{strategy_name}/{asset_class} (tp={tp}, sl={sl}): {e}")
                break  # Ein Grid pro Strategy reicht

        assert not failures, (
            f"Exit-Strategy-Fehler in Strategy-JSONs:\n" +
            "\n".join(f"  - {f}" for f in failures)
        )


import os


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
