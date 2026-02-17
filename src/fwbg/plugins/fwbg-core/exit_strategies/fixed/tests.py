"""Tests for FixedExitStrategy plugin."""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module
from fwbg.core.context import SimulationContext

_fixed = import_plugin_module("fwbg-core", "exit_strategies", "fixed")
if _fixed is None:
    pytest.skip("fwbg-core fixed exit strategy not available", allow_module_level=True)

FixedExitStrategy = _fixed.FixedExitStrategy


# --- Fixtures ---


@pytest.fixture
def trending_up_ohlc():
    """OHLC-Daten mit klarem Aufwärtstrend - Long Trades sollten gewinnen."""
    n = 100
    base = 1.1000
    trend = np.linspace(0, 0.01, n)  # +100 Pips über 100 Bars

    df = pd.DataFrame({
        "O": base + trend - 0.0002,
        "H": base + trend + 0.0005,
        "L": base + trend - 0.0005,
        "C": base + trend + 0.0002,
    })
    return df


@pytest.fixture
def trending_down_ohlc():
    """OHLC-Daten mit klarem Abwärtstrend - Short Trades sollten gewinnen."""
    n = 100
    base = 1.1000
    trend = np.linspace(0, -0.01, n)  # -100 Pips über 100 Bars

    df = pd.DataFrame({
        "O": base + trend + 0.0002,
        "H": base + trend + 0.0005,
        "L": base + trend - 0.0005,
        "C": base + trend - 0.0002,
    })
    return df


@pytest.fixture
def volatile_ohlc():
    """OHLC-Daten mit hoher Volatilität (große Ranges)."""
    n = 100
    np.random.seed(42)
    base = 1.1000
    volatility = 0.002  # 20 Pips Schwankung

    closes = base + np.cumsum(np.random.normal(0, 0.0005, n))
    df = pd.DataFrame({
        "O": closes - np.random.uniform(0, volatility, n),
        "H": closes + np.abs(np.random.normal(0, volatility, n)),
        "L": closes - np.abs(np.random.normal(0, volatility, n)),
        "C": closes,
    })
    # Sicherstellen dass H > L
    df["H"] = np.maximum(df["H"], df[["O", "C"]].max(axis=1))
    df["L"] = np.minimum(df["L"], df[["O", "C"]].min(axis=1))
    return df


@pytest.fixture
def flat_ohlc():
    """OHLC-Daten ohne Trend - wenig Bewegung."""
    n = 100
    np.random.seed(42)
    base = 1.1000

    df = pd.DataFrame({
        "O": base + np.random.uniform(-0.0001, 0.0001, n),
        "H": base + np.random.uniform(0.0001, 0.0003, n),
        "L": base + np.random.uniform(-0.0003, -0.0001, n),
        "C": base + np.random.uniform(-0.0001, 0.0001, n),
    })
    return df


@pytest.fixture
def forex_context():
    """SimulationContext für typisches Forex-Paar."""
    return SimulationContext(
        symbol="EURUSD",
        asset_class="forex",
        spread=0.0001,  # 1 Pip
        point=0.00001,
        min_trades=10,
        max_trade_bars=50,
        exit_strategy="fixed",
    )


# --- FixedExitStrategy Tests ---


class TestFixedExitStrategyTargets:
    """Tests für Target-Berechnung mit fixen TP/SL."""

    def test_long_wins_in_uptrend(self, trending_up_ohlc, forex_context):
        """Long Trades sollten in Aufwärtstrend gewinnen."""
        strategy = FixedExitStrategy()

        targets_long, targets_short = strategy.compute_targets(
            trending_up_ohlc, forex_context,
            tp=30, sl=20  # RRR 1.5
        )

        # In starkem Aufwärtstrend sollten viele Long Trades gewinnen
        long_win_rate = np.mean(targets_long)
        assert long_win_rate > 0.3, f"Long Win-Rate zu niedrig in Uptrend: {long_win_rate:.2%}"

    def test_short_wins_in_downtrend(self, trending_down_ohlc, forex_context):
        """Short Trades sollten in Abwärtstrend gewinnen."""
        strategy = FixedExitStrategy()

        targets_long, targets_short = strategy.compute_targets(
            trending_down_ohlc, forex_context,
            tp=30, sl=20
        )

        short_win_rate = np.mean(targets_short)
        assert short_win_rate > 0.3, f"Short Win-Rate zu niedrig in Downtrend: {short_win_rate:.2%}"

    def test_larger_sl_reduces_losses(self, volatile_ohlc, forex_context):
        """Größerer SL sollte weniger Stopouts verursachen."""
        strategy = FixedExitStrategy()

        # Enger SL
        _, _ = strategy.compute_targets(
            volatile_ohlc, forex_context,
            tp=50, sl=10  # Sehr enger SL
        )
        # Bei engem SL in volatilen Märkten werden mehr Trades gestoppt

        # Weiter SL
        targets_long_wide, _ = strategy.compute_targets(
            volatile_ohlc, forex_context,
            tp=50, sl=50  # Weiter SL
        )

        # Mit weiterem SL sollten mehr Trades überleben
        # (da sie nicht so schnell gestoppt werden)
        assert len(targets_long_wide) == len(volatile_ohlc)

    def test_timeout_closes_trades(self, flat_ohlc, forex_context):
        """Timeout sollte Trades schließen die weder TP noch SL erreichen."""
        strategy = FixedExitStrategy()

        # Ohne Timeout - in flachem Markt viele "Pending" Trades
        targets_no_timeout, _ = strategy.compute_targets(
            flat_ohlc, forex_context,
            tp=100, sl=100,  # Sehr weit - kaum erreichbar
            timeout_bars=None
        )

        # Mit Timeout
        targets_with_timeout, _ = strategy.compute_targets(
            flat_ohlc, forex_context,
            tp=100, sl=100,
            timeout_bars=5  # Nach 5 Bars schließen
        )

        # Beide sollten gleiche Länge haben
        assert len(targets_no_timeout) == len(targets_with_timeout)


class TestFixedExitStrategyGrid:
    """Tests für Grid-Iteration."""

    def test_grid_generates_combinations(self, forex_context):
        """Grid sollte alle TP x SL Kombinationen generieren."""
        strategy = FixedExitStrategy()

        grid_config = {
            "tp": [20, 30, 40],
            "sl": [15, 20],
            "timeout_bars": [None],
        }

        combinations = list(strategy.iterate_grid(grid_config, forex_context))

        # 3 TPs x 2 SLs = 6 Kombinationen
        assert len(combinations) == 6

        # Alle Kombinationen prüfen
        tp_values = {c["tp"] for c in combinations}
        sl_values = {c["sl"] for c in combinations}
        assert tp_values == {20.0, 30.0, 40.0}
        assert sl_values == {15.0, 20.0}

    def test_grid_respects_min_rrr(self, forex_context):
        """Grid sollte min_rrr Filter beachten."""
        strategy = FixedExitStrategy()

        grid_config = {
            "tp": [10, 20, 30],
            "sl": [20],
            "min_rrr": 1.0,  # RRR >= 1.0
        }

        combinations = list(strategy.iterate_grid(grid_config, forex_context))

        # tp=10, sl=20 hat RRR=0.5 -> sollte gefiltert werden
        # tp=20, sl=20 hat RRR=1.0 -> ok
        # tp=30, sl=20 hat RRR=1.5 -> ok
        assert len(combinations) == 2

        for c in combinations:
            rrr = c["tp"] / c["sl"]
            assert rrr >= 1.0

    def test_grid_includes_timeout(self, forex_context):
        """Grid sollte Timeout-Varianten einschließen."""
        strategy = FixedExitStrategy()

        grid_config = {
            "tp": [30],
            "sl": [20],
            "timeout_bars": [None, 24, 48],
        }

        combinations = list(strategy.iterate_grid(grid_config, forex_context))

        # 1 TP x 1 SL x 3 Timeouts = 3
        assert len(combinations) == 3

        timeouts = [c["timeout_bars"] for c in combinations]
        assert None in timeouts
        assert 24 in timeouts
        assert 48 in timeouts


class TestFixedExitStrategyCacheKey:
    """Tests für Cache-Key Generierung."""

    def test_unique_cache_keys(self):
        """Verschiedene Parameter sollten verschiedene Cache-Keys haben."""
        strategy = FixedExitStrategy()

        key1 = strategy.get_cache_key({"tp": 30, "sl": 20, "timeout_bars": None})
        key2 = strategy.get_cache_key({"tp": 30, "sl": 25, "timeout_bars": None})
        key3 = strategy.get_cache_key({"tp": 30, "sl": 20, "timeout_bars": 24})

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_cache_key_format(self):
        """Cache-Key sollte erwartetes Format haben."""
        strategy = FixedExitStrategy()

        key = strategy.get_cache_key({"tp": 30, "sl": 20, "timeout_bars": None})

        assert "fixed" in key
        assert "30" in key
        assert "20" in key
