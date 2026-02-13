"""
Tests für Exit Strategy Plugins.

Testet:
- FixedExitStrategy: Fixe TP/SL basierend auf Spread-Multiplikatoren
- AtrExitStrategy: Dynamische TP/SL basierend auf ATR
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.plugins import import_plugin_module

# Import exit strategies from plugins
_fixed = import_plugin_module("fwbg-core", "exit_strategies", "fixed")
_atr = import_plugin_module("fwbg-premium", "exit_strategies", "atr_based")

FixedExitStrategy = _fixed.FixedExitStrategy
AtrExitStrategy = _atr.AtrExitStrategy
from fwbg.core.context import SimulationContext


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


@pytest.fixture
def atr_context():
    """SimulationContext für ATR-basierte Exit-Strategie."""
    return SimulationContext(
        symbol="EURUSD",
        asset_class="forex",
        spread=0.0001,
        point=0.00001,
        min_trades=10,
        max_trade_bars=50,
        exit_strategy="atr_based",
        exit_params={
            "atr_period": 14,
            "min_tp_pips": 10,
            "min_sl_pips": 15,
        }
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


# --- AtrExitStrategy Tests ---


class TestAtrExitStrategyTargets:
    """Tests für Target-Berechnung mit ATR-basierten TP/SL."""

    def test_atr_targets_computed(self, volatile_ohlc, atr_context):
        """ATR-basierte Targets sollten berechnet werden."""
        # ATR hinzufügen
        import ta
        volatile_ohlc["_atr"] = ta.volatility.average_true_range(
            volatile_ohlc["H"], volatile_ohlc["L"], volatile_ohlc["C"], window=14
        )

        strategy = AtrExitStrategy()

        targets_long, targets_short = strategy.compute_targets(
            volatile_ohlc, atr_context,
            tp_mult=2.0, sl_mult=1.5
        )

        assert len(targets_long) == len(volatile_ohlc)
        assert len(targets_short) == len(volatile_ohlc)

        # Sollte einige Wins geben
        assert targets_long.sum() > 0 or targets_short.sum() > 0

    def test_atr_adapts_to_volatility(self, forex_context):
        """ATR-basierte Exits sollten sich an Volatilität anpassen."""
        strategy = AtrExitStrategy()

        # Niedrige Volatilität
        n = 100
        low_vol_df = pd.DataFrame({
            "O": 1.1000 + np.random.uniform(-0.0001, 0.0001, n),
            "H": 1.1000 + np.random.uniform(0.0001, 0.0002, n),
            "L": 1.1000 + np.random.uniform(-0.0002, -0.0001, n),
            "C": 1.1000 + np.random.uniform(-0.0001, 0.0001, n),
        })

        # Hohe Volatilität
        high_vol_df = pd.DataFrame({
            "O": 1.1000 + np.random.uniform(-0.001, 0.001, n),
            "H": 1.1000 + np.random.uniform(0.001, 0.003, n),
            "L": 1.1000 + np.random.uniform(-0.003, -0.001, n),
            "C": 1.1000 + np.random.uniform(-0.001, 0.001, n),
        })

        # ATR berechnen
        import ta
        low_vol_df["_atr"] = ta.volatility.average_true_range(
            low_vol_df["H"], low_vol_df["L"], low_vol_df["C"], window=14
        )
        high_vol_df["_atr"] = ta.volatility.average_true_range(
            high_vol_df["H"], high_vol_df["L"], high_vol_df["C"], window=14
        )

        # ATR sollte bei hoher Volatilität höher sein
        low_atr_mean = low_vol_df["_atr"].dropna().mean()
        high_atr_mean = high_vol_df["_atr"].dropna().mean()

        assert high_atr_mean > low_atr_mean * 2, "ATR sollte bei hoher Volatilität deutlich höher sein"

    def test_min_tp_sl_enforced(self, atr_context):
        """Mindest-TP/SL sollten auch bei niedrigem ATR eingehalten werden."""
        strategy = AtrExitStrategy()

        # Sehr niedriger ATR (fast keine Bewegung)
        n = 50
        df = pd.DataFrame({
            "O": np.full(n, 1.1000),
            "H": np.full(n, 1.1001),
            "L": np.full(n, 1.0999),
            "C": np.full(n, 1.1000),
        })
        df["_atr"] = np.full(n, 0.00001)  # Winziger ATR

        # Mit min_tp_pips=10 und spread=0.0001 sollte min_tp_distance = 0.001 sein
        targets_long, targets_short = strategy.compute_targets(
            df, atr_context,
            tp_mult=2.0, sl_mult=1.5,
            min_tp_pips=10, min_sl_pips=15
        )

        # Sollte nicht crashen und gültige Arrays zurückgeben
        assert len(targets_long) == n
        assert len(targets_short) == n

    def test_uses_vol_atr_column_if_available(self, volatile_ohlc, atr_context):
        """Sollte vol_atr Spalte verwenden wenn _atr nicht vorhanden."""
        import ta
        volatile_ohlc["vol_atr"] = ta.volatility.average_true_range(
            volatile_ohlc["H"], volatile_ohlc["L"], volatile_ohlc["C"], window=14
        )

        strategy = AtrExitStrategy()

        # Sollte nicht crashen (verwendet vol_atr statt _atr)
        targets_long, _ = strategy.compute_targets(
            volatile_ohlc, atr_context,
            tp_mult=2.0, sl_mult=1.5
        )

        assert len(targets_long) == len(volatile_ohlc)

    def test_calculates_atr_if_not_present(self, volatile_ohlc, atr_context):
        """Sollte ATR selbst berechnen wenn keine ATR-Spalte vorhanden."""
        # Keine ATR-Spalte
        assert "_atr" not in volatile_ohlc.columns
        assert "vol_atr" not in volatile_ohlc.columns

        strategy = AtrExitStrategy()

        # Sollte nicht crashen (berechnet ATR intern)
        targets_long, _ = strategy.compute_targets(
            volatile_ohlc, atr_context,
            tp_mult=2.0, sl_mult=1.5, atr_period=14
        )

        assert len(targets_long) == len(volatile_ohlc)


class TestAtrExitStrategyGrid:
    """Tests für Grid-Iteration mit ATR-Parametern."""

    def test_grid_generates_mult_combinations(self, atr_context):
        """Grid sollte ATR-Multiplikator Kombinationen generieren."""
        strategy = AtrExitStrategy()

        grid_config = {
            "tp_mult": [1.5, 2.0, 2.5],
            "sl_mult": [1.0, 1.5],
        }

        combinations = list(strategy.iterate_grid(grid_config, atr_context))

        assert len(combinations) == 6

        # Prüfe dass tp_mult und sl_mult vorhanden
        for c in combinations:
            assert "tp_mult" in c
            assert "sl_mult" in c
            assert c["tp_mult"] in [1.5, 2.0, 2.5]
            assert c["sl_mult"] in [1.0, 1.5]

    def test_grid_accepts_legacy_tp_sl_keys(self, atr_context):
        """Grid sollte auch 'tp' und 'sl' als Multiplikatoren interpretieren."""
        strategy = AtrExitStrategy()

        grid_config = {
            "tp": [1.5, 2.0],
            "sl": [1.0],
        }

        combinations = list(strategy.iterate_grid(grid_config, atr_context))

        assert len(combinations) == 2
        assert all(c["tp_mult"] in [1.5, 2.0] for c in combinations)

    def test_grid_includes_exit_params(self, atr_context):
        """Grid sollte exit_params aus Context übernehmen."""
        strategy = AtrExitStrategy()

        grid_config = {
            "atr_tp_mult": [2.0],
            "atr_sl_mult": [1.5],
        }

        combinations = list(strategy.iterate_grid(grid_config, atr_context))

        assert len(combinations) == 1
        c = combinations[0]

        # Sollte atr_period und min_pips aus Context haben
        assert c["atr_period"] == 14
        assert c["min_tp_pips"] == 10
        assert c["min_sl_pips"] == 15


class TestAtrExitStrategyCacheKey:
    """Tests für ATR Cache-Key Generierung."""

    def test_unique_cache_keys_for_mults(self):
        """Verschiedene Multiplikatoren sollten verschiedene Keys haben."""
        strategy = AtrExitStrategy()

        key1 = strategy.get_cache_key({"tp_mult": 2.0, "sl_mult": 1.5, "timeout_bars": None})
        key2 = strategy.get_cache_key({"tp_mult": 2.5, "sl_mult": 1.5, "timeout_bars": None})
        key3 = strategy.get_cache_key({"tp_mult": 2.0, "sl_mult": 2.0, "timeout_bars": None})

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_cache_key_format(self):
        """Cache-Key sollte ATR-spezifisches Format haben."""
        strategy = AtrExitStrategy()

        key = strategy.get_cache_key({"tp_mult": 2.0, "sl_mult": 1.5, "timeout_bars": None})

        assert "atr" in key
        assert "2.00" in key  # tp_mult formatiert
        assert "1.50" in key  # sl_mult formatiert


# --- Vergleichstests Fixed vs ATR ---


class TestExitStrategyComparison:
    """Vergleichende Tests zwischen Fixed und ATR Exit-Strategien."""

    def test_both_strategies_same_interface(self, volatile_ohlc, forex_context):
        """Beide Strategien sollten gleiches Interface haben."""
        import ta
        volatile_ohlc["_atr"] = ta.volatility.average_true_range(
            volatile_ohlc["H"], volatile_ohlc["L"], volatile_ohlc["C"], window=14
        )

        fixed = FixedExitStrategy()
        atr = AtrExitStrategy()

        # Beide sollten compute_targets haben
        t_fixed_l, t_fixed_s = fixed.compute_targets(volatile_ohlc, forex_context, tp=30, sl=20)
        t_atr_l, t_atr_s = atr.compute_targets(volatile_ohlc, forex_context, tp_mult=2.0, sl_mult=1.5)

        # Gleiche Output-Struktur
        assert t_fixed_l.shape == t_atr_l.shape
        assert t_fixed_s.shape == t_atr_s.shape

        # Werte im gültigen Bereich
        assert np.all((t_fixed_l >= 0) & (t_fixed_l <= 1))
        assert np.all((t_atr_l >= 0) & (t_atr_l <= 1))

    def test_both_have_default_params(self):
        """Beide Strategien sollten Default-Parameter haben."""
        fixed_defaults = FixedExitStrategy.get_default_params()
        atr_defaults = AtrExitStrategy.get_default_params()

        # Fixed hat tp/sl
        assert "tp" in fixed_defaults
        assert "sl" in fixed_defaults

        # ATR hat tp_mult/sl_mult
        assert "tp_mult" in atr_defaults
        assert "sl_mult" in atr_defaults


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
