"""Tests für die Trade-Simulation."""
import numpy as np
import pandas as pd
import pytest
from optimizer.simulation import simulate_pro_trade, calculate_sharpe_ratio, resolve_tp_sl_collision_m15, _m15_cache


class TestSimulateTrade:
    """Tests für simulate_pro_trade."""

    def setup_method(self):
        """Setup test data."""
        self.closes = np.array([1.0] * 100)
        self.highs = np.array([1.0] * 100)
        self.lows = np.array([1.0] * 100)
        self.atrs = np.array([0.01] * 100)
        self.spread = 0.01
        self.tp_mult = 1.0
        self.sl_mult = 1.0

    def test_both_tp_sl_hit_returns_loss(self):
        """Wenn TP und SL im selben Bar erreicht werden, sollte Loss zurückgegeben werden."""
        # Short: Entry=0.985, TP=0.98, SL=1.0
        self.highs[1] = 1.02  # SL hit
        self.lows[1] = 0.98   # TP hit

        trade_result = simulate_pro_trade(
            self.closes, self.highs, self.lows, self.atrs,
            0, -1, self.tp_mult, self.sl_mult, self.spread
        )
        assert trade_result["result"] == -1.0, "Both TP/SL hit should return loss (conservative)"
        # bars_held = 0 bedeutet Exit im selben Bar wie Entry (sofortiger Ausgang)
        assert trade_result["bars_held"] >= 0

    def test_clear_tp_hit_short(self):
        """Short Trade mit klarem TP Hit."""
        self.lows[1] = 0.95   # Clear TP hit
        self.highs[1] = 0.99  # No SL hit

        trade_result = simulate_pro_trade(
            self.closes, self.highs, self.lows, self.atrs,
            0, -1, self.tp_mult, self.sl_mult, self.spread
        )
        assert trade_result["result"] == 1.0, "Clear TP hit should return win"

    def test_clear_sl_hit_short(self):
        """Short Trade mit klarem SL Hit."""
        self.highs[1] = 1.05  # Clear SL hit
        self.lows[1] = 0.99   # No TP hit

        trade_result = simulate_pro_trade(
            self.closes, self.highs, self.lows, self.atrs,
            0, -1, self.tp_mult, self.sl_mult, self.spread
        )
        assert trade_result["result"] == -1.0, "Clear SL hit should return loss"

    def test_clear_tp_hit_long(self):
        """Long Trade mit klarem TP Hit."""
        self.highs[1] = 1.05  # Clear TP hit
        self.lows[1] = 1.01   # No SL hit

        trade_result = simulate_pro_trade(
            self.closes, self.highs, self.lows, self.atrs,
            0, 1, self.tp_mult, self.sl_mult, self.spread
        )
        assert trade_result["result"] == 1.0, "Clear TP hit should return win"

    def test_clear_sl_hit_long(self):
        """Long Trade mit klarem SL Hit."""
        self.lows[1] = 0.95   # Clear SL hit
        self.highs[1] = 1.01  # No TP hit

        trade_result = simulate_pro_trade(
            self.closes, self.highs, self.lows, self.atrs,
            0, 1, self.tp_mult, self.sl_mult, self.spread
        )
        assert trade_result["result"] == -1.0, "Clear SL hit should return loss"

    def test_no_exit_returns_none(self):
        """Trade ohne TP/SL Hit sollte None zurückgeben."""
        # Alle Bars bei Entry-Level - weder TP noch SL erreicht
        trade_result = simulate_pro_trade(
            self.closes, self.highs, self.lows, self.atrs,
            0, 1, self.tp_mult, self.sl_mult, self.spread, max_bars=10
        )
        # No exit means None is returned
        assert trade_result is None, "No exit should return None"

    def test_insufficient_bars_returns_none(self):
        """Zu wenig Bars sollte None zurückgeben."""
        short_closes = np.array([1.0, 1.0, 1.0])
        short_highs = np.array([1.0, 1.0, 1.0])
        short_lows = np.array([1.0, 1.0, 1.0])
        short_atrs = np.array([0.01, 0.01, 0.01])

        trade_result = simulate_pro_trade(
            short_closes, short_highs, short_lows, short_atrs,
            2, 1, self.tp_mult, self.sl_mult, self.spread, max_bars=10
        )
        assert trade_result is None, "Insufficient bars should return None"


class TestSharpeRatio:
    """Tests für calculate_sharpe_ratio."""

    def test_empty_returns(self):
        """Leere Returns sollten 0 ergeben."""
        assert calculate_sharpe_ratio([]) == 0.0

    def test_single_return(self):
        """Einzelner Return sollte 0 ergeben."""
        assert calculate_sharpe_ratio([0.1]) == 0.0

    def test_zero_std_returns(self):
        """Konstante Returns sollten 0 ergeben (Division by zero vermeiden)."""
        assert calculate_sharpe_ratio([0.1, 0.1, 0.1, 0.1]) == 0.0

    def test_normal_returns(self):
        """Normale Returns sollten plausiblen Sharpe ergeben."""
        returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.005]
        sharpe = calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert -100 < sharpe < 100  # Plausibilitätscheck

    def test_all_positive_returns(self):
        """Alle positiven Returns sollten positiven Sharpe ergeben."""
        returns = [0.01, 0.02, 0.015, 0.025]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe > 0

    def test_all_negative_returns(self):
        """Alle negativen Returns sollten negativen Sharpe ergeben."""
        returns = [-0.01, -0.02, -0.015, -0.025]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe < 0


class TestM15Lookup:
    """Tests für den 15-Minuten-Daten Lookup bei TP/SL Kollision."""

    def test_m15_lookup_returns_none_without_data(self):
        """Ohne M15-Daten sollte None zurückgegeben werden."""
        # Clear cache
        _m15_cache.clear()
        result = resolve_tp_sl_collision_m15("NONEXISTENT", pd.Timestamp("2024-01-01 10:00"), 1, 1.1, 0.9)
        assert result is None

    def test_simulation_with_timestamps_and_symbol(self):
        """Simulation sollte mit timestamps und symbol Parametern funktionieren."""
        closes = np.array([1.0] * 100)
        highs = np.array([1.0] * 100)
        lows = np.array([1.0] * 100)
        atrs = np.array([0.01] * 100)

        # Klarer TP hit
        highs[1] = 1.05
        lows[1] = 1.01

        # Mit Timestamps (aber ohne echte M15 Daten - Fallback)
        timestamps = pd.date_range("2024-01-01", periods=100, freq="h").values

        trade_result = simulate_pro_trade(
            closes, highs, lows, atrs, 0, 1, 1.0, 1.0, 0.01,
            timestamps=timestamps, symbol="TEST"
        )
        assert trade_result["result"] == 1.0  # Sollte trotzdem funktionieren
