"""Tests für die Trade-Simulation."""
import numpy as np
import pytest
from optimizer.simulation import simulate_pro_trade, calculate_sharpe_ratio


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

        res, bars = simulate_pro_trade(
            self.closes, self.highs, self.lows, self.atrs,
            0, -1, self.tp_mult, self.sl_mult, self.spread
        )
        assert res == -1.0, "Both TP/SL hit should return loss (conservative)"
        assert bars == 1

    def test_clear_tp_hit_short(self):
        """Short Trade mit klarem TP Hit."""
        self.lows[1] = 0.95   # Clear TP hit
        self.highs[1] = 0.99  # No SL hit

        res, _ = simulate_pro_trade(
            self.closes, self.highs, self.lows, self.atrs,
            0, -1, self.tp_mult, self.sl_mult, self.spread
        )
        assert res == 1.0, "Clear TP hit should return win"

    def test_clear_sl_hit_short(self):
        """Short Trade mit klarem SL Hit."""
        self.highs[1] = 1.05  # Clear SL hit
        self.lows[1] = 0.99   # No TP hit

        res, _ = simulate_pro_trade(
            self.closes, self.highs, self.lows, self.atrs,
            0, -1, self.tp_mult, self.sl_mult, self.spread
        )
        assert res == -1.0, "Clear SL hit should return loss"

    def test_clear_tp_hit_long(self):
        """Long Trade mit klarem TP Hit."""
        self.highs[1] = 1.05  # Clear TP hit
        self.lows[1] = 1.01   # No SL hit

        res, _ = simulate_pro_trade(
            self.closes, self.highs, self.lows, self.atrs,
            0, 1, self.tp_mult, self.sl_mult, self.spread
        )
        assert res == 1.0, "Clear TP hit should return win"

    def test_clear_sl_hit_long(self):
        """Long Trade mit klarem SL Hit."""
        self.lows[1] = 0.95   # Clear SL hit
        self.highs[1] = 1.01  # No TP hit

        res, _ = simulate_pro_trade(
            self.closes, self.highs, self.lows, self.atrs,
            0, 1, self.tp_mult, self.sl_mult, self.spread
        )
        assert res == -1.0, "Clear SL hit should return loss"

    def test_no_exit_returns_neutral(self):
        """Trade ohne TP/SL Hit sollte 0 zurückgeben."""
        # Alle Bars bei Entry-Level - weder TP noch SL erreicht
        res, bars = simulate_pro_trade(
            self.closes, self.highs, self.lows, self.atrs,
            0, 1, self.tp_mult, self.sl_mult, self.spread, max_bars=10
        )
        assert res == 0.0, "No exit should return neutral"

    def test_insufficient_bars_returns_invalid(self):
        """Zu wenig Bars sollte 0 zurückgeben."""
        short_closes = np.array([1.0, 1.0, 1.0])
        short_highs = np.array([1.0, 1.0, 1.0])
        short_lows = np.array([1.0, 1.0, 1.0])
        short_atrs = np.array([0.01, 0.01, 0.01])

        res, _ = simulate_pro_trade(
            short_closes, short_highs, short_lows, short_atrs,
            2, 1, self.tp_mult, self.sl_mult, self.spread, max_bars=10
        )
        assert res == 0.0, "Insufficient bars should return invalid"


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
