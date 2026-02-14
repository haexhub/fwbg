"""
Tests for equity simulation consistency.

Verifies:
1. simulate_equity and monte_carlo_equity_simulation produce consistent results
   for the same risk_per_trade/rrr/trades (below compound cap)
2. Monte Carlo equity stays within reasonable bounds for realistic parameters
3. In process_symbol, Monte Carlo uses the risk_per_trade from the risk
   manager plugin — NOT a hardcoded quick estimate

BUG CONTEXT: Monte Carlo equity simulation was called with a hardcoded Kelly
estimate (~0.05), while the actual risk_per_trade from the plugin was the
DD-adjusted value (~0.009). This produced equity values in the BILLIONS.
"""
import numpy as np
import pytest

from fwbg.simulation.equity import simulate_equity
from fwbg.simulation.trade import monte_carlo_equity_simulation


# =============================================================================
# Fixtures
# =============================================================================

def _make_trades(n_wins, n_losses):
    """Create a list of binary trade results."""
    return [1.0] * n_wins + [-1.0] * n_losses


# =============================================================================
# Test: simulate_equity and MC equity must agree
# =============================================================================

class TestEquityFunctionConsistency:
    """simulate_equity and MC observed_equity must agree for the same inputs."""

    def test_observed_equity_matches_simulate_equity(self):
        """MC's observed_equity must match simulate_equity for same kelly/rrr.

        These are two independent implementations of equity compounding.
        If they diverge, one of them is wrong.
        """
        trades = _make_trades(70, 30)
        kelly = 0.01
        rrr = 1.5

        # simulate_equity (from equity.py) — with high cap to avoid cap effects
        eq_result = simulate_equity(trades, kelly, rrr, compound_cap=1e15)
        final_eq = eq_result["final_equity"]

        # monte_carlo_equity_simulation's observed_equity (from trade.py)
        mc_result = monte_carlo_equity_simulation(
            trades, kelly, rrr, n_simulations=10
        )
        mc_observed = mc_result["observed_equity"]

        assert final_eq == pytest.approx(mc_observed, rel=1e-6), (
            f"simulate_equity ({final_eq:.4f}) must match MC observed_equity "
            f"({mc_observed:.4f}) for same inputs"
        )

    def test_consistency_with_realistic_kelly(self):
        """Both functions agree with DD-adjusted kelly (~1% risk)."""
        trades = _make_trades(350, 150)  # 70% WR, 500 trades
        kelly = 0.0094  # Typical DD-adjusted kelly
        rrr = 0.75

        eq_result = simulate_equity(trades, kelly, rrr, compound_cap=1e15)
        mc_result = monte_carlo_equity_simulation(
            trades, kelly, rrr, n_simulations=10
        )

        assert eq_result["final_equity"] == pytest.approx(
            mc_result["observed_equity"], rel=1e-6
        )

    def test_consistency_with_high_kelly(self):
        """Both agree even with high kelly (the buggy value)."""
        trades = _make_trades(350, 150)
        kelly = 0.05  # Pre-risk-management quick kelly
        rrr = 0.75

        eq_result = simulate_equity(trades, kelly, rrr, compound_cap=1e15)
        mc_result = monte_carlo_equity_simulation(
            trades, kelly, rrr, n_simulations=10
        )

        assert eq_result["final_equity"] == pytest.approx(
            mc_result["observed_equity"], rel=1e-6
        )


# =============================================================================
# Test: MC equity must be reasonable for realistic parameters
# =============================================================================

class TestMonteCarloEquityBounds:
    """Monte Carlo equity must produce reasonable values for realistic kelly."""

    def test_mc_equity_under_1m_for_low_kelly(self):
        """With DD-adjusted kelly (~1%), 500 trades should NOT reach millions.

        REGRESSION: The bug produced equity of 1.2 BILLION because MC used
        kelly=0.05 instead of kelly=0.009.
        """
        trades = _make_trades(350, 150)  # 70% WR
        kelly = 0.0094
        rrr = 0.75

        mc_result = monte_carlo_equity_simulation(
            trades, kelly, rrr, n_simulations=100
        )

        # Starting from 100, with 1% kelly, equity should stay well under 1M
        assert mc_result["median_equity"] < 1_000_000, (
            f"MC median equity {mc_result['median_equity']:.0f} should be < 1M "
            f"for kelly={kelly}, but got unrealistic value"
        )
        assert mc_result["observed_equity"] < 1_000_000

    def test_mc_equity_under_1m_for_1750_trades(self):
        """Gold-like scenario: 1752 trades, 70% WR, kelly=0.009.

        This mimics the exact GOLD parameters from run 20260214_041022.
        """
        n_wins = int(1752 * 0.697)  # 1221
        n_losses = 1752 - n_wins    # 531
        trades = _make_trades(n_wins, n_losses)
        kelly = 0.009421875
        rrr = 0.75

        mc_result = monte_carlo_equity_simulation(
            trades, kelly, rrr, n_simulations=50
        )

        # With correct kelly, equity should be in the thousands, not billions
        assert mc_result["median_equity"] < 100_000, (
            f"MC median equity {mc_result['median_equity']:.0f} is unrealistic "
            f"for kelly={kelly}, 1752 trades"
        )
        # Verify it's at least growing (positive edge)
        assert mc_result["median_equity"] > 100, "Should have positive growth"

    def test_mc_equity_explodes_with_high_kelly(self):
        """Demonstrate that kelly=0.05 produces absurd results — this is the bug.

        With kelly=0.05 (pre-risk-management value) instead of kelly=0.009
        (DD-adjusted), equity reaches billions. This test documents the failure
        mode so we never accidentally use the wrong kelly again.
        """
        n_wins = int(1752 * 0.697)
        n_losses = 1752 - n_wins
        trades = _make_trades(n_wins, n_losses)
        kelly_wrong = 0.05  # Quick Kelly before DD adjustment
        rrr = 0.75

        mc_result = monte_carlo_equity_simulation(
            trades, kelly_wrong, rrr, n_simulations=10
        )

        # This SHOULD produce absurd results — documenting the failure mode
        assert mc_result["observed_equity"] > 1_000_000_000, (
            "With kelly=0.05 on 1752 trades, equity should reach billions — "
            "this demonstrates why using the wrong kelly is catastrophic"
        )


# =============================================================================
# Test: process_symbol must use DD-adjusted kelly for Monte Carlo
# =============================================================================

class TestProcessSymbolRiskConsistency:
    """Monte Carlo in process_symbol must use the risk manager plugin's value.

    BUG: process.py computed a hardcoded Kelly estimate, ran Monte Carlo with
    that value, then LATER computed the real risk_per_trade via the risk manager
    plugin. The Monte Carlo must use the plugin-computed value.
    """

    def test_risk_manager_before_monte_carlo(self):
        """Risk management must run BEFORE Monte Carlo equity simulation.

        This ensures MC uses the plugin-computed risk_per_trade, not a
        hardcoded quick estimate.
        """
        import inspect
        from fwbg.optimization.process import process_symbol

        source = inspect.getsource(process_symbol)

        risk_mgr_pos = source.find("get_risk_manager(")
        mc_equity_pos = source.find("monte_carlo_equity_simulation(")

        assert risk_mgr_pos != -1, "get_risk_manager call not found"
        assert mc_equity_pos != -1, "monte_carlo_equity_simulation call not found"

        assert risk_mgr_pos < mc_equity_pos, (
            f"Risk management (pos {risk_mgr_pos}) must come BEFORE "
            f"Monte Carlo equity simulation (pos {mc_equity_pos})."
        )

    def test_risk_manager_before_sharpe_calmar(self):
        """Risk management must run BEFORE Sharpe/Calmar calculations.

        Both the 'not_significant' and 'ok' paths must use the plugin's
        risk_per_trade for metric computation.
        """
        import inspect
        from fwbg.optimization.process import process_symbol

        source = inspect.getsource(process_symbol)

        risk_mgr_pos = source.find("get_risk_manager(")
        first_sharpe_pos = source.find("calculate_sharpe_ratio(")

        assert risk_mgr_pos != -1, "get_risk_manager call not found"
        assert first_sharpe_pos != -1, "calculate_sharpe_ratio call not found"

        assert risk_mgr_pos < first_sharpe_pos, (
            f"Risk management (pos {risk_mgr_pos}) must come BEFORE first "
            f"Sharpe calculation (pos {first_sharpe_pos})."
        )

    def test_risk_per_trade_from_plugin_before_mc(self):
        """fk must be assigned from risk_result before MC equity call."""
        import inspect
        from fwbg.optimization.process import process_symbol

        source = inspect.getsource(process_symbol)
        lines = source.split('\n')
        risk_assign_line = None
        mc_equity_line = None

        for i, line in enumerate(lines):
            if 'fk = risk_result["risk_per_trade"]' in line:
                risk_assign_line = i
            if 'monte_carlo_equity_simulation(' in line:
                mc_equity_line = i

        assert risk_assign_line is not None, (
            "fk = risk_result[\"risk_per_trade\"] not found — "
            "risk must come from the plugin, not hardcoded"
        )
        assert mc_equity_line is not None, "MC equity call not found"

        assert risk_assign_line < mc_equity_line, (
            f"fk = risk_result[\"risk_per_trade\"] (line {risk_assign_line}) "
            f"must come BEFORE monte_carlo_equity_simulation (line {mc_equity_line})"
        )

    def test_no_hardcoded_kelly_formula(self):
        """process_symbol must NOT contain a hardcoded Kelly criterion formula.

        The formula (win_rate * rrr - (1 - win_rate)) / rrr belongs in the
        risk management plugin, not in process.py.
        """
        import inspect
        from fwbg.optimization.process import process_symbol

        source = inspect.getsource(process_symbol)

        # The Kelly formula pattern
        assert "full_kelly" not in source, (
            "process_symbol contains 'full_kelly' — Kelly computation "
            "must be in the risk management plugin, not hardcoded"
        )
        assert "kelly_fraction" not in source, (
            "process_symbol contains 'kelly_fraction' — Kelly parameters "
            "must be in the risk management plugin, not hardcoded"
        )
