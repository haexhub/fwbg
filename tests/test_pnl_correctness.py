"""
Tests for PnL correctness: verifies that PnL uses actual pnl_raw values
(accounting for TP/SL asymmetry) rather than binary ±1.0 win/loss counting.

Uses DETERMINISTIC synthetic data where we know exactly which trades
hit TP and which hit SL, so we can verify exact PnL values.

These tests would have caught the original bug where:
- trades.append(trade["result"])  # ±1.0 only
- pnl = sum(trades)              # counts wins minus losses
"""
import numpy as np
import pandas as pd
import pytest

from fwbg.optimization.targets import simulate_trades_sequential
from fwbg.core.context import SimulationContext


# =============================================================================
# FIXTURES: Deterministic data with known trade outcomes
# =============================================================================


def _make_ctx(spread=0.0001):
    """Minimal SimulationContext."""
    return SimulationContext(
        symbol="EURUSD",
        asset_class="FOREX",
        spread=spread,
        point=0.00001,
        min_trades=1,
        long_enabled=True,
        short_enabled=True,
        exit_strategy="fixed",
        grid_ct=[0.5],
        grid_tp=[10],
        grid_sl=[10],
    )


def _make_winning_long_df(n_trades=10, spread=0.0001, tp_mult=10, sl_mult=10):
    """Creates data where every Long trade hits TP.

    Strategy: Each bar after signal has High well above entry + TP distance,
    and Low stays above entry - SL distance.
    """
    tp_dist = spread * tp_mult
    sl_dist = spread * sl_mult
    slippage = spread * 0.5

    bars = []
    for i in range(n_trades * 3):
        base = 1.1000
        if i % 3 == 0:
            # Signal bar: normal bar
            bars.append({"O": base, "H": base + 0.0001, "L": base - 0.0001, "C": base})
        elif i % 3 == 1:
            # Entry+exit bar: price jumps up to hit TP, stays above SL
            entry = base + spread + slippage
            bars.append({
                "O": base,
                "H": entry + tp_dist + 0.0001,  # Above TP
                "L": entry - sl_dist + 0.0002,  # Above SL (no SL hit)
                "C": base + tp_dist,
            })
        else:
            # Gap bar (trade already exited)
            bars.append({"O": base, "H": base + 0.0001, "L": base - 0.0001, "C": base})

    dates = pd.date_range("2023-01-01", periods=len(bars), freq="1h")
    df = pd.DataFrame(bars, index=dates)
    df["_atr"] = 0.001
    df["_regime"] = np.int8(7)
    return df


def _make_losing_long_df(n_trades=10, spread=0.0001, tp_mult=10, sl_mult=10):
    """Creates data where every Long trade hits SL.

    Strategy: Each bar after signal has Low below entry - SL distance,
    and High stays below entry + TP distance.
    """
    tp_dist = spread * tp_mult
    sl_dist = spread * sl_mult
    slippage = spread * 0.5

    bars = []
    for i in range(n_trades * 3):
        base = 1.1000
        if i % 3 == 0:
            # Signal bar
            bars.append({"O": base, "H": base + 0.0001, "L": base - 0.0001, "C": base})
        elif i % 3 == 1:
            # Entry+exit bar: price drops to hit SL, stays below TP
            entry = base + spread + slippage
            bars.append({
                "O": base,
                "H": entry + tp_dist - 0.0002,  # Below TP (no TP hit)
                "L": entry - sl_dist - 0.0001,  # Below SL (SL hit)
                "C": base - sl_dist,
            })
        else:
            bars.append({"O": base, "H": base + 0.0001, "L": base - 0.0001, "C": base})

    dates = pd.date_range("2023-01-01", periods=len(bars), freq="1h")
    df = pd.DataFrame(bars, index=dates)
    df["_atr"] = 0.001
    df["_regime"] = np.int8(7)
    return df


def _make_mixed_df(n_wins=6, n_losses=4, spread=0.0001, tp_mult=10, sl_mult=20):
    """Creates data with exactly n_wins TP hits and n_losses SL hits for Long trades.

    Alternates wins then losses to ensure sequential simulation captures all.
    """
    tp_dist = spread * tp_mult
    sl_dist = spread * sl_mult
    slippage = spread * 0.5

    bars = []
    trade_idx = 0
    total = n_wins + n_losses

    for i in range((total) * 3):
        base = 1.1000
        if i % 3 == 0:
            bars.append({"O": base, "H": base + 0.0001, "L": base - 0.0001, "C": base})
        elif i % 3 == 1:
            entry = base + spread + slippage
            if trade_idx < n_wins:
                # Win: TP hit
                bars.append({
                    "O": base,
                    "H": entry + tp_dist + 0.0001,
                    "L": entry - sl_dist + 0.0002,
                    "C": base + tp_dist,
                })
            else:
                # Loss: SL hit
                bars.append({
                    "O": base,
                    "H": entry + tp_dist - 0.0002,
                    "L": entry - sl_dist - 0.0001,
                    "C": base - sl_dist,
                })
            trade_idx += 1
        else:
            bars.append({"O": base, "H": base + 0.0001, "L": base - 0.0001, "C": base})

    dates = pd.date_range("2023-01-01", periods=len(bars), freq="1h")
    df = pd.DataFrame(bars, index=dates)
    df["_atr"] = 0.001
    df["_regime"] = np.int8(7)
    return df


# =============================================================================
# TESTS
# =============================================================================


class TestTradeStructure:
    """Trades must be dicts with 'result' and 'pnl_raw'."""

    def test_trades_are_dicts(self):
        """Each trade from simulate_trades_sequential is a dict."""
        df = _make_winning_long_df(n_trades=5)
        ctx = _make_ctx()
        probs = np.zeros((len(df), 2))
        probs[:, 1] = 0.9  # high confidence on every bar

        result = simulate_trades_sequential(
            df, probs, None, 1, None,
            ct=0.5, tp=10, sl=10, ctx=ctx,
        )

        trades = result["trades"]
        assert len(trades) > 0, "Should generate trades"

        for i, t in enumerate(trades):
            assert isinstance(t, dict), f"Trade {i}: expected dict, got {type(t)}"
            assert "result" in t, f"Trade {i}: missing 'result'"
            assert "pnl_raw" in t, f"Trade {i}: missing 'pnl_raw'"
            assert t["result"] in (1.0, -1.0), f"Trade {i}: result must be ±1.0"

    def test_all_wins_have_positive_pnl(self):
        """When all trades hit TP, every pnl_raw should be positive."""
        df = _make_winning_long_df(n_trades=5)
        ctx = _make_ctx()
        probs = np.zeros((len(df), 2))
        probs[:, 1] = 0.9

        result = simulate_trades_sequential(
            df, probs, None, 1, None,
            ct=0.5, tp=10, sl=10, ctx=ctx,
        )

        trades = result["trades"]
        for t in trades:
            assert t["result"] == 1.0, f"Expected TP hit, got result={t['result']}"
            assert t["pnl_raw"] > 0, f"TP hit must have positive pnl_raw, got {t['pnl_raw']}"

    def test_all_losses_have_negative_pnl(self):
        """When all trades hit SL, every pnl_raw should be negative."""
        df = _make_losing_long_df(n_trades=5)
        ctx = _make_ctx()
        probs = np.zeros((len(df), 2))
        probs[:, 1] = 0.9

        result = simulate_trades_sequential(
            df, probs, None, 1, None,
            ct=0.5, tp=10, sl=10, ctx=ctx,
        )

        trades = result["trades"]
        for t in trades:
            assert t["result"] == -1.0, f"Expected SL hit, got result={t['result']}"
            assert t["pnl_raw"] < 0, f"SL hit must have negative pnl_raw, got {t['pnl_raw']}"


class TestPnLAsymmetry:
    """PnL must correctly reflect TP/SL asymmetry."""

    def test_pnl_not_equal_to_binary_when_asymmetric(self):
        """REGRESSION GUARD: With TP=10, SL=20, pnl_raw sum MUST differ from win-loss count.

        This is the EXACT test that would have caught the original bug.

        6 wins × TP=10 × spread = 6 × 0.001 = 0.006
        4 losses × SL=20 × spread = 4 × 0.002 = 0.008
        Binary: 6 - 4 = +2
        Real: ~0.006 - 0.008 = ~-0.002 (negative!)
        """
        df = _make_mixed_df(n_wins=6, n_losses=4, tp_mult=10, sl_mult=20)
        ctx = _make_ctx()
        probs = np.zeros((len(df), 2))
        probs[:, 1] = 0.9

        result = simulate_trades_sequential(
            df, probs, None, 1, None,
            ct=0.5, tp=10, sl=20, ctx=ctx,
        )

        trades = result["trades"]
        assert len(trades) > 0, "Should generate trades"

        binary_pnl = sum(t["result"] for t in trades)
        real_pnl = sum(t["pnl_raw"] for t in trades)

        # They MUST differ when TP != SL
        assert binary_pnl != pytest.approx(real_pnl, abs=0.0001), (
            f"PnL ({real_pnl:.6f}) must NOT equal binary count ({binary_pnl:.1f}) "
            f"when TP != SL"
        )

    def test_loss_magnitude_proportional_to_sl(self):
        """With TP=10, SL=20: each loss should cost ~2x what each win earns."""
        df = _make_mixed_df(n_wins=5, n_losses=5, tp_mult=10, sl_mult=20)
        ctx = _make_ctx()
        probs = np.zeros((len(df), 2))
        probs[:, 1] = 0.9

        result = simulate_trades_sequential(
            df, probs, None, 1, None,
            ct=0.5, tp=10, sl=20, ctx=ctx,
        )

        trades = result["trades"]
        wins = [t for t in trades if t["result"] == 1.0]
        losses = [t for t in trades if t["result"] == -1.0]

        if not wins or not losses:
            pytest.skip("Need both wins and losses")

        avg_win = np.mean([t["pnl_raw"] for t in wins])
        avg_loss = np.mean([t["pnl_raw"] for t in losses])

        # SL is 2x TP, so loss magnitude should be ~2x win magnitude
        ratio = abs(avg_loss) / abs(avg_win)
        assert 1.8 < ratio < 2.2, (
            f"Loss/Win magnitude ratio should be ~2.0, got {ratio:.2f} "
            f"(avg_win={avg_win:.6f}, avg_loss={avg_loss:.6f})"
        )

    def test_symmetric_tp_sl_gives_proportional_pnl(self):
        """With TP=SL=15, binary sum and real sum should be proportional."""
        df = _make_mixed_df(n_wins=7, n_losses=3, tp_mult=15, sl_mult=15)
        ctx = _make_ctx()
        probs = np.zeros((len(df), 2))
        probs[:, 1] = 0.9

        result = simulate_trades_sequential(
            df, probs, None, 1, None,
            ct=0.5, tp=15, sl=15, ctx=ctx,
        )

        trades = result["trades"]
        if not trades:
            pytest.skip("No trades generated")

        binary_pnl = sum(t["result"] for t in trades)
        real_pnl = sum(t["pnl_raw"] for t in trades)

        # With TP=SL, sign must match
        if abs(binary_pnl) > 0.5:
            assert np.sign(binary_pnl) == np.sign(real_pnl), (
                f"Symmetric TP/SL: signs must match. "
                f"Binary={binary_pnl:.1f}, Real={real_pnl:.6f}"
            )


class TestPnLValues:
    """Verify exact PnL values for known trade outcomes."""

    def test_tp_pnl_equals_tp_distance(self):
        """A single TP hit should produce pnl_raw ≈ tp_distance."""
        spread = 0.0001
        tp_mult = 10
        tp_dist = spread * tp_mult  # 0.001

        df = _make_winning_long_df(n_trades=3, spread=spread, tp_mult=tp_mult, sl_mult=10)
        ctx = _make_ctx(spread=spread)
        probs = np.zeros((len(df), 2))
        probs[:, 1] = 0.9

        result = simulate_trades_sequential(
            df, probs, None, 1, None,
            ct=0.5, tp=tp_mult, sl=10, ctx=ctx,
            return_detailed=True,
        )

        trades = result["trades"]
        assert len(trades) > 0

        for t in trades:
            assert t["result"] == 1.0
            # pnl_raw should be close to tp_distance
            assert t["pnl_raw"] == pytest.approx(tp_dist, rel=0.01), (
                f"TP hit pnl_raw should be ~{tp_dist:.6f}, got {t['pnl_raw']:.6f}"
            )

    def test_sl_pnl_equals_sl_distance(self):
        """A single SL hit should produce pnl_raw ≈ -sl_distance."""
        spread = 0.0001
        sl_mult = 15
        sl_dist = spread * sl_mult  # 0.0015

        df = _make_losing_long_df(n_trades=3, spread=spread, tp_mult=10, sl_mult=sl_mult)
        ctx = _make_ctx(spread=spread)
        probs = np.zeros((len(df), 2))
        probs[:, 1] = 0.9

        result = simulate_trades_sequential(
            df, probs, None, 1, None,
            ct=0.5, tp=10, sl=sl_mult, ctx=ctx,
        )

        trades = result["trades"]
        assert len(trades) > 0

        for t in trades:
            assert t["result"] == -1.0
            assert t["pnl_raw"] == pytest.approx(-sl_dist, rel=0.01), (
                f"SL hit pnl_raw should be ~{-sl_dist:.6f}, got {t['pnl_raw']:.6f}"
            )

    def test_total_pnl_exact_for_known_outcomes(self):
        """With 6 wins (TP=10) and 4 losses (SL=20), verify total PnL.

        Expected: 6 × 0.001 - 4 × 0.002 = 0.006 - 0.008 = -0.002
        Binary would show: 6 - 4 = +2
        """
        spread = 0.0001
        tp_mult, sl_mult = 10, 20
        n_wins, n_losses = 6, 4

        df = _make_mixed_df(n_wins=n_wins, n_losses=n_losses,
                           tp_mult=tp_mult, sl_mult=sl_mult, spread=spread)
        ctx = _make_ctx(spread=spread)
        probs = np.zeros((len(df), 2))
        probs[:, 1] = 0.9

        result = simulate_trades_sequential(
            df, probs, None, 1, None,
            ct=0.5, tp=tp_mult, sl=sl_mult, ctx=ctx,
        )

        trades = result["trades"]
        real_pnl = sum(t["pnl_raw"] for t in trades)
        binary_pnl = sum(t["result"] for t in trades)

        actual_wins = sum(1 for t in trades if t["result"] == 1.0)
        actual_losses = sum(1 for t in trades if t["result"] == -1.0)

        # Expected PnL: wins × tp_dist - losses × sl_dist
        expected_pnl = actual_wins * (spread * tp_mult) - actual_losses * (spread * sl_mult)

        assert real_pnl == pytest.approx(expected_pnl, rel=0.05), (
            f"Total PnL should be ~{expected_pnl:.6f} ({actual_wins}W×{spread*tp_mult:.4f} - "
            f"{actual_losses}L×{spread*sl_mult:.4f}), got {real_pnl:.6f}"
        )

        # Binary MUST differ
        if actual_wins != actual_losses:
            assert abs(binary_pnl) > abs(real_pnl) * 100, (
                f"Binary PnL ({binary_pnl}) should be orders of magnitude larger "
                f"than real PnL ({real_pnl:.6f})"
            )


class TestWinRate:
    """Win rate must still use result field, not pnl_raw."""

    def test_win_rate_from_result_field(self):
        """Win-Rate = count(result==1.0) / total, not based on pnl_raw sign."""
        df = _make_mixed_df(n_wins=7, n_losses=3, tp_mult=10, sl_mult=10)
        ctx = _make_ctx()
        probs = np.zeros((len(df), 2))
        probs[:, 1] = 0.9

        result = simulate_trades_sequential(
            df, probs, None, 1, None,
            ct=0.5, tp=10, sl=10, ctx=ctx,
        )

        trades = result["trades"]
        if not trades:
            pytest.skip("No trades")

        wr_result = sum(1 for t in trades if t["result"] == 1.0) / len(trades)
        wr_pnl = sum(1 for t in trades if t["pnl_raw"] > 0) / len(trades)

        # Both methods should agree
        assert wr_result == wr_pnl, (
            f"WR from result ({wr_result:.2f}) should equal WR from pnl_raw ({wr_pnl:.2f})"
        )
