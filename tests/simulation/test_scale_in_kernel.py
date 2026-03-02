"""Tests for _simulate_trade_scale_in_numba kernel."""
import numpy as np
import pytest

from fwbg.simulation.numba_core import _simulate_trade_scale_in_numba


def _make_levels(fractions):
    """Pack level fractions into fixed-size array."""
    arr = np.full(10, -1.0, dtype=np.float64)
    for i, f in enumerate(fractions):
        arr[i] = f
    return arr, len(fractions)


def _call(
    opens,
    highs,
    lows,
    closes,
    idx=0,
    direction=1,
    tp_distance=5.0,
    sl_distance=5.0,
    spread=0.0,
    slippage=0.0,
    max_bars=50,
    timeout_bars=0,
    scale_fractions=None,
    scale_qty_mult=1.0,
    breakeven_trigger=0.0,
    trail_distance=0.0,
    trail_tp_dist=0.0,
):
    """Convenience wrapper to call the kernel with sensible defaults."""
    if scale_fractions is None:
        scale_fractions = []
    levels, n = _make_levels(scale_fractions)
    return _simulate_trade_scale_in_numba(
        np.asarray(opens, dtype=np.float64),
        np.asarray(closes, dtype=np.float64),
        np.asarray(highs, dtype=np.float64),
        np.asarray(lows, dtype=np.float64),
        idx,
        direction,
        tp_distance,
        sl_distance,
        spread,
        slippage,
        max_bars,
        timeout_bars,
        levels,
        n,
        scale_qty_mult,
        breakeven_trigger,
        trail_distance,
        trail_tp_dist,
    )


class TestNoScaleInStraightTP:
    """Price goes straight up to TP without retracing."""

    def test_long_straight_tp(self):
        # Entry at bar 1 open = 100. TP = 105, SL = 95.
        # Bar 1: price goes to 106 (TP hit).
        opens  = [100.0, 100.0, 100.0]
        highs  = [100.0, 106.0, 100.0]
        lows   = [100.0,  99.5, 100.0]
        closes = [100.0, 105.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            scale_fractions=[0.3, 0.5],
        )

        assert result == 1.0
        assert exit_idx == 1
        assert exit_price == 105.0  # TP level
        assert exit_reason == 0     # TP
        assert n_fills == 1         # No scale-in
        assert avg_price == 100.0
        assert total_qty == 1.0


class TestOneScaleInThenTP:
    """Price retraces 30% toward SL, fills one level, then recovers to adjusted TP."""

    def test_one_fill_then_tp(self):
        # Entry = 100.0, SL = 95.0, tp_distance = 5.0, sl_distance = 5.0
        # Scale level at 0.3 => trigger = 100 - 0.3*5 = 98.5
        # Bar 1: price drops to 98.0 => fills 98.5 level
        #   avg_price = (100*1 + 98.5*1)/2 = 99.25
        #   new TP = 99.25 + 5 = 104.25
        # Bar 2: price goes to 105 => hits adjusted TP at 104.25
        opens  = [100.0, 100.0, 100.0, 100.0]
        highs  = [100.0, 100.5, 105.0, 100.0]
        lows   = [100.0,  98.0,  99.0, 100.0]
        closes = [100.0,  99.0, 104.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            scale_fractions=[0.3],
        )

        assert result == 1.0
        assert exit_reason == 0
        assert n_fills == 2
        assert total_qty == 2.0
        assert avg_price == pytest.approx(99.25)
        assert exit_price == pytest.approx(104.25)  # avg_price + tp_distance


class TestSLHitNoScaleIn:
    """Price drops straight to SL. No scale-in triggered."""

    def test_sl_hit_immediate(self):
        # Entry = 100, SL = 95, tp_distance = 5, sl_distance = 5.
        # Scale at 0.3 => trigger = 98.5
        # We use a gap down to ensure price doesn't pass through scale trigger
        # Bar 1: opens/highs at 94, low at 94 => SL hit without touching 98.5 range
        # Actually, on this bar lows[1]=94 <= scale_price(98.5) too, so it fills.
        # To truly avoid scale-in: use no levels.
        opens  = [100.0, 100.0, 100.0]
        highs  = [100.0, 100.0, 100.0]
        lows   = [100.0,  94.0, 100.0]
        closes = [100.0,  95.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            scale_fractions=[],  # No scale levels — straight SL hit
        )

        assert result == -1.0
        assert exit_reason == 1  # SL
        assert n_fills == 1
        assert total_qty == 1.0


class TestSLHitAfterScaleIn:
    """Price retraces, fills one level, then drops to SL."""

    def test_sl_after_one_fill(self):
        # Entry = 100, SL = 95, scale at 0.3 => trigger = 98.5
        # Bar 1: drops to 98 (fills 98.5), avg = 99.25, qty = 2
        # Bar 2: drops to 94 (SL hit at 95)
        # PnL = (95 - 99.25) * 2 = -8.5 => loss
        opens  = [100.0, 100.0, 100.0, 100.0]
        highs  = [100.0, 100.0,  96.0, 100.0]
        lows   = [100.0,  98.0,  94.0, 100.0]
        closes = [100.0,  99.0,  95.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            scale_fractions=[0.3],
        )

        assert result == -1.0
        assert exit_reason == 1  # SL
        assert n_fills == 2
        assert total_qty == 2.0
        assert avg_price == pytest.approx(99.25)
        assert exit_price == 95.0


class TestTPAdjustment:
    """After scale-in, TP = avg_price + tp_distance (not original TP)."""

    def test_tp_moves_down_after_scale_in(self):
        # Entry = 100, tp_distance = 5, sl_distance = 10
        # Original TP = 105, SL = 90
        # Scale at 0.5 => trigger = 100 - 0.5*10 = 95
        # Bar 1: drops to 94 => fills at 95. avg = (100+95)/2 = 97.5
        # New TP = 97.5 + 5 = 102.5  (lower than original 105!)
        # Bar 2: hits 103 => TP hit at 102.5
        opens  = [100.0, 100.0, 100.0, 100.0]
        highs  = [100.0, 100.5, 103.0, 100.0]
        lows   = [100.0,  94.0,  97.0, 100.0]
        closes = [100.0,  96.0, 102.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            tp_distance=5.0,
            sl_distance=10.0,
            scale_fractions=[0.5],
        )

        assert result == 1.0
        assert exit_reason == 0
        assert n_fills == 2
        assert avg_price == pytest.approx(97.5)
        assert exit_price == pytest.approx(102.5)


class TestAllLevelsFilled:
    """Price retraces through all 3 levels, then recovers to TP."""

    def test_three_levels_filled(self):
        # Entry = 100, tp_distance = 10, sl_distance = 10
        # SL = 90, TP = 110
        # Levels at 0.2, 0.5, 0.8 => triggers at 98, 95, 92
        # scale_qty_mult = 1.0
        #
        # Bar 1: low=97 => fills 98 level. avg=(100+98)/2=99, qty=2, TP=109
        # Bar 2: low=94 => fills 95 level. avg=(100+98+95)/3=97.667, qty=3, TP=107.667
        # Bar 3: low=91 => fills 92 level. avg=(100+98+95+92)/4=96.25, qty=4, TP=106.25
        # Bar 4: high=107 => TP hit at 106.25
        opens  = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        highs  = [100.0, 100.0,  96.0,  93.0, 107.0, 100.0]
        lows   = [100.0,  97.0,  94.0,  91.0,  96.0, 100.0]
        closes = [100.0,  98.0,  95.0,  92.0, 106.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            tp_distance=10.0,
            sl_distance=10.0,
            scale_fractions=[0.2, 0.5, 0.8],
        )

        assert result == 1.0
        assert exit_reason == 0
        assert n_fills == 4  # initial + 3
        assert total_qty == 4.0
        assert avg_price == pytest.approx(96.25)
        assert exit_price == pytest.approx(106.25)


class TestShortDirection:
    """Mirror of one_scale_in_then_tp for short trades."""

    def test_short_one_fill_then_tp(self):
        # Entry = 100 (short), SL = 105, TP = 95
        # Scale at 0.3 => trigger = 100 + 0.3*5 = 101.5
        # Bar 1: high=102 => fills 101.5 level
        #   avg = (100+101.5)/2 = 100.75
        #   TP = 100.75 - 5 = 95.75
        # Bar 2: low=95 => TP hit at 95.75
        opens  = [100.0, 100.0, 100.0, 100.0]
        highs  = [100.0, 102.0, 100.0, 100.0]
        lows   = [100.0,  99.0,  95.0, 100.0]
        closes = [100.0, 101.0,  96.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            direction=-1,
            scale_fractions=[0.3],
        )

        assert result == 1.0
        assert exit_reason == 0
        assert n_fills == 2
        assert total_qty == 2.0
        assert avg_price == pytest.approx(100.75)
        assert exit_price == pytest.approx(95.75)  # avg - tp_distance


class TestWithTrailingStop:
    """Scale-in + trailing. Verify breakeven uses avg_price."""

    def test_breakeven_uses_avg_price(self):
        # Entry = 100, tp_distance = 10, sl_distance = 10
        # SL = 90, TP = 110
        # Scale at 0.3 => trigger = 100 - 0.3*10 = 97
        # breakeven_trigger = 0.5 => be_trigger recalc after scale-in
        #
        # Bar 1: low=96 => fills 97. avg=(100+97)/2=98.5, qty=2
        #   new TP = 98.5+10 = 108.5
        #   be_trigger = 98.5 + 10*0.5 = 103.5
        # Bar 2: high=104 => breakeven triggered, SL moves to avg_price=98.5
        # Bar 3: low=98 => SL hit at 98.5
        #   PnL = (98.5 - 98.5)*2 = 0 => result = -1.0 (not > 0)
        opens  = [100.0, 100.0, 100.0, 100.0, 100.0]
        highs  = [100.0, 100.0, 104.0, 100.0, 100.0]
        lows   = [100.0,  96.0, 100.0,  98.0, 100.0]
        closes = [100.0,  97.0, 103.0,  99.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            tp_distance=10.0,
            sl_distance=10.0,
            scale_fractions=[0.3],
            breakeven_trigger=0.5,
        )

        assert exit_reason == 1  # SL
        assert n_fills == 2
        assert avg_price == pytest.approx(98.5)
        assert exit_price == pytest.approx(98.5)  # SL moved to avg_price
        # PnL = 0 => result = -1.0
        assert result == -1.0


class TestScaleLevelBelowSLNotFilled:
    """Scale level at 0.95 (95% toward SL) — trigger price is very close to SL."""

    def test_level_near_sl_not_filled(self):
        # Entry = 100, sl_distance = 10, SL = 90
        # Scale at 0.95 => trigger = 100 - 0.95*10 = 90.5
        # Bar 1: low = 90.5 => trigger hit BUT scale_price (90.5) > sl (90), so it fills
        # Actually 90.5 > 90 so it does fill. Let's test at 1.0 exactly:
        # Scale at 1.0 => trigger = 100 - 1.0*10 = 90.0
        # trigger (90) is NOT > sl (90), so it should NOT fill
        opens  = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        highs  = [100.0, 100.0, 100.0, 100.0, 111.0, 100.0]
        lows   = [100.0,  90.0,  91.0,  92.0, 100.0, 100.0]
        closes = [100.0,  91.0,  92.0,  93.0, 110.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            tp_distance=10.0,
            sl_distance=10.0,
            scale_fractions=[1.0],  # trigger exactly at SL
        )

        # The level trigger = SL exactly, so guard (trigger > sl) fails => no fill
        assert n_fills == 1
        assert total_qty == 1.0

    def test_level_just_above_sl_fills(self):
        # Scale at 0.95 => trigger = 100 - 0.95*10 = 90.5
        # 90.5 > 90 (SL) => should fill
        opens  = [100.0, 100.0, 100.0, 100.0]
        highs  = [100.0, 100.0, 111.0, 100.0]
        lows   = [100.0,  90.0,  95.0, 100.0]
        closes = [100.0,  91.0, 110.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            tp_distance=10.0,
            sl_distance=10.0,
            scale_fractions=[0.95],
        )

        assert n_fills == 2  # scale-in filled
        assert total_qty == 2.0


class TestTimeoutWithScaleIn:
    """Timeout uses avg_price for PnL calculation."""

    def test_timeout_pnl_uses_avg_price(self):
        # Entry = 100, tp_distance = 10, sl_distance = 10
        # Scale at 0.3 => trigger = 97
        # timeout_bars = 3
        # Bar 1: low=96 => fills 97. avg=98.5, qty=2
        # Bar 2: timeout bar. close=99
        # PnL = (99 - 98.5)*2 = 1.0 => win
        opens  = [100.0, 100.0, 100.0, 100.0]
        highs  = [100.0, 100.0, 100.0, 100.0]
        lows   = [100.0,  96.0,  98.0, 100.0]
        closes = [100.0,  97.0,  99.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            tp_distance=10.0,
            sl_distance=10.0,
            timeout_bars=3,
            scale_fractions=[0.3],
        )

        assert exit_reason == 2  # Timeout
        assert n_fills == 2
        assert avg_price == pytest.approx(98.5)
        assert result == 1.0  # PnL > 0


class TestScaleQtyMult:
    """Verify scale_qty_mult changes total_qty and avg_price correctly."""

    def test_half_qty_scale_in(self):
        # Entry = 100, scale at 0.3 => trigger = 98.5
        # scale_qty_mult = 0.5
        # Bar 1: fills 98.5. avg = (100*1 + 98.5*0.5) / 1.5 = 149.25/1.5 = 99.5
        # qty = 1.5, TP = 99.5 + 5 = 104.5
        # Bar 2: high = 105 => TP hit at 104.5
        opens  = [100.0, 100.0, 100.0, 100.0]
        highs  = [100.0, 100.0, 105.0, 100.0]
        lows   = [100.0,  98.0,  99.0, 100.0]
        closes = [100.0,  99.0, 104.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            scale_fractions=[0.3],
            scale_qty_mult=0.5,
        )

        assert result == 1.0
        assert n_fills == 2
        assert total_qty == pytest.approx(1.5)
        assert avg_price == pytest.approx(99.5)
        assert exit_price == pytest.approx(104.5)


class TestNoExit:
    """Price stays flat, no TP/SL/timeout hit within max_bars."""

    def test_no_exit_returns_zero(self):
        # 5 bars, price stays at 100, max_bars=3, no timeout
        opens  = [100.0, 100.0, 100.0, 100.0, 100.0]
        highs  = [100.0, 100.5, 100.5, 100.5, 100.0]
        lows   = [100.0,  99.5,  99.5,  99.5, 100.0]
        closes = [100.0, 100.0, 100.0, 100.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            max_bars=3,
            scale_fractions=[0.3],
        )

        assert result == 0.0
        assert exit_idx == -1
        assert exit_reason == -1


class TestShortSLHitAfterScaleIn:
    """Short: scale-in then SL hit."""

    def test_short_sl_after_scale(self):
        # Short entry = 100, SL = 105, TP = 95
        # Scale at 0.4 => trigger = 100 + 0.4*5 = 102
        # Bar 1: high=103 => fills 102. avg=(100+102)/2=101, qty=2
        #   TP = 101 - 5 = 96
        # Bar 2: high=106 => SL hit at 105
        # PnL = (101 - 105)*2 = -8 => loss
        opens  = [100.0, 100.0, 100.0, 100.0]
        highs  = [100.0, 103.0, 106.0, 100.0]
        lows   = [100.0,  99.0, 100.0, 100.0]
        closes = [100.0, 102.0, 105.0, 100.0]

        result, exit_idx, exit_price, exit_reason, avg_price, total_qty, n_fills = _call(
            opens, highs, lows, closes,
            direction=-1,
            scale_fractions=[0.4],
        )

        assert result == -1.0
        assert exit_reason == 1
        assert n_fills == 2
        assert avg_price == pytest.approx(101.0)
        assert exit_price == 105.0
