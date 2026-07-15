"""Tests für den Wrong-Side-Guard bei absoluten SL-Levels.

Hintergrund (Run 20260715_042300_0b19fe): ein sl_level_abs auf der falschen
Seite des Entry (z. B. Range-HÖHE ~0.005 statt Preis-LEVEL) führte bei Shorts
zu Instant-Exits am Phantompreis, die als riesige Wins gebucht wurden, und
bei Longs zu faktisch deaktiviertem Stop-Loss.
"""
import numpy as np
import pytest

from fwbg.simulation.trade import simulate_pro_trade


def _mk_arrays(prices):
    """(open, close, high, low) Arrays aus (o, h, l, c)-Tupeln."""
    o = np.array([p[0] for p in prices])
    h = np.array([p[1] for p in prices])
    l = np.array([p[2] for p in prices])
    c = np.array([p[3] for p in prices])
    return o, h, l, c


class TestWrongSideSlLevelGuard:
    SPREAD = 0.0003

    def test_short_with_distance_as_level_does_not_book_phantom_win(self):
        # Kurs ~1.26, sl_level_abs=0.005 (Distanz statt Preis, unter Entry).
        prices = [
            (1.2600, 1.2605, 1.2595, 1.2600),  # signal bar
            (1.2600, 1.2608, 1.2592, 1.2598),  # entry bar
            (1.2598, 1.2604, 1.2590, 1.2592),
            (1.2592, 1.2596, 1.2550, 1.2555),  # TP-Bar (Short-TP wird erreicht)
        ]
        o, h, l, c = _mk_arrays(prices)
        trade = simulate_pro_trade(
            c, h, l, idx=0, direction=-1,
            tp_distance=0.0030, sl_distance=0.0024, spread=self.SPREAD,
            opens=o, sl_level_abs=0.005,
        )
        assert trade is not None
        # Kein Instant-Exit am Phantompreis:
        assert trade["exit_price"] > 1.0
        assert trade["bars_held"] >= 1
        # SL wurde auf entry + sl_distance zurückgesetzt
        entry = trade["entry_price"]
        assert trade["sl_level"] == pytest.approx(entry + 0.0024)
        # Gewinn ist die TP-Distanz, nicht der volle Entry-Preis
        assert trade["result"] == 1.0
        assert trade["pnl_raw"] == pytest.approx(0.0030)

    def test_long_with_level_above_entry_keeps_stop_loss(self):
        # sl_level_abs über dem Entry würde den Long-SL sofort/nie triggern —
        # Guard fällt auf entry - sl_distance zurück, SL greift beim Absturz.
        prices = [
            (1.2600, 1.2605, 1.2595, 1.2600),  # signal bar
            (1.2600, 1.2606, 1.2596, 1.2602),  # entry bar
            (1.2602, 1.2604, 1.2540, 1.2545),  # Absturz → SL muss greifen
        ]
        o, h, l, c = _mk_arrays(prices)
        trade = simulate_pro_trade(
            c, h, l, idx=0, direction=1,
            tp_distance=0.0030, sl_distance=0.0024, spread=self.SPREAD,
            opens=o, sl_level_abs=5.0,
        )
        assert trade is not None
        entry = trade["entry_price"]
        assert trade["sl_level"] == pytest.approx(entry - 0.0024)
        assert trade["result"] == -1.0
        assert trade["pnl_raw"] == pytest.approx(-0.0024)

    def test_valid_level_on_correct_side_still_used(self):
        # Gültiges strukturelles Level (Short: über Entry) bleibt in Kraft.
        prices = [
            (1.2600, 1.2605, 1.2595, 1.2600),
            (1.2600, 1.2608, 1.2592, 1.2598),
            (1.2598, 1.2680, 1.2596, 1.2670),  # Spike über das Level
        ]
        o, h, l, c = _mk_arrays(prices)
        trade = simulate_pro_trade(
            c, h, l, idx=0, direction=-1,
            tp_distance=0.0030, sl_distance=0.0024, spread=self.SPREAD,
            opens=o, sl_level_abs=1.2650,
        )
        assert trade is not None
        assert trade["sl_level"] == pytest.approx(1.2650)
        assert trade["result"] == -1.0
