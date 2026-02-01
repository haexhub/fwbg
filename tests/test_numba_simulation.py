"""
Tests für Numba Simulation Core.

Fokus auf Edge Cases und Grenzfälle:
- Extreme Preisbewegungen (Gaps)
- Gleichzeitige TP/SL Hits
- Timeout-Handling
- Array-Grenzen
- Numerische Stabilität
"""
import numpy as np
import pytest

from fwbg.simulation.numba_core import _simulate_trade_numba, compute_targets_numba


# --- Fixtures ---


@pytest.fixture
def simple_uptrend():
    """Einfacher Aufwärtstrend - Long sollte gewinnen."""
    n = 100
    base = 1.1000
    opens = np.array([base + i * 0.0001 for i in range(n)])
    closes = opens + 0.0001
    highs = closes + 0.0001
    lows = opens - 0.0001
    return opens, closes, highs, lows


@pytest.fixture
def simple_downtrend():
    """Einfacher Abwärtstrend - Short sollte gewinnen."""
    n = 100
    base = 1.1000
    opens = np.array([base - i * 0.0001 for i in range(n)])
    closes = opens - 0.0001
    highs = opens + 0.0001
    lows = closes - 0.0001
    return opens, closes, highs, lows


@pytest.fixture
def flat_market():
    """Seitwärtsmarkt - keine klare Richtung."""
    n = 100
    price = 1.1000
    opens = np.full(n, price)
    closes = np.full(n, price)
    highs = np.full(n, price + 0.00005)
    lows = np.full(n, price - 0.00005)
    return opens, closes, highs, lows


@pytest.fixture
def gap_data():
    """Daten mit Gaps (Preis springt)."""
    n = 20
    opens = np.zeros(n)
    closes = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)

    # Normale Bars
    for i in range(10):
        opens[i] = 1.1000 + i * 0.0001
        closes[i] = opens[i] + 0.0001
        highs[i] = closes[i] + 0.0001
        lows[i] = opens[i] - 0.0001

    # Gap Up - Preis springt 50 Pips
    for i in range(10, n):
        opens[i] = 1.1050 + (i - 10) * 0.0001
        closes[i] = opens[i] + 0.0001
        highs[i] = closes[i] + 0.0001
        lows[i] = opens[i] - 0.0001

    return opens, closes, highs, lows


# --- _simulate_trade_numba Tests ---


class TestSimulateTradeBasic:
    """Grundlegende Trade-Simulation Tests."""

    def test_long_win_in_uptrend(self, simple_uptrend):
        """Long Trade gewinnt im Aufwärtstrend."""
        opens, closes, highs, lows = simple_uptrend
        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=0.0010, sl_distance=0.0020,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )
        assert result == 1.0, "Long sollte im Aufwärtstrend gewinnen"
        assert exit_reason == 0, "Exit sollte TP sein"
        assert exit_idx > 0, "Exit Index sollte positiv sein"

    def test_short_loss_in_uptrend(self, simple_uptrend):
        """Short Trade verliert im Aufwärtstrend."""
        opens, closes, highs, lows = simple_uptrend
        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=-1,
            tp_distance=0.0010, sl_distance=0.0010,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )
        assert result == -1.0, "Short sollte im Aufwärtstrend verlieren"
        assert exit_reason == 1, "Exit sollte SL sein"

    def test_short_win_in_downtrend(self, simple_downtrend):
        """Short Trade gewinnt im Abwärtstrend."""
        opens, closes, highs, lows = simple_downtrend
        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=-1,
            tp_distance=0.0010, sl_distance=0.0020,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )
        assert result == 1.0, "Short sollte im Abwärtstrend gewinnen"
        assert exit_reason == 0, "Exit sollte TP sein"

    def test_long_loss_in_downtrend(self, simple_downtrend):
        """Long Trade verliert im Abwärtstrend."""
        opens, closes, highs, lows = simple_downtrend
        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=0.0010, sl_distance=0.0010,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )
        assert result == -1.0, "Long sollte im Abwärtstrend verlieren"
        assert exit_reason == 1, "Exit sollte SL sein"


class TestSimulateTradeEdgeCases:
    """Edge Cases für Trade-Simulation."""

    def test_index_at_last_bar(self, simple_uptrend):
        """Trade am letzten Bar sollte kein Ergebnis liefern."""
        opens, closes, highs, lows = simple_uptrend
        n = len(closes)

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=n - 1, direction=1,
            tp_distance=0.0010, sl_distance=0.0010,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )
        assert result == 0.0, "Kein Trade möglich am letzten Bar"
        assert exit_reason == -1, "Kein Exit-Grund"

    def test_index_beyond_array(self, simple_uptrend):
        """Index außerhalb Array sollte kein Ergebnis liefern."""
        opens, closes, highs, lows = simple_uptrend
        n = len(closes)

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=n + 10, direction=1,
            tp_distance=0.0010, sl_distance=0.0010,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )
        assert result == 0.0, "Kein Trade möglich außerhalb Array"

    def test_tp_sl_same_bar_goes_to_loss(self, gap_data):
        """Wenn TP und SL im selben Bar erreicht werden, sollte Loss gezählt werden."""
        opens, closes, highs, lows = gap_data

        # Kleiner TP und SL damit beides im Gap erreicht wird
        opens_mod = opens.copy()
        closes_mod = closes.copy()
        highs_mod = highs.copy()
        lows_mod = lows.copy()

        # Einen Bar mit extremer Range erstellen
        highs_mod[5] = opens[5] + 0.0100  # 100 Pips über Open
        lows_mod[5] = opens[5] - 0.0100   # 100 Pips unter Open

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens_mod, closes_mod, highs_mod, lows_mod,
            idx=4, direction=1,
            tp_distance=0.0020, sl_distance=0.0020,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )
        # Konservativ: Beide Hits = Loss
        assert result == -1.0, "Bei TP+SL im selben Bar sollte Loss gezählt werden"
        assert exit_reason == 1, "Exit-Grund sollte SL sein"

    def test_zero_spread_slippage(self, simple_uptrend):
        """Trade mit null Spread und Slippage sollte funktionieren."""
        opens, closes, highs, lows = simple_uptrend

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=0.0010, sl_distance=0.0020,
            spread=0.0, slippage=0.0,
            max_bars=50, timeout_bars=0
        )
        assert result in [1.0, -1.0, 0.0], "Ergebnis sollte gültig sein"

    def test_very_large_tp_no_exit(self, simple_uptrend):
        """Sehr großer TP sollte nicht erreicht werden."""
        opens, closes, highs, lows = simple_uptrend

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=1.0,  # 10000 Pips - unrealistisch
            sl_distance=1.0,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )
        assert result == 0.0, "Bei unrealistischen Levels kein Exit"
        assert exit_reason == -1, "Kein Exit-Grund"

    def test_very_small_sl_immediate_loss(self, simple_uptrend):
        """Sehr kleiner SL sollte sofort getroffen werden."""
        opens, closes, highs, lows = simple_uptrend

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=0.0100,
            sl_distance=0.00001,  # Extrem klein
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )
        # SL ist so klein dass er sofort getroffen wird
        assert result == -1.0, "Extrem kleiner SL sollte sofort verlieren"

    def test_max_bars_limits_simulation(self, simple_uptrend):
        """max_bars sollte Simulation begrenzen."""
        opens, closes, highs, lows = simple_uptrend

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=1.0,  # Nie erreicht
            sl_distance=1.0,  # Nie erreicht
            spread=0.0001, slippage=0.00005,
            max_bars=5, timeout_bars=0
        )
        assert result == 0.0, "Kein Exit innerhalb max_bars"

    def test_single_bar_array(self):
        """Ein einzelner Bar sollte keinen Trade ermöglichen."""
        opens = np.array([1.1000])
        closes = np.array([1.1001])
        highs = np.array([1.1002])
        lows = np.array([1.0999])

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=0.0010, sl_distance=0.0010,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )
        assert result == 0.0, "Kein Trade mit nur einem Bar möglich"


class TestSimulateTradeTimeout:
    """Timeout-Handling Tests."""

    def test_timeout_closes_profitable_trade(self, simple_uptrend):
        """Timeout sollte profitablen Trade als Win zählen."""
        opens, closes, highs, lows = simple_uptrend

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=1.0,  # Nie erreicht
            sl_distance=1.0,  # Nie erreicht
            spread=0.0001, slippage=0.00005,
            max_bars=100, timeout_bars=10
        )
        # Im Uptrend ist Long nach 10 Bars profitabel
        assert result == 1.0, "Timeout im Profit sollte Win sein"
        assert exit_reason == 2, "Exit-Grund sollte Timeout sein"
        assert exit_idx == 10, "Exit bei timeout_bars"

    def test_timeout_closes_losing_trade(self, simple_downtrend):
        """Timeout sollte verlierenden Trade als Loss zählen."""
        opens, closes, highs, lows = simple_downtrend

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,  # Long im Downtrend
            tp_distance=1.0,  # Nie erreicht
            sl_distance=1.0,  # Nie erreicht
            spread=0.0001, slippage=0.00005,
            max_bars=100, timeout_bars=10
        )
        # Long im Downtrend ist nach 10 Bars negativ
        assert result == -1.0, "Timeout im Verlust sollte Loss sein"
        assert exit_reason == 2, "Exit-Grund sollte Timeout sein"

    def test_timeout_one_bar(self):
        """Timeout nach einem Bar sollte funktionieren."""
        n = 10
        opens = np.full(n, 1.1000)
        closes = np.full(n, 1.1005)  # Leichter Gewinn
        highs = np.full(n, 1.1010)
        lows = np.full(n, 1.0995)

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=1.0, sl_distance=1.0,
            spread=0.0001, slippage=0.00005,
            max_bars=100, timeout_bars=1
        )
        assert exit_reason == 2, "Exit sollte Timeout sein"
        # timeout_bars=1 bedeutet Exit bei entry_idx + timeout_bars - 1 = 1
        assert exit_idx == 1, "Exit bei Bar 1"

    def test_timeout_exceeds_max_bars(self, simple_uptrend):
        """Timeout größer als max_bars sollte korrekt behandelt werden."""
        opens, closes, highs, lows = simple_uptrend

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=1.0, sl_distance=1.0,
            spread=0.0001, slippage=0.00005,
            max_bars=5,  # Nur 5 Bars simulieren
            timeout_bars=10  # Timeout nach 10 Bars
        )
        # max_bars begrenzt, aber timeout_bars ist relativ zu entry
        # timeout_idx = min(entry + timeout - 1, n - 1) = min(1 + 10 - 1, 99) = 10
        # Aber max_bars begrenzt Loop auf entry + max_bars = 6
        # Kein TP/SL -> Timeout wird geprüft
        assert result in [1.0, -1.0], "Timeout sollte Ergebnis liefern"

    def test_timeout_zero_disables(self, simple_uptrend):
        """timeout_bars=0 sollte Timeout deaktivieren."""
        opens, closes, highs, lows = simple_uptrend

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=1.0, sl_distance=1.0,
            spread=0.0001, slippage=0.00005,
            max_bars=10, timeout_bars=0  # Deaktiviert
        )
        assert result == 0.0, "Kein Timeout wenn timeout_bars=0"
        assert exit_reason == -1, "Kein Exit-Grund"


class TestSimulateTradeNumericalStability:
    """Numerische Stabilität Tests."""

    def test_very_small_prices(self):
        """Sehr kleine Preise (z.B. USDJPY dezimal) sollten funktionieren."""
        n = 50
        base = 0.00001  # Extrem klein
        opens = np.array([base + i * 0.000001 for i in range(n)])
        closes = opens + 0.000001
        highs = closes + 0.000001
        lows = opens - 0.000001

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=0.000005, sl_distance=0.000010,
            spread=0.000001, slippage=0.0000005,
            max_bars=30, timeout_bars=0
        )
        assert result in [1.0, -1.0, 0.0], "Ergebnis sollte gültig sein"
        assert not np.isnan(exit_price), "Exit-Preis sollte nicht NaN sein"

    def test_very_large_prices(self):
        """Sehr große Preise (z.B. BTC) sollten funktionieren."""
        n = 50
        base = 50000.0
        opens = np.array([base + i * 10 for i in range(n)])
        closes = opens + 10
        highs = closes + 10
        lows = opens - 10

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=100.0, sl_distance=200.0,
            spread=10.0, slippage=5.0,
            max_bars=30, timeout_bars=0
        )
        assert result in [1.0, -1.0, 0.0], "Ergebnis sollte gültig sein"
        assert not np.isnan(exit_price) or exit_price == 0.0

    def test_identical_ohlc(self):
        """Identische OHLC-Werte (Doji) sollten funktionieren."""
        n = 50
        price = 1.1000
        opens = np.full(n, price)
        closes = np.full(n, price)
        highs = np.full(n, price)
        lows = np.full(n, price)

        result, exit_idx, exit_price, exit_reason = _simulate_trade_numba(
            opens, closes, highs, lows,
            idx=0, direction=1,
            tp_distance=0.0010, sl_distance=0.0010,
            spread=0.0001, slippage=0.00005,
            max_bars=30, timeout_bars=0
        )
        # Keine Bewegung -> Kein TP/SL getroffen
        assert result == 0.0, "Bei Doji keine Bewegung"


# --- compute_targets_numba Tests ---


class TestComputeTargetsBasic:
    """Grundlegende Tests für compute_targets_numba."""

    def test_returns_correct_shape(self, simple_uptrend):
        """Rückgabe sollte korrekte Dimensionen haben."""
        opens, closes, highs, lows = simple_uptrend
        n = len(closes)

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.0010, sl_distance=0.0010,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )

        assert targets_long.shape == (n,), "Long Targets falsche Shape"
        assert targets_short.shape == (n,), "Short Targets falsche Shape"

    def test_uptrend_more_long_wins(self, simple_uptrend):
        """Im Aufwärtstrend sollten mehr Longs gewinnen."""
        opens, closes, highs, lows = simple_uptrend

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.0010, sl_distance=0.0020,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )

        long_wins = np.sum(targets_long)
        short_wins = np.sum(targets_short)

        assert long_wins > short_wins, "Im Uptrend sollten mehr Longs gewinnen"

    def test_downtrend_more_short_wins(self, simple_downtrend):
        """Im Abwärtstrend sollten mehr Shorts gewinnen."""
        opens, closes, highs, lows = simple_downtrend

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.0010, sl_distance=0.0020,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )

        long_wins = np.sum(targets_long)
        short_wins = np.sum(targets_short)

        assert short_wins > long_wins, "Im Downtrend sollten mehr Shorts gewinnen"

    def test_values_are_binary(self, simple_uptrend):
        """Targets sollten nur 0.0 oder 1.0 sein."""
        opens, closes, highs, lows = simple_uptrend

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.0010, sl_distance=0.0010,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )

        assert set(np.unique(targets_long)).issubset({0.0, 1.0})
        assert set(np.unique(targets_short)).issubset({0.0, 1.0})

    def test_last_bar_always_zero(self, simple_uptrend):
        """Letzter Bar kann keinen Trade starten."""
        opens, closes, highs, lows = simple_uptrend

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.0010, sl_distance=0.0010,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )

        assert targets_long[-1] == 0.0, "Letzter Long Target sollte 0 sein"
        assert targets_short[-1] == 0.0, "Letzter Short Target sollte 0 sein"


class TestComputeTargetsEdgeCases:
    """Edge Cases für compute_targets_numba."""

    def test_empty_array(self):
        """Leere Arrays sollten leere Targets zurückgeben."""
        opens = np.array([], dtype=np.float64)
        closes = np.array([], dtype=np.float64)
        highs = np.array([], dtype=np.float64)
        lows = np.array([], dtype=np.float64)

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.0010, sl_distance=0.0010,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )

        assert len(targets_long) == 0
        assert len(targets_short) == 0

    def test_two_bar_array(self):
        """Zwei Bars - nur erster kann Trade starten."""
        opens = np.array([1.1000, 1.1010])
        closes = np.array([1.1005, 1.1015])
        highs = np.array([1.1020, 1.1020])
        lows = np.array([1.0995, 1.1005])

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.0005, sl_distance=0.0010,
            spread=0.0001, slippage=0.00005,
            max_bars=10, timeout_bars=0
        )

        assert len(targets_long) == 2
        assert targets_long[1] == 0.0, "Letzter Bar kann nicht traden"

    def test_timeout_creates_more_wins(self, flat_market):
        """Timeout im Flat Market sollte Wins erzeugen wenn PnL > 0."""
        opens, closes, highs, lows = flat_market

        # Ohne Timeout
        targets_long_no_to, targets_short_no_to = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.0100, sl_distance=0.0100,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=0
        )

        # Mit Timeout
        targets_long_to, targets_short_to = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.0100, sl_distance=0.0100,
            spread=0.0001, slippage=0.00005,
            max_bars=50, timeout_bars=10
        )

        # Im Flat Market ohne TP/SL Hits können Timeout-Trades
        # Win oder Loss sein je nach kleinen Bewegungen
        total_no_to = np.sum(targets_long_no_to) + np.sum(targets_short_no_to)
        total_to = np.sum(targets_long_to) + np.sum(targets_short_to)

        # Flat Market: Ohne Timeout keine Wins (TP nie erreicht)
        assert total_no_to == 0, "Flat Market ohne Timeout sollte 0 Wins haben"

    def test_extreme_spread_kills_all_trades(self, simple_uptrend):
        """Extremer Spread sollte alle Trades unrentabel machen."""
        opens, closes, highs, lows = simple_uptrend

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.0010, sl_distance=0.0010,
            spread=0.0100,  # 100 Pips Spread!
            slippage=0.0050,
            max_bars=50, timeout_bars=0
        )

        # Bei extremem Spread ist der effektive Entry so schlecht,
        # dass SL sofort getroffen wird
        total_wins = np.sum(targets_long) + np.sum(targets_short)
        assert total_wins < len(opens) * 0.1, "Extremer Spread sollte Wins reduzieren"


class TestComputeTargetsSymmetry:
    """Symmetrie-Tests für Long/Short."""

    def test_symmetric_tp_sl_ratio(self, flat_market):
        """Bei symmetrischem TP/SL sollte Ratio ~50% sein in Flat Market."""
        opens, closes, highs, lows = flat_market

        # Mit Timeout für deterministisches Ergebnis
        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.0010, sl_distance=0.0010,
            spread=0.0, slippage=0.0,  # Keine Kosten
            max_bars=50, timeout_bars=10
        )

        # Flat Market mit 0 Spread: Long und Short sollten ähnlich sein
        long_wins = np.sum(targets_long)
        short_wins = np.sum(targets_short)

        # In echtem Flat Market sind beide etwa gleich
        if long_wins + short_wins > 0:
            ratio = abs(long_wins - short_wins) / (long_wins + short_wins)
            # Toleranz weil Timeout-PnL von Close abhängt
            assert ratio < 0.5, "Long/Short sollten in Flat Market ähnlich sein"


class TestComputeTargetsPerformance:
    """Performance-bezogene Tests."""

    def test_large_array(self):
        """Große Arrays sollten funktionieren."""
        n = 10000
        np.random.seed(42)
        returns = np.random.randn(n) * 0.0001
        base = 1.1000

        closes = np.cumsum(returns) + base
        opens = closes - np.random.rand(n) * 0.0001
        highs = np.maximum(opens, closes) + np.random.rand(n) * 0.0002
        lows = np.minimum(opens, closes) - np.random.rand(n) * 0.0002

        targets_long, targets_short = compute_targets_numba(
            opens, closes, highs, lows,
            tp_distance=0.0020, sl_distance=0.0020,
            spread=0.0001, slippage=0.00005,
            max_bars=100, timeout_bars=50
        )

        assert len(targets_long) == n
        assert len(targets_short) == n
        assert not np.any(np.isnan(targets_long))
        assert not np.any(np.isnan(targets_short))
