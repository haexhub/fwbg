"""
Unit-Tests für IG Broker Mappings.

Testet alle Symbol- und Timeframe-Mappings für die IG Markets Integration.
"""
import pytest

from fwbg.core.enums import Symbol, Timeframe
from .mappings import (
    SYMBOL_TO_EPIC,
    SYMBOL_TO_YFINANCE,
    SYMBOL_POINT_VALUE,
    TIMEFRAME_TO_RESOLUTION,
    TIMEFRAME_TO_YF_INTERVAL,
)


class TestSymbolToEpicMapping:
    """Tests für Symbol -> Epic Mapping."""

    def test_all_forex_majors_have_epics(self):
        """Alle Forex Majors sollten Epic-Mappings haben."""
        forex_majors = [
            Symbol.EURUSD, Symbol.GBPUSD, Symbol.USDJPY,
            Symbol.USDCHF, Symbol.USDCAD, Symbol.AUDUSD, Symbol.NZDUSD
        ]
        for symbol in forex_majors:
            assert symbol in SYMBOL_TO_EPIC, f"{symbol} fehlt in SYMBOL_TO_EPIC"
            assert SYMBOL_TO_EPIC[symbol].startswith("CS.D."), f"{symbol} Epic hat falsches Präfix"

    def test_all_indices_have_epics(self):
        """Alle Indizes sollten Epic-Mappings haben."""
        indices = [Symbol.DAX, Symbol.DOW30, Symbol.NAS100, Symbol.SPX500, Symbol.FTSE100]
        for symbol in indices:
            assert symbol in SYMBOL_TO_EPIC, f"{symbol} fehlt in SYMBOL_TO_EPIC"
            assert SYMBOL_TO_EPIC[symbol].startswith("IX.D."), f"{symbol} Epic hat falsches Präfix"

    def test_commodities_have_epics(self):
        """Commodities sollten Epic-Mappings haben."""
        commodities = [Symbol.XAUUSD, Symbol.XAGUSD, Symbol.BRENT, Symbol.WTI]
        for symbol in commodities:
            assert symbol in SYMBOL_TO_EPIC, f"{symbol} fehlt in SYMBOL_TO_EPIC"

    def test_crypto_have_epics(self):
        """Crypto sollten Epic-Mappings haben."""
        crypto = [Symbol.BTCUSD, Symbol.ETHUSD]
        for symbol in crypto:
            assert symbol in SYMBOL_TO_EPIC, f"{symbol} fehlt in SYMBOL_TO_EPIC"

    def test_epic_format_is_valid(self):
        """Alle Epics sollten gültiges Format haben."""
        for symbol, epic in SYMBOL_TO_EPIC.items():
            assert isinstance(epic, str), f"Epic für {symbol} ist kein String"
            assert len(epic) > 5, f"Epic für {symbol} ist zu kurz: {epic}"
            assert "." in epic, f"Epic für {symbol} hat kein Punkt-Trennzeichen"

    def test_no_duplicate_epics(self):
        """Keine doppelten Epic-Werte."""
        epics = list(SYMBOL_TO_EPIC.values())
        assert len(epics) == len(set(epics)), "Doppelte Epic-Werte gefunden"


class TestSymbolToYfinanceMapping:
    """Tests für Symbol -> yfinance Ticker Mapping."""

    def test_forex_tickers_have_x_suffix(self):
        """Forex Ticker sollten =X Suffix haben."""
        forex_symbols = [s for s in SYMBOL_TO_YFINANCE.keys()
                        if s.value in ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]]
        for symbol in forex_symbols:
            ticker = SYMBOL_TO_YFINANCE[symbol]
            assert ticker.endswith("=X"), f"{symbol} Ticker hat kein =X Suffix: {ticker}"

    def test_index_tickers_have_caret_prefix(self):
        """Index Ticker sollten ^ Präfix haben."""
        index_symbols = [Symbol.DAX, Symbol.DOW30, Symbol.NAS100, Symbol.SPX500, Symbol.FTSE100]
        for symbol in index_symbols:
            if symbol in SYMBOL_TO_YFINANCE:
                ticker = SYMBOL_TO_YFINANCE[symbol]
                assert ticker.startswith("^"), f"{symbol} Ticker hat kein ^ Präfix: {ticker}"

    def test_commodity_tickers_have_f_suffix(self):
        """Commodity Ticker sollten =F Suffix haben."""
        commodity_symbols = [Symbol.XAUUSD, Symbol.XAGUSD, Symbol.BRENT, Symbol.WTI]
        for symbol in commodity_symbols:
            if symbol in SYMBOL_TO_YFINANCE:
                ticker = SYMBOL_TO_YFINANCE[symbol]
                assert ticker.endswith("=F"), f"{symbol} Ticker hat kein =F Suffix: {ticker}"

    def test_crypto_ticker_format(self):
        """Crypto Ticker sollten korrektes Format haben."""
        assert SYMBOL_TO_YFINANCE.get(Symbol.BTCUSD) == "BTC-USD"
        assert SYMBOL_TO_YFINANCE.get(Symbol.ETHUSD) == "ETH-USD"


class TestSymbolPointValue:
    """Tests für Point Value Mapping."""

    def test_forex_majors_have_standard_point_values(self):
        """Forex Majors sollten 0.0001 oder 0.01 (JPY) Point Value haben."""
        jpy_pairs = [Symbol.USDJPY, Symbol.EURJPY, Symbol.GBPJPY, Symbol.AUDJPY,
                     Symbol.NZDJPY, Symbol.CADJPY, Symbol.CHFJPY]

        for symbol in SYMBOL_POINT_VALUE:
            if symbol in jpy_pairs:
                assert SYMBOL_POINT_VALUE[symbol] == 0.01, f"{symbol} sollte 0.01 sein"
            elif symbol.value.endswith("USD") or symbol.value.endswith("CHF") or \
                 symbol.value.endswith("CAD") or symbol.value.endswith("GBP") or \
                 symbol.value.endswith("AUD") or symbol.value.endswith("NZD"):
                if symbol not in [Symbol.XAUUSD, Symbol.XAGUSD, Symbol.BTCUSD, Symbol.ETHUSD]:
                    assert SYMBOL_POINT_VALUE[symbol] == 0.0001, f"{symbol} sollte 0.0001 sein"

    def test_all_mapped_symbols_have_point_values(self):
        """Alle Symbols mit Epics sollten Point Values haben."""
        for symbol in SYMBOL_TO_EPIC.keys():
            assert symbol in SYMBOL_POINT_VALUE, f"{symbol} fehlt in SYMBOL_POINT_VALUE"

    def test_point_values_are_positive(self):
        """Alle Point Values sollten positiv sein."""
        for symbol, value in SYMBOL_POINT_VALUE.items():
            assert value > 0, f"Point Value für {symbol} ist nicht positiv: {value}"


class TestTimeframeToResolution:
    """Tests für Timeframe -> IG Resolution Mapping."""

    def test_all_common_timeframes_mapped(self):
        """Alle gängigen Timeframes sollten gemappt sein."""
        expected = [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.D1]
        for tf in expected:
            assert tf in TIMEFRAME_TO_RESOLUTION, f"{tf} fehlt in TIMEFRAME_TO_RESOLUTION"

    def test_resolution_format(self):
        """Resolutions sollten korrektes IG API Format haben."""
        assert TIMEFRAME_TO_RESOLUTION[Timeframe.M1] == "MINUTE"
        assert TIMEFRAME_TO_RESOLUTION[Timeframe.M5] == "MINUTE_5"
        assert TIMEFRAME_TO_RESOLUTION[Timeframe.H1] == "HOUR"
        assert TIMEFRAME_TO_RESOLUTION[Timeframe.D1] == "DAY"


class TestTimeframeToYfinanceInterval:
    """Tests für Timeframe -> yfinance Interval Mapping."""

    def test_minute_timeframes_have_m_suffix(self):
        """Minute Timeframes sollten m Suffix haben."""
        minute_tfs = [Timeframe.M1, Timeframe.M5, Timeframe.M15]
        for tf in minute_tfs:
            if tf in TIMEFRAME_TO_YF_INTERVAL:
                interval = TIMEFRAME_TO_YF_INTERVAL[tf]
                assert interval.endswith("m"), f"{tf} Interval hat kein m Suffix: {interval}"

    def test_hour_timeframes_have_h_suffix(self):
        """Hour Timeframes sollten h Suffix haben."""
        if Timeframe.H1 in TIMEFRAME_TO_YF_INTERVAL:
            assert TIMEFRAME_TO_YF_INTERVAL[Timeframe.H1] == "1h"
        if Timeframe.H4 in TIMEFRAME_TO_YF_INTERVAL:
            assert TIMEFRAME_TO_YF_INTERVAL[Timeframe.H4] == "4h"

    def test_daily_timeframe(self):
        """Daily Timeframe sollte '1d' sein."""
        if Timeframe.D1 in TIMEFRAME_TO_YF_INTERVAL:
            assert TIMEFRAME_TO_YF_INTERVAL[Timeframe.D1] == "1d"


class TestMappingConsistency:
    """Tests für Konsistenz zwischen Mappings."""

    def test_epic_and_yfinance_coverage_similar(self):
        """Epic und yfinance Mappings sollten ähnliche Coverage haben."""
        epic_symbols = set(SYMBOL_TO_EPIC.keys())
        yfinance_symbols = set(SYMBOL_TO_YFINANCE.keys())

        # Alle yfinance Symbols sollten auch Epic haben
        for symbol in yfinance_symbols:
            assert symbol in epic_symbols, f"{symbol} hat yfinance aber kein Epic"

    def test_resolution_and_yf_interval_timeframes_overlap(self):
        """Resolution und yfinance Interval sollten gemeinsame Timeframes haben."""
        resolution_tfs = set(TIMEFRAME_TO_RESOLUTION.keys())
        yf_tfs = set(TIMEFRAME_TO_YF_INTERVAL.keys())

        common = resolution_tfs & yf_tfs
        assert len(common) >= 4, f"Zu wenig gemeinsame Timeframes: {common}"
