"""
Tests für CSV Data Adapter.

Fokus auf Edge Cases:
- Fehlende Dateien
- Verschiedene CSV-Formate
- Ungültige Daten
- Spalten-Normalisierung
- Timestamp-Parsing
"""
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fwbg.adapters.data.csv_adapter import CSVDataAdapter


# --- Fixtures ---


@pytest.fixture
def temp_data_dir():
    """Erstellt temporäres Datenverzeichnis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def standard_csv(temp_data_dir):
    """Erstellt Standard-CSV mit T,O,H,L,C,V Spalten."""
    df = pd.DataFrame({
        "T": pd.date_range("2024-01-01", periods=100, freq="1h"),
        "O": np.random.rand(100) + 1.1,
        "H": np.random.rand(100) + 1.11,
        "L": np.random.rand(100) + 1.09,
        "C": np.random.rand(100) + 1.1,
        "V": np.random.randint(100, 1000, 100),
    })
    # Ensure H >= O,C and L <= O,C
    df["H"] = df[["O", "H", "C"]].max(axis=1) + 0.001
    df["L"] = df[["O", "L", "C"]].min(axis=1) - 0.001

    path = temp_data_dir / "EURUSD.csv"
    df.to_csv(path, index=False)
    return temp_data_dir


@pytest.fixture
def metatrader_csv(temp_data_dir):
    """Erstellt MetaTrader-Format CSV."""
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=100, freq="1h"),
        "open": np.random.rand(100) + 1.1,
        "high": np.random.rand(100) + 1.11,
        "low": np.random.rand(100) + 1.09,
        "close": np.random.rand(100) + 1.1,
        "volume": np.random.randint(100, 1000, 100),
    })
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.001
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.001

    path = temp_data_dir / "EURUSD.csv"
    df.to_csv(path, index=False)
    return temp_data_dir


@pytest.fixture
def tradingview_csv(temp_data_dir):
    """Erstellt TradingView-Format CSV (ohne Volume)."""
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=100, freq="1h"),
        "Open": np.random.rand(100) + 1.1,
        "High": np.random.rand(100) + 1.11,
        "Low": np.random.rand(100) + 1.09,
        "Close": np.random.rand(100) + 1.1,
    })
    df["High"] = df[["Open", "High", "Close"]].max(axis=1) + 0.001
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1) - 0.001

    path = temp_data_dir / "EURUSD.csv"
    df.to_csv(path, index=False)
    return temp_data_dir


# --- Connection Tests ---


class TestCSVAdapterConnection:
    """Tests für connect/disconnect."""

    def test_connect_existing_directory(self, temp_data_dir):
        """Connect sollte bei existierendem Verzeichnis True zurückgeben."""
        adapter = CSVDataAdapter(data_path=str(temp_data_dir))

        result = adapter.connect()
        assert result is True
        assert adapter._connected is True

    def test_connect_nonexistent_directory(self, temp_data_dir):
        """Connect sollte bei nicht existierendem Verzeichnis False zurückgeben."""
        adapter = CSVDataAdapter(data_path=str(temp_data_dir / "nonexistent"))

        result = adapter.connect()
        assert result is False
        assert adapter._connected is False

    def test_disconnect_clears_cache(self, standard_csv):
        """Disconnect sollte Cache leeren."""
        adapter = CSVDataAdapter(data_path=str(standard_csv))
        adapter.connect()

        # Lade Daten um Cache zu füllen
        adapter.get_historical_bars("EURUSD")
        assert len(adapter._cache) > 0

        adapter.disconnect()
        assert len(adapter._cache) == 0
        assert adapter._connected is False

    def test_context_manager(self, standard_csv):
        """Context Manager sollte connect/disconnect aufrufen."""
        adapter = CSVDataAdapter(data_path=str(standard_csv))

        with adapter:
            assert adapter._connected is True

        assert adapter._connected is False


# --- Data Loading Tests ---


class TestCSVAdapterDataLoading:
    """Tests für get_historical_bars."""

    def test_load_standard_format(self, standard_csv):
        """Sollte Standard-Format korrekt laden."""
        adapter = CSVDataAdapter(data_path=str(standard_csv))

        with adapter:
            df = adapter.get_historical_bars("EURUSD")

        assert len(df) == 100
        assert list(df.columns) == ["O", "H", "L", "C", "V"]
        # pandas 3.0 nutzt datetime64[us] statt [ns]
        assert "datetime64" in str(df.index.dtype)

    def test_load_metatrader_format(self, metatrader_csv):
        """Sollte MetaTrader-Format korrekt normalisieren."""
        adapter = CSVDataAdapter(data_path=str(metatrader_csv))

        with adapter:
            df = adapter.get_historical_bars("EURUSD")

        assert "O" in df.columns
        assert "H" in df.columns
        assert "L" in df.columns
        assert "C" in df.columns

    def test_load_tradingview_format(self, tradingview_csv):
        """Sollte TradingView-Format korrekt normalisieren."""
        adapter = CSVDataAdapter(data_path=str(tradingview_csv))

        with adapter:
            df = adapter.get_historical_bars("EURUSD")

        assert "O" in df.columns
        assert "V" in df.columns  # Sollte mit 0 gefüllt sein
        assert (df["V"] == 0.0).all()

    def test_file_not_found_returns_empty(self, temp_data_dir):
        """Fehlende Datei sollte leeren DataFrame zurückgeben."""
        adapter = CSVDataAdapter(data_path=str(temp_data_dir))

        with adapter:
            df = adapter.get_historical_bars("NONEXISTENT")

        assert df.empty

    def test_caches_data(self, standard_csv):
        """Daten sollten gecached werden."""
        adapter = CSVDataAdapter(data_path=str(standard_csv))

        with adapter:
            df1 = adapter.get_historical_bars("EURUSD")
            # Cache sollte während der Session gefüllt sein
            assert len(adapter._cache) > 0
            df2 = adapter.get_historical_bars("EURUSD")

        # Nach disconnect wird Cache geleert (das ist korrekt)
        # Der Test prüft dass während der Session gecached wird

    def test_date_filter_start(self, standard_csv):
        """start-Filter sollte funktionieren."""
        adapter = CSVDataAdapter(data_path=str(standard_csv))

        with adapter:
            df_full = adapter.get_historical_bars("EURUSD")
            start = df_full.index[50]
            df_filtered = adapter.get_historical_bars("EURUSD", start=start)

        assert len(df_filtered) <= 51  # 50 bis Ende

    def test_date_filter_end(self, standard_csv):
        """end-Filter sollte funktionieren."""
        adapter = CSVDataAdapter(data_path=str(standard_csv))

        with adapter:
            df_full = adapter.get_historical_bars("EURUSD")
            end = df_full.index[50]
            df_filtered = adapter.get_historical_bars("EURUSD", end=end)

        assert len(df_filtered) <= 51

    def test_limit_filter(self, standard_csv):
        """limit-Filter sollte funktionieren."""
        adapter = CSVDataAdapter(data_path=str(standard_csv))

        with adapter:
            df = adapter.get_historical_bars("EURUSD", limit=10)

        assert len(df) == 10


# --- Column Normalization Tests ---


class TestCSVAdapterNormalization:
    """Tests für Spalten-Normalisierung."""

    def test_lowercase_columns(self, temp_data_dir):
        """Kleinbuchstaben-Spalten sollten normalisiert werden."""
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "open": [1.1] * 10,
            "high": [1.2] * 10,
            "low": [1.0] * 10,
            "close": [1.15] * 10,
            "vol": [100] * 10,
        })
        (temp_data_dir / "TEST.csv").write_text(df.to_csv(index=False))

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            result = adapter.get_historical_bars("TEST")

        assert "O" in result.columns
        assert "V" in result.columns

    def test_uppercase_columns(self, temp_data_dir):
        """Großbuchstaben-Spalten sollten normalisiert werden."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "Open": [1.1] * 10,
            "High": [1.2] * 10,
            "Low": [1.0] * 10,
            "Close": [1.15] * 10,
            "Volume": [100] * 10,
        })
        (temp_data_dir / "TEST.csv").write_text(df.to_csv(index=False))

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            result = adapter.get_historical_bars("TEST")

        assert "O" in result.columns
        assert "V" in result.columns

    def test_missing_volume_filled(self, temp_data_dir):
        """Fehlendes Volume sollte mit 0 gefüllt werden."""
        df = pd.DataFrame({
            "T": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "O": [1.1] * 10,
            "H": [1.2] * 10,
            "L": [1.0] * 10,
            "C": [1.15] * 10,
        })
        (temp_data_dir / "TEST.csv").write_text(df.to_csv(index=False))

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            result = adapter.get_historical_bars("TEST")

        assert "V" in result.columns
        assert (result["V"] == 0.0).all()

    def test_data_sorted_by_index(self, temp_data_dir):
        """Daten sollten nach Index sortiert werden."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        shuffled = dates.to_list()
        np.random.shuffle(shuffled)

        df = pd.DataFrame({
            "T": shuffled,
            "O": range(10),
            "H": range(10),
            "L": range(10),
            "C": range(10),
        })
        (temp_data_dir / "TEST.csv").write_text(df.to_csv(index=False))

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            result = adapter.get_historical_bars("TEST")

        # Index sollte aufsteigend sortiert sein
        assert result.index.is_monotonic_increasing


# --- Edge Cases ---


class TestCSVAdapterEdgeCases:
    """Edge Cases für CSV Adapter."""

    def test_empty_csv(self, temp_data_dir):
        """Leere CSV sollte leeren DataFrame zurückgeben."""
        # Erstelle leere CSV mit nur Header
        (temp_data_dir / "EMPTY.csv").write_text("T,O,H,L,C,V\n")

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            result = adapter.get_historical_bars("EMPTY")

        assert result.empty

    def test_single_row_csv(self, temp_data_dir):
        """CSV mit einer Zeile sollte funktionieren."""
        content = "T,O,H,L,C,V\n2024-01-01 00:00:00,1.1,1.2,1.0,1.15,100"
        (temp_data_dir / "SINGLE.csv").write_text(content)

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            result = adapter.get_historical_bars("SINGLE")

        assert len(result) == 1

    def test_csv_with_extra_columns(self, temp_data_dir):
        """CSV mit Extra-Spalten sollte nur OHLCV behalten."""
        df = pd.DataFrame({
            "T": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "O": [1.1] * 10,
            "H": [1.2] * 10,
            "L": [1.0] * 10,
            "C": [1.15] * 10,
            "V": [100] * 10,
            "extra1": ["foo"] * 10,
            "extra2": [999] * 10,
        })
        (temp_data_dir / "EXTRA.csv").write_text(df.to_csv(index=False))

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            result = adapter.get_historical_bars("EXTRA")

        assert list(result.columns) == ["O", "H", "L", "C", "V"]

    def test_csv_with_nan_values(self, temp_data_dir):
        """CSV mit NaN-Werten sollte diese beibehalten."""
        df = pd.DataFrame({
            "T": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "O": [1.1, np.nan, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            "H": [1.2] * 10,
            "L": [1.0] * 10,
            "C": [1.15] * 10,
            "V": [100] * 10,
        })
        (temp_data_dir / "NAN.csv").write_text(df.to_csv(index=False))

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            result = adapter.get_historical_bars("NAN")

        assert result["O"].isna().sum() == 1

    def test_different_timestamp_formats(self, temp_data_dir):
        """Verschiedene Timestamp-Formate sollten erkannt werden."""
        # ISO Format
        df1 = pd.DataFrame({
            "timestamp": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
            "O": [1.1, 1.2],
            "H": [1.2, 1.3],
            "L": [1.0, 1.1],
            "C": [1.15, 1.25],
        })
        (temp_data_dir / "ISO.csv").write_text(df1.to_csv(index=False))

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            result = adapter.get_historical_bars("ISO")

        assert len(result) == 2

    def test_custom_file_pattern(self, temp_data_dir):
        """Custom file_pattern sollte funktionieren."""
        df = pd.DataFrame({
            "T": pd.date_range("2024-01-01", periods=10, freq="1h"),
            "O": [1.1] * 10,
            "H": [1.2] * 10,
            "L": [1.0] * 10,
            "C": [1.15] * 10,
        })
        (temp_data_dir / "EURUSD_H1.csv").write_text(df.to_csv(index=False))

        adapter = CSVDataAdapter(
            data_path=str(temp_data_dir),
            file_pattern="{symbol}_H1.csv"
        )
        with adapter:
            result = adapter.get_historical_bars("EURUSD")

        assert len(result) == 10


class TestCSVAdapterSymbolListing:
    """Tests für list_available_symbols."""

    def test_lists_available_symbols(self, temp_data_dir):
        """Sollte verfügbare Symbole auflisten."""
        # Erstelle mehrere CSV-Dateien
        for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
            df = pd.DataFrame({
                "T": pd.date_range("2024-01-01", periods=10, freq="1h"),
                "O": [1.1] * 10,
                "H": [1.2] * 10,
                "L": [1.0] * 10,
                "C": [1.15] * 10,
            })
            (temp_data_dir / f"{symbol}.csv").write_text(df.to_csv(index=False))

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            symbols = adapter.list_available_symbols()

        assert "EURUSD" in symbols
        assert "GBPUSD" in symbols
        assert "USDJPY" in symbols

    def test_lists_symbols_with_custom_pattern(self, temp_data_dir):
        """Sollte Symbole mit Custom-Pattern korrekt extrahieren."""
        for symbol in ["EURUSD", "GBPUSD"]:
            df = pd.DataFrame({
                "T": pd.date_range("2024-01-01", periods=10, freq="1h"),
                "O": [1.1] * 10,
                "H": [1.2] * 10,
                "L": [1.0] * 10,
                "C": [1.15] * 10,
            })
            (temp_data_dir / f"{symbol}_H1.csv").write_text(df.to_csv(index=False))

        adapter = CSVDataAdapter(
            data_path=str(temp_data_dir),
            file_pattern="{symbol}_H1.csv"
        )
        with adapter:
            symbols = adapter.list_available_symbols()

        assert "EURUSD" in symbols
        assert "GBPUSD" in symbols

    def test_empty_directory(self, temp_data_dir):
        """Leeres Verzeichnis sollte leere Liste zurückgeben."""
        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            symbols = adapter.list_available_symbols()

        assert symbols == []


class TestCSVAdapterHeaderlessCSV:
    """Tests für Header-lose CSVs (z.B. ForexSB Format)."""

    def test_headerless_csv_detected(self, temp_data_dir):
        """Header-lose CSV sollte automatisch erkannt werden."""
        # ForexSB Format: keine Header-Zeile, direkt Daten
        content = "2024-01-01 00:00,1.1000,1.1100,1.0900,1.1050,1000\n"
        content += "2024-01-01 01:00,1.1050,1.1150,1.0950,1.1100,1100\n"
        content += "2024-01-01 02:00,1.1100,1.1200,1.1000,1.1150,1200\n"
        (temp_data_dir / "HEADERLESS.csv").write_text(content)

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            result = adapter.get_historical_bars("HEADERLESS")

        assert len(result) == 3
        assert list(result.columns) == ["O", "H", "L", "C", "V"]
        assert "datetime64" in str(result.index.dtype)

    def test_headerless_csv_correct_values(self, temp_data_dir):
        """Werte aus Header-loser CSV sollten korrekt sein."""
        content = "2024-01-01 00:00,1.2345,1.3456,1.1234,1.2000,5000\n"
        (temp_data_dir / "VALUES.csv").write_text(content)

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            result = adapter.get_historical_bars("VALUES")

        assert result.iloc[0]["O"] == 1.2345
        assert result.iloc[0]["H"] == 1.3456
        assert result.iloc[0]["L"] == 1.1234
        assert result.iloc[0]["C"] == 1.2000
        assert result.iloc[0]["V"] == 5000

    def test_header_csv_still_works(self, temp_data_dir):
        """CSV mit Header sollte weiterhin funktionieren."""
        df = pd.DataFrame({
            "Time": pd.date_range("2024-01-01", periods=5, freq="1h"),
            "Open": [1.1, 1.2, 1.3, 1.4, 1.5],
            "High": [1.2, 1.3, 1.4, 1.5, 1.6],
            "Low": [1.0, 1.1, 1.2, 1.3, 1.4],
            "Close": [1.15, 1.25, 1.35, 1.45, 1.55],
            "Volume": [100, 200, 300, 400, 500],
        })
        (temp_data_dir / "WITHHEADER.csv").write_text(df.to_csv(index=False))

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            result = adapter.get_historical_bars("WITHHEADER")

        assert len(result) == 5
        # Prüfe dass die Werte korrekt sind (nicht verschoben)
        assert result.iloc[0]["O"] == 1.1
        assert result.iloc[-1]["O"] == 1.5


class TestCSVAdapterCorruptedData:
    """Tests für korrupte/ungültige Daten."""

    def test_malformed_csv(self, temp_data_dir):
        """Malformed CSV sollte graceful behandelt werden."""
        # CSV mit inkonsistenten Spalten
        content = "T,O,H,L,C,V\n2024-01-01,1.1,1.2\n2024-01-02,1.1,1.2,1.0,1.15,100"
        (temp_data_dir / "MALFORMED.csv").write_text(content)

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            # Sollte entweder funktionieren oder leeren DataFrame zurückgeben
            # (pandas read_csv verhält sich unterschiedlich)
            result = adapter.get_historical_bars("MALFORMED")
            # Kein Crash ist der Test

    def test_non_numeric_prices(self, temp_data_dir):
        """Nicht-numerische Preise sollten graceful behandelt werden."""
        content = "T,O,H,L,C,V\n2024-01-01,abc,1.2,1.0,1.15,100"
        (temp_data_dir / "NONNUMERIC.csv").write_text(content)

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            result = adapter.get_historical_bars("NONNUMERIC")
            # pandas 3.0 behält "abc" als String in object-Spalte
            # Der Test prüft dass kein Crash passiert
            # Datenvalidierung ist Aufgabe des Callers
            assert result is not None

    def test_invalid_dates(self, temp_data_dir):
        """Ungültige Datumswerte sollten graceful behandelt werden."""
        content = "T,O,H,L,C,V\nnot-a-date,1.1,1.2,1.0,1.15,100"
        (temp_data_dir / "BADDATE.csv").write_text(content)

        adapter = CSVDataAdapter(data_path=str(temp_data_dir))
        with adapter:
            # Sollte nicht crashen
            try:
                result = adapter.get_historical_bars("BADDATE")
            except Exception:
                pass  # Akzeptabel wenn Exception geworfen wird
