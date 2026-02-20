"""
Tests für den Data Loader - insbesondere CSV-Format-Erkennung.
"""
import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from fwbg.data.loader import load_macro_csv, load_data_aligned


class TestLoadMacroCsv:
    """Tests für die flexible Makro-CSV-Ladefunktion."""

    def test_datetime_column(self, tmp_path):
        """Testet Laden von CSVs mit 'Datetime' Spalte (yfinance Format)."""
        csv_path = tmp_path / "test_datetime.csv"
        csv_path.write_text("Datetime,Close\n2024-01-01,100.0\n2024-01-02,101.5\n")

        df = load_macro_csv(str(csv_path))

        assert df is not None
        assert len(df) == 2
        assert "Close" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_date_column_uppercase(self, tmp_path):
        """Testet Laden von CSVs mit 'DATE' Spalte (Legacy Format)."""
        csv_path = tmp_path / "test_date.csv"
        csv_path.write_text("DATE,Close\n2024-01-01,100.0\n2024-01-02,101.5\n")

        df = load_macro_csv(str(csv_path))

        assert df is not None
        assert len(df) == 2
        assert "Close" in df.columns

    def test_date_column_titlecase(self, tmp_path):
        """Testet Laden von CSVs mit 'Date' Spalte."""
        csv_path = tmp_path / "test_date_title.csv"
        csv_path.write_text("Date,Close\n2024-01-01,100.0\n2024-01-02,101.5\n")

        df = load_macro_csv(str(csv_path))

        assert df is not None
        assert len(df) == 2

    def test_time_column(self, tmp_path):
        """Testet Laden von CSVs mit 'Time' Spalte."""
        csv_path = tmp_path / "test_time.csv"
        csv_path.write_text("Time,Close\n2024-01-01 10:00,100.0\n2024-01-01 11:00,101.5\n")

        df = load_macro_csv(str(csv_path))

        assert df is not None
        assert len(df) == 2

    def test_missing_file(self):
        """Testet Verhalten bei fehlender Datei."""
        df = load_macro_csv("/nonexistent/path/file.csv")
        assert df is None

    def test_no_date_column(self, tmp_path):
        """Testet Verhalten bei CSV ohne erkennbare Datumsspalte."""
        csv_path = tmp_path / "test_nodate.csv"
        csv_path.write_text("Value,Close\n1,100.0\n2,101.5\n")

        df = load_macro_csv(str(csv_path))
        assert df is None

    def test_empty_file(self, tmp_path):
        """Testet Verhalten bei leerer Datei."""
        csv_path = tmp_path / "test_empty.csv"
        csv_path.write_text("")

        df = load_macro_csv(str(csv_path))
        assert df is None

    def test_vix_real_format(self, tmp_path):
        """Testet das echte VIX-Format von yfinance."""
        csv_path = tmp_path / "VIX_DAY.csv"
        csv_path.write_text(
            "Datetime,Close\n"
            "1990-01-02,17.239999771118164\n"
            "1990-01-03,18.190000534057617\n"
            "1990-01-04,19.220000267028809\n"
        )

        df = load_macro_csv(str(csv_path))

        assert df is not None
        assert len(df) == 3
        assert df["Close"].iloc[0] == pytest.approx(17.24, rel=0.01)


class TestLoadDataAligned:
    """Tests für die OHLC-Datenladefunktion."""

    def test_standard_ohlcv(self, tmp_path):
        """Testet Standard OHLCV Format mit Header und Volume."""
        csv_path = tmp_path / "test_ohlc.csv"
        csv_path.write_text(
            "Time,Open,High,Low,Close,Volume\n"
            "2024-01-01 10:00,100.0,101.0,99.0,100.5,1000\n"
            "2024-01-01 11:00,100.5,102.0,100.0,101.5,1100\n"
        )

        df = load_data_aligned(str(csv_path))

        assert df is not None
        assert len(df) == 2
        assert "O" in df.columns
        assert "H" in df.columns
        assert "L" in df.columns
        assert "C" in df.columns
        assert "V" in df.columns
        assert df["V"].iloc[0] == 1000

    def test_close_only(self, tmp_path):
        """Testet Close-only Format (wie VIX)."""
        csv_path = tmp_path / "test_close.csv"
        csv_path.write_text(
            "Time,Close\n"
            "2024-01-01 10:00,100.0\n"
            "2024-01-01 11:00,101.5\n"
        )

        df = load_data_aligned(str(csv_path))

        assert df is not None
        assert len(df) == 2
        # Bei Close-only werden O=H=L=C
        assert df["O"].equals(df["C"])
        assert df["H"].equals(df["C"])
        assert df["L"].equals(df["C"])


class TestRealDataFormats:
    """Integration Tests mit echten Datenformaten."""

    def test_forexsb_raw_format_no_header(self, tmp_path):
        """Testet das echte ForexSB-Format OHNE Header."""
        csv_path = tmp_path / "EURUSD_HOUR.csv"
        # Das echte Format hat keinen Header!
        csv_path.write_text(
            "2024-01-01 10:00,1.1000,1.1010,1.0990,1.1005,100\n"
            "2024-01-01 11:00,1.1005,1.1015,1.0995,1.1010,110\n"
        )

        df = load_data_aligned(str(csv_path))

        assert df is not None
        assert len(df) == 2
        assert "O" in df.columns
        assert "H" in df.columns
        assert "L" in df.columns
        assert "C" in df.columns
        assert "V" in df.columns
        assert df["C"].iloc[0] == pytest.approx(1.1005, rel=0.01)
        assert df["V"].iloc[0] == 100

    def test_forexsb_imported_format(self, tmp_path):
        """Testet das von forexsb_importer.py erzeugte Format."""
        # Nach dem Import hat ForexSB-Daten ein sauberes Format
        csv_path = tmp_path / "EURUSD_HOUR.csv"
        csv_path.write_text(
            "Time,Open,High,Low,Close,Volume\n"
            "2024-01-01 10:00:00,1.1000,1.1010,1.0990,1.1005,100\n"
            "2024-01-01 11:00:00,1.1005,1.1015,1.0995,1.1010,110\n"
        )

        df = load_data_aligned(str(csv_path))

        assert df is not None
        assert len(df) == 2
        assert df["C"].iloc[0] == pytest.approx(1.1005, rel=0.01)

    def test_dukascopy_unix_ms_format(self, tmp_path):
        """Testet das Dukascopy-Format: Unix-ms-Timestamps, kein Volume."""
        csv_path = tmp_path / "DAX_MINUTE_15.csv"
        # 1388534400000 = 2014-01-01 00:00:00 UTC
        csv_path.write_text(
            "timestamp,open,high,low,close\n"
            "1388534400000,9552.1,9553.0,9551.0,9552.5\n"
            "1388535300000,9552.5,9554.0,9552.0,9553.0\n"
        )

        df = load_data_aligned(str(csv_path))

        assert df is not None
        assert len(df) == 2
        assert "O" in df.columns and "H" in df.columns
        assert "L" in df.columns and "C" in df.columns
        # Timestamp muss korrekt geparst sein (kein Unix-Integer als Datum)
        assert df.index[0].year == 2014
        assert df["O"].iloc[0] == pytest.approx(9552.1)
        assert df["C"].iloc[0] == pytest.approx(9552.5)
        # Kein Volume → V=0
        assert df["V"].iloc[0] == 0

    def test_yfinance_macro_format(self, tmp_path):
        """Testet yfinance Makro-Daten Format."""
        csv_path = tmp_path / "yfinance_macro.csv"
        csv_path.write_text(
            "Datetime,Close\n"
            "2024-01-01,100.0\n"
            "2024-01-02,101.5\n"
            "2024-01-03,99.8\n"
        )

        df = load_macro_csv(str(csv_path))

        assert df is not None
        assert len(df) == 3
        # Index sollte DatetimeIndex sein
        assert isinstance(df.index, pd.DatetimeIndex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
