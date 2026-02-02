"""
CSV Data Adapter - Lädt Daten aus CSV-Dateien.

Unterstützt verschiedene CSV-Formate:
- Standard: T,O,H,L,C,V
- MetaTrader: datetime,open,high,low,close,volume
- TradingView: time,open,high,low,close

Beispiel:
    adapter = CSVDataAdapter(
        data_path="data/",
        symbols=["EURUSD", "GBPUSD"],
        file_pattern="{symbol}_H1.csv"
    )

    with adapter:
        df = adapter.get_historical_bars("EURUSD")
        for bar in adapter.stream_historical_bars("EURUSD"):
            print(bar)
"""
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import pandas as pd

from . import DataAdapter


class CSVDataAdapter(DataAdapter):
    """
    Lädt Marktdaten aus CSV-Dateien.

    Flexibles Format-Handling:
    - Automatische Spalten-Erkennung
    - Verschiedene Timestamp-Formate
    - Konfigurierbare Datei-Patterns
    """

    adapter_type: str = "csv"

    # Bekannte Spalten-Mappings
    COLUMN_MAPPINGS = {
        "open": "O",
        "high": "H",
        "low": "L",
        "close": "C",
        "volume": "V",
        "vol": "V",
        "Open": "O",
        "High": "H",
        "Low": "L",
        "Close": "C",
        "Volume": "V",
    }

    TIMESTAMP_COLUMNS = ["T", "time", "datetime", "date", "timestamp", "Date", "Time"]

    def __init__(
        self,
        data_path: str = "data",
        symbols: List[str] = None,
        timeframe: str = "1H",
        file_pattern: str = "{symbol}.csv",
        **kwargs
    ):
        """
        Args:
            data_path: Pfad zum Daten-Verzeichnis
            symbols: Liste von Symbolen
            timeframe: Standard-Timeframe
            file_pattern: Pattern für Dateinamen ({symbol} wird ersetzt)
        """
        super().__init__(symbols=symbols, timeframe=timeframe, **kwargs)

        self._data_path = Path(data_path)
        self._file_pattern = file_pattern
        self._cache: Dict[str, pd.DataFrame] = {}

    @property
    def data_path(self) -> Path:
        return self._data_path

    def connect(self) -> bool:
        """Prüft ob Daten-Verzeichnis existiert."""
        if not self._data_path.exists():
            self.log_error(f"Data path does not exist: {self._data_path}")
            return False

        self._connected = True
        self.log_info(f"Connected to {self._data_path}")
        return True

    def disconnect(self):
        """Räumt Cache auf."""
        self._cache.clear()
        self._connected = False

    def _get_file_path(self, symbol: str) -> Path:
        """Erstellt Dateipfad für Symbol."""
        filename = self._file_pattern.format(symbol=symbol)
        return self._data_path / filename

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalisiert Spalten-Namen zu O,H,L,C,V."""
        # Timestamp als Index setzen
        for col in self.TIMESTAMP_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break

        # Falls erste Spalte numerisch ist (Index-Spalte)
        if df.columns[0].isdigit() or df.index.dtype == "int64":
            # Versuche erste Spalte als Timestamp
            try:
                df.index = pd.to_datetime(df.iloc[:, 0])
                df = df.iloc[:, 1:]
            except Exception:
                pass

        # Spalten umbenennen
        rename_map = {}
        for old, new in self.COLUMN_MAPPINGS.items():
            if old in df.columns:
                rename_map[old] = new

        if rename_map:
            df = df.rename(columns=rename_map)

        # Nur OHLCV Spalten behalten
        keep_cols = [c for c in ["O", "H", "L", "C", "V"] if c in df.columns]
        df = df[keep_cols]

        # Fehlende Spalten ergänzen
        if "V" not in df.columns:
            df["V"] = 0.0

        return df

    def _detect_headerless_csv(self, file_path: Path) -> bool:
        """
        Erkennt ob CSV keinen Header hat.

        Heuristik: Wenn die erste Zelle wie ein Datetime aussieht,
        hat die Datei keinen Header.
        """
        try:
            with open(file_path, "r") as f:
                first_line = f.readline().strip()
                if not first_line:
                    return False

                first_cell = first_line.split(",")[0]

                # Typische Header-Namen
                header_names = {"time", "datetime", "date", "timestamp", "t"}
                if first_cell.lower() in header_names:
                    return False

                # Versuche als Datetime zu parsen
                try:
                    pd.to_datetime(first_cell)
                    return True  # Sieht wie Datetime aus -> kein Header
                except Exception:
                    return False

        except Exception:
            return False

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = None,
        start: datetime = None,
        end: datetime = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """Lädt historische Bars aus CSV."""
        # Cache prüfen
        cache_key = f"{symbol}_{timeframe or self._timeframe}"
        if cache_key in self._cache:
            df = self._cache[cache_key]
        else:
            file_path = self._get_file_path(symbol)

            if not file_path.exists():
                self.log_error(f"File not found: {file_path}")
                return pd.DataFrame()

            try:
                # Prüfe ob CSV einen Header hat
                if self._detect_headerless_csv(file_path):
                    # Header-lose CSV: ForexSB Format (Time,O,H,L,C,V)
                    df = pd.read_csv(
                        file_path,
                        header=None,
                        names=["T", "O", "H", "L", "C", "V"]
                    )
                else:
                    df = pd.read_csv(file_path)

                df = self._normalize_columns(df)
                df = df.sort_index()

                # Cache
                self._cache[cache_key] = df
                self.log_info(f"Loaded {len(df)} bars for {symbol}")

            except Exception as e:
                self.log_error(f"Failed to load {file_path}", e)
                return pd.DataFrame()

        # Filter anwenden
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        if limit:
            df = df.tail(limit)

        return df

    def list_available_symbols(self) -> List[str]:
        """Listet alle verfügbaren Symbole basierend auf Dateien."""
        symbols = []

        # Pattern analysieren um Symbole zu extrahieren
        pattern_parts = self._file_pattern.split("{symbol}")
        prefix = pattern_parts[0] if len(pattern_parts) > 0 else ""
        suffix = pattern_parts[1] if len(pattern_parts) > 1 else ""

        for file in self._data_path.glob("*.csv"):
            name = file.stem + ".csv"
            if name.startswith(prefix) and name.endswith(suffix):
                symbol = name[len(prefix):len(name) - len(suffix)]
                if symbol:
                    symbols.append(symbol)

        return sorted(symbols)


__all__ = ["CSVDataAdapter"]
