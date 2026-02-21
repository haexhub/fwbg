"""
Tests for CSVSourceConfig.prepare() - ETL raw → datasource conversion.

Sichert ab:
- Unix-ms-Timestamps werden korrekt konvertiert
- symbol_map wird angewendet (Dateiname-Präfix → Symbol)
- Ausgabe enthält korrekte OHLCV-Spalten (T, O, H, L, C, V=0)
- Dateien werden korrekt im datasource-Verzeichnis abgelegt
- Fehlende symbol_map-Einträge werden übersprungen (kein Absturz)
- timezone-Feld konvertiert UTC-Timestamps in die Ziel-Zeitzone (DST-aware)
"""
import csv
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_dir(tmp_path):
    """Erstellt ein raw-Verzeichnis mit Dukascopy-Beispieldateien."""
    raw = tmp_path / "raw"
    raw.mkdir()

    # Standard-Dukascopy-Format: Unix-ms, kein Volume
    (raw / "DE40_DAX_m15.csv").write_text(
        "timestamp,open,high,low,close\n"
        "1388534400000,9552.1,9553.0,9551.0,9552.5\n"
        "1388535300000,9552.5,9554.0,9552.0,9553.0\n"
    )
    (raw / "US30_DJI_m15.csv").write_text(
        "timestamp,open,high,low,close\n"
        "1388534400000,16500.0,16510.0,16495.0,16505.0\n"
    )
    # Datei ohne Symbol-Mapping (soll übersprungen werden)
    (raw / "XX99_UNKNOWN_m15.csv").write_text(
        "timestamp,open,high,low,close\n"
        "1388534400000,100.0,101.0,99.0,100.5\n"
    )
    return raw


@pytest.fixture
def datasource_dir(tmp_path):
    return tmp_path / "datasource"


@pytest.fixture
def csv_source(raw_dir, datasource_dir):
    """CSVSourceConfig mit dukascopy-typischen Feldern."""
    from fwbg.core.data_sources import CSVSourceConfig
    return CSVSourceConfig(
        name="dukascopy_test",
        path=datasource_dir,
        file_pattern="{symbol}_MINUTE_15.csv",
        timeframe_map={"MINUTE_15": "MINUTE_15"},
        raw_path=raw_dir,
        raw_pattern="{raw_symbol}_m15.csv",
        timestamp_unit="ms",
        symbol_map={
            "DE40_DAX": "DAX",
            "US30_DJI": "DOW30",
        },
    )


# ---------------------------------------------------------------------------
# Tests: prepare() Grundverhalten
# ---------------------------------------------------------------------------

class TestCSVSourcePrepare:

    def test_prepare_creates_datasource_dir(self, csv_source, datasource_dir):
        """prepare() erstellt das Ausgabe-Verzeichnis falls nötig."""
        assert not datasource_dir.exists()
        csv_source.prepare()
        assert datasource_dir.exists()

    def test_prepare_creates_correct_files(self, csv_source, datasource_dir):
        """Für jeden symbol_map-Eintrag muss eine Datei erzeugt werden."""
        csv_source.prepare()
        assert (datasource_dir / "DAX_MINUTE_15.csv").exists()
        assert (datasource_dir / "DOW30_MINUTE_15.csv").exists()

    def test_prepare_skips_unmapped_files(self, csv_source, datasource_dir):
        """Dateien ohne symbol_map-Eintrag werden übersprungen."""
        csv_source.prepare()
        # XX99_UNKNOWN hat kein Mapping → keine UNKNOWN_MINUTE_15.csv
        unknown = list(datasource_dir.glob("UNKNOWN*.csv"))
        assert not unknown, f"Unbekannte Datei wurde trotzdem erstellt: {unknown}"

    def test_prepare_converts_unix_ms_timestamps(self, csv_source, datasource_dir):
        """Unix-ms-Timestamps müssen als lesbare Datetime-Strings gespeichert werden."""
        csv_source.prepare()
        rows = list(csv.DictReader((datasource_dir / "DAX_MINUTE_15.csv").open()))
        t = rows[0]["T"]
        # Kein Integer mehr, sondern ISO-Format
        assert not t.isdigit(), f"Timestamp wurde nicht konvertiert: {t!r}"
        assert "2014" in t, f"Falsches Jahr im Timestamp: {t!r}"

    def test_prepare_output_has_ohlcv_columns(self, csv_source, datasource_dir):
        """Ausgabe-CSV muss Spalten T, O, H, L, C, V enthalten."""
        csv_source.prepare()
        rows = list(csv.DictReader((datasource_dir / "DAX_MINUTE_15.csv").open()))
        assert rows, "Ausgabe-CSV ist leer"
        for col in ["T", "O", "H", "L", "C", "V"]:
            assert col in rows[0], f"Spalte '{col}' fehlt in Ausgabe"

    def test_prepare_volume_is_zero(self, csv_source, datasource_dir):
        """Dukascopy hat kein Volume → V muss 0 sein."""
        csv_source.prepare()
        rows = list(csv.DictReader((datasource_dir / "DAX_MINUTE_15.csv").open()))
        for row in rows:
            assert float(row["V"]) == 0.0, f"V ist nicht 0: {row['V']}"

    def test_prepare_preserves_ohlc_values(self, csv_source, datasource_dir):
        """OHLC-Werte müssen erhalten bleiben."""
        csv_source.prepare()
        rows = list(csv.DictReader((datasource_dir / "DAX_MINUTE_15.csv").open()))
        assert float(rows[0]["O"]) == pytest.approx(9552.1)
        assert float(rows[0]["H"]) == pytest.approx(9553.0)
        assert float(rows[0]["L"]) == pytest.approx(9551.0)
        assert float(rows[0]["C"]) == pytest.approx(9552.5)

    def test_prepare_all_rows_converted(self, csv_source, datasource_dir):
        """Alle Zeilen müssen in der Ausgabe vorhanden sein."""
        csv_source.prepare()
        rows = list(csv.DictReader((datasource_dir / "DAX_MINUTE_15.csv").open()))
        assert len(rows) == 2  # 2 Zeilen in der Testdatei

    def test_prepare_without_raw_path_is_noop(self, datasource_dir):
        """CSVSourceConfig ohne raw_path → prepare() tut nichts (kein Absturz)."""
        from fwbg.core.data_sources import CSVSourceConfig
        src = CSVSourceConfig(name="no_raw", path=datasource_dir)
        # Kein Fehler, keine Ausgabe
        src.prepare()
        assert not datasource_dir.exists() or not list(datasource_dir.glob("*.csv"))


# ---------------------------------------------------------------------------
# Tests: Serialisierung / Deserialisierung
# ---------------------------------------------------------------------------

class TestCSVSourceSerialization:

    def test_to_dict_includes_raw_fields(self, csv_source):
        """to_dict() muss raw_path, raw_pattern, timestamp_unit, symbol_map enthalten."""
        d = csv_source.to_dict()
        assert "raw_path" in d
        assert "raw_pattern" in d
        assert "timestamp_unit" in d
        assert "symbol_map" in d

    def test_source_from_dict_roundtrip(self, csv_source, raw_dir, datasource_dir):
        """source_from_dict(to_dict()) muss identische Config reproduzieren."""
        from fwbg.core.data_sources import source_from_dict
        d = csv_source.to_dict()
        restored = source_from_dict(d)
        assert restored.name == csv_source.name
        assert str(restored.raw_path) == str(csv_source.raw_path)
        assert restored.timestamp_unit == csv_source.timestamp_unit
        assert restored.symbol_map == csv_source.symbol_map


# ---------------------------------------------------------------------------
# Tests: Timezone-Konvertierung
# ---------------------------------------------------------------------------

class TestCSVSourceTimezone:
    """Sichert ab dass UTC-Timestamps in die konfigurierte Zeitzone konvertiert werden.

    Referenz-Timestamp: 1388534400000 ms
      = 2014-01-01 00:00:00 UTC
      = 2014-01-01 01:00:00 Europe/Berlin (CET = UTC+1 im Winter)
      = 2014-01-01 09:00:00 Asia/Tokyo    (JST = UTC+9)
    """

    def _make_source(self, raw_dir, datasource_dir, timezone):
        from fwbg.core.data_sources import CSVSourceConfig
        return CSVSourceConfig(
            name="tz_test",
            path=datasource_dir,
            file_pattern="{symbol}_MINUTE_15.csv",
            raw_path=raw_dir,
            raw_pattern="{raw_symbol}_m15.csv",
            timestamp_unit="ms",
            symbol_map={"DE40_DAX": "DAX"},
            timezone=timezone,
        )

    def test_no_timezone_keeps_utc(self, raw_dir, datasource_dir):
        """Ohne timezone bleibt der Timestamp in UTC (bisheriges Verhalten)."""
        from fwbg.core.data_sources import CSVSourceConfig
        src = CSVSourceConfig(
            name="no_tz",
            path=datasource_dir,
            file_pattern="{symbol}_MINUTE_15.csv",
            raw_path=raw_dir,
            raw_pattern="{raw_symbol}_m15.csv",
            timestamp_unit="ms",
            symbol_map={"DE40_DAX": "DAX"},
        )
        src.prepare()
        rows = list(csv.DictReader((datasource_dir / "DAX_MINUTE_15.csv").open()))
        # 1388534400000 ms = 2014-01-01 00:00:00 UTC
        assert rows[0]["T"] == "2014-01-01 00:00:00"

    def test_europe_berlin_winter_offset(self, raw_dir, datasource_dir):
        """timezone='Europe/Berlin' verschiebt UTC+1h im Winter (CET)."""
        src = self._make_source(raw_dir, datasource_dir, "Europe/Berlin")
        src.prepare()
        rows = list(csv.DictReader((datasource_dir / "DAX_MINUTE_15.csv").open()))
        # 2014-01-01 00:00:00 UTC → 2014-01-01 01:00:00 CET
        assert rows[0]["T"] == "2014-01-01 01:00:00", (
            f"Erwartet 01:00:00 CET, bekommen: {rows[0]['T']}"
        )

    def test_europe_berlin_second_bar(self, raw_dir, datasource_dir):
        """Auch der zweite Bar wird korrekt verschoben (+15 min)."""
        src = self._make_source(raw_dir, datasource_dir, "Europe/Berlin")
        src.prepare()
        rows = list(csv.DictReader((datasource_dir / "DAX_MINUTE_15.csv").open()))
        # 1388535300000 ms = 2014-01-01 00:15:00 UTC → 01:15:00 CET
        assert rows[1]["T"] == "2014-01-01 01:15:00", (
            f"Erwartet 01:15:00 CET, bekommen: {rows[1]['T']}"
        )

    def test_europe_berlin_dst_summer(self, tmp_path):
        """timezone='Europe/Berlin' verschiebt UTC+2h im Sommer (CEST)."""
        raw = tmp_path / "raw"
        raw.mkdir()
        # 2024-07-01 06:00:00 UTC = 2024-07-01 08:00:00 CEST (UTC+2)
        ts_ms = 1719813600000  # 2024-07-01 06:00:00 UTC
        (raw / "DE40_DAX_m15.csv").write_text(
            f"timestamp,open,high,low,close\n{ts_ms},100,101,99,100\n"
        )
        from fwbg.core.data_sources import CSVSourceConfig
        src = CSVSourceConfig(
            name="dst_test",
            path=tmp_path / "out",
            file_pattern="{symbol}_MINUTE_15.csv",
            raw_path=raw,
            raw_pattern="{raw_symbol}_m15.csv",
            timestamp_unit="ms",
            symbol_map={"DE40_DAX": "DAX"},
            timezone="Europe/Berlin",
        )
        src.prepare()
        rows = list(csv.DictReader((tmp_path / "out" / "DAX_MINUTE_15.csv").open()))
        assert rows[0]["T"] == "2024-07-01 08:00:00", (
            f"Erwartet 08:00:00 CEST, bekommen: {rows[0]['T']}"
        )

    def test_timezone_in_to_dict(self, raw_dir, datasource_dir):
        """to_dict() muss timezone enthalten wenn gesetzt."""
        src = self._make_source(raw_dir, datasource_dir, "Europe/Berlin")
        d = src.to_dict()
        assert d.get("timezone") == "Europe/Berlin"

    def test_no_timezone_not_in_to_dict(self, csv_source):
        """to_dict() darf 'timezone' nicht enthalten wenn nicht gesetzt."""
        d = csv_source.to_dict()
        assert "timezone" not in d

    def test_source_from_dict_restores_timezone(self, raw_dir, datasource_dir):
        """source_from_dict() muss timezone korrekt deserialisieren."""
        from fwbg.core.data_sources import source_from_dict
        src = self._make_source(raw_dir, datasource_dir, "Europe/Berlin")
        restored = source_from_dict(src.to_dict())
        assert restored.timezone == "Europe/Berlin"

    def test_dst_fallback_removes_duplicate_timestamps(self, tmp_path):
        """Duplikate naive Timestamps beim DST-Rückfall (Oktober) werden entfernt.

        Beim Rückfall in Europa/Berlin (letzter Sonntag Oktober) drehen die Uhren
        von 03:00 CEST (= 01:00 UTC) zurück auf 02:00 CET. Dadurch entstehen
        in den naiven Strings doppelte Einträge für 02:00-02:45 local.

        Input: 9 Bars  00:00–02:00 UTC (15-Min-Raster)
        Erwartung: 5 Zeilen (02:00, 02:15, 02:30, 02:45, 03:00 lokal, je erste UTC behalten)
        """
        raw = tmp_path / "raw"
        raw.mkdir()
        # 2024-10-27 DST fall-back: 01:00 UTC = 03:00 CEST → 02:00 CET
        # Bars: 00:00..02:00 UTC every 15 min = 9 bars
        base_ms = 1729987200000  # 2024-10-27 00:00:00 UTC
        step_ms = 900000         # 15 min
        lines = ["timestamp,open,high,low,close"]
        for i in range(9):
            lines.append(f"{base_ms + i * step_ms},100,101,99,100")
        (raw / "DE40_DAX_m15.csv").write_text("\n".join(lines) + "\n")

        from fwbg.core.data_sources import CSVSourceConfig
        src = CSVSourceConfig(
            name="dst_dedup",
            path=tmp_path / "out",
            file_pattern="{symbol}_MINUTE_15.csv",
            raw_path=raw,
            raw_pattern="{raw_symbol}_m15.csv",
            timestamp_unit="ms",
            symbol_map={"DE40_DAX": "DAX"},
            timezone="Europe/Berlin",
        )
        src.prepare()
        rows = list(csv.DictReader((tmp_path / "out" / "DAX_MINUTE_15.csv").open()))
        timestamps = [r["T"] for r in rows]
        # Deduplicated: no duplicate naive timestamps
        assert len(timestamps) == len(set(timestamps)), (
            f"Duplikate in Timestamps: {[t for t in timestamps if timestamps.count(t) > 1]}"
        )
        # 9 raw → 5 unique (4 pairs deduplicated + 1 last bar)
        assert len(rows) == 5, f"Erwartet 5 Zeilen nach DST-Dedup, bekommen: {len(rows)}"
        assert rows[0]["T"] == "2024-10-27 02:00:00"
        assert rows[-1]["T"] == "2024-10-27 03:00:00"
