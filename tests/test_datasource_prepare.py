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
import pandas as pd
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
        "timestamp,open,high,low,close\n1388534400000,16500.0,16510.0,16495.0,16505.0\n"
    )
    # Datei ohne Symbol-Mapping (soll übersprungen werden)
    (raw / "XX99_UNKNOWN_m15.csv").write_text(
        "timestamp,open,high,low,close\n1388534400000,100.0,101.0,99.0,100.5\n"
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
# Tests: Timestamp-Konvention (naive UTC) und timezone
# ---------------------------------------------------------------------------


def _write_raw(raw_dir: Path, header: str, rows: list[str]) -> None:
    (raw_dir / "DE40_DAX_m15.csv").write_text("\n".join([header, *rows]) + "\n")


def _source(tmp_path: Path, **kwargs):
    from fwbg.core.data_sources import CSVSourceConfig

    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    defaults = dict(
        name="tz_test",
        path=tmp_path / "out",
        file_pattern="{symbol}_MINUTE_15.csv",
        raw_path=raw,
        raw_pattern="{raw_symbol}_m15.csv",
        symbol_map={"DE40_DAX": "DAX"},
    )
    defaults.update(kwargs)
    return CSVSourceConfig(**defaults)


def _timestamps(src, name="DAX_MINUTE_15.csv") -> list[str]:
    return [r["T"] for r in csv.DictReader((src.path / name).open())]


class TestCSVSourceTimezone:
    """Datasource CSVs hold naive UTC. ``timezone`` names the zone of string timestamps.

    Referenz-Timestamp: 1388534400000 ms = 2014-01-01 00:00:00 UTC.
    """

    def test_epoch_timestamps_stay_utc_even_with_timezone(
        self, raw_dir, datasource_dir
    ):
        """Unix epochs are absolute: a configured timezone must not shift them."""
        from fwbg.core.data_sources import CSVSourceConfig

        src = CSVSourceConfig(
            name="tz",
            path=datasource_dir,
            file_pattern="{symbol}_MINUTE_15.csv",
            raw_path=raw_dir,
            raw_pattern="{raw_symbol}_m15.csv",
            timestamp_unit="ms",
            symbol_map={"DE40_DAX": "DAX"},
            timezone="Europe/Berlin",
        )
        src.prepare()
        assert _timestamps(src) == ["2014-01-01 00:00:00", "2014-01-01 00:15:00"]

    def test_no_timezone_keeps_utc(self, raw_dir, datasource_dir):
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
        assert _timestamps(src)[0] == "2014-01-01 00:00:00"

    def test_local_strings_are_converted_to_utc_in_winter(self, tmp_path):
        src = _source(tmp_path, timezone="Europe/Berlin")
        _write_raw(
            tmp_path / "raw",
            "timestamp,open,high,low,close",
            ["2014-01-01 01:00:00,1,2,0.5,1", "2014-01-01 01:15:00,1,2,0.5,1"],
        )
        src.prepare()
        assert _timestamps(src) == ["2014-01-01 00:00:00", "2014-01-01 00:15:00"]

    def test_local_strings_are_converted_to_utc_in_summer(self, tmp_path):
        src = _source(tmp_path, timezone="Europe/Berlin")
        _write_raw(
            tmp_path / "raw",
            "timestamp,open,high,low,close",
            ["2024-07-01 08:00:00,1,2,0.5,1"],
        )
        src.prepare()
        assert _timestamps(src) == ["2024-07-01 06:00:00"]

    def test_strings_without_timezone_are_taken_as_utc(self, tmp_path):
        src = _source(tmp_path)
        _write_raw(
            tmp_path / "raw",
            "timestamp,open,high,low,close",
            ["2024-07-01 08:00:00,1,2,0.5,1"],
        )
        src.prepare()
        assert _timestamps(src) == ["2024-07-01 08:00:00"]

    def test_dst_fallback_keeps_both_occurrences_of_the_repeated_hour(self, tmp_path):
        """2024-10-27 in Berlin: 02:00-02:45 happens twice (CEST, then CET).

        Both passes must survive as distinct UTC bars (00:00-00:45 and 01:00-01:45).
        """
        src = _source(tmp_path, timezone="Europe/Berlin")
        wall = ["01:45"] + [f"02:{m:02d}" for m in (0, 15, 30, 45)] * 2 + ["03:00"]
        _write_raw(
            tmp_path / "raw",
            "timestamp,open,high,low,close",
            [f"2024-10-27 {t}:00,1,2,0.5,1" for t in wall],
        )
        src.prepare()
        assert _timestamps(src) == [
            "2024-10-26 23:45:00",
            *[f"2024-10-27 00:{m:02d}:00" for m in (0, 15, 30, 45)],
            *[f"2024-10-27 01:{m:02d}:00" for m in (0, 15, 30, 45)],
            "2024-10-27 02:00:00",
        ]

    def test_dst_fallback_with_a_single_ambiguous_bar_does_not_fail(self, tmp_path):
        """Hourly data with only one 02:00 cannot be inferred; the first reading wins."""
        src = _source(tmp_path, timezone="Europe/Berlin")
        _write_raw(
            tmp_path / "raw",
            "timestamp,open,high,low,close",
            [
                "2024-10-27 01:00:00,1,2,0.5,1",
                "2024-10-27 02:00:00,1,2,0.5,1",
                "2024-10-27 03:00:00,1,2,0.5,1",
            ],
        )
        src.prepare()
        assert _timestamps(src) == [
            "2024-10-26 23:00:00",
            "2024-10-27 00:00:00",
            "2024-10-27 02:00:00",
        ]

    def test_dst_spring_gap_rows_are_dropped_explicitly(self, tmp_path, caplog):
        """2024-03-31 in Berlin: 02:00-02:59 local never existed."""
        src = _source(tmp_path, timezone="Europe/Berlin")
        _write_raw(
            tmp_path / "raw",
            "timestamp,open,high,low,close",
            [
                "2024-03-31 01:45:00,1,2,0.5,1",
                "2024-03-31 02:15:00,1,2,0.5,1",
                "2024-03-31 03:00:00,1,2,0.5,1",
            ],
        )
        with caplog.at_level("WARNING"):
            src.prepare()
        assert _timestamps(src) == ["2024-03-31 00:45:00", "2024-03-31 01:00:00"]
        assert "nicht existierender Ortszeit" in caplog.text

    def test_timezone_in_to_dict(self, raw_dir, datasource_dir):
        """to_dict() muss timezone enthalten wenn gesetzt."""
        src = _source(datasource_dir.parent, timezone="Europe/Berlin")
        assert src.to_dict().get("timezone") == "Europe/Berlin"

    def test_no_timezone_not_in_to_dict(self, csv_source):
        """to_dict() darf 'timezone' nicht enthalten wenn nicht gesetzt."""
        assert "timezone" not in csv_source.to_dict()

    def test_source_from_dict_restores_timezone(self, tmp_path):
        """source_from_dict() muss timezone korrekt deserialisieren."""
        from fwbg.core.data_sources import source_from_dict

        src = _source(tmp_path, timezone="Europe/Berlin")
        restored = source_from_dict(src.to_dict())
        assert restored.timezone == "Europe/Berlin"


class TestPreparedFileName:
    """The output name comes from file_pattern with the symbol and timeframe filled in."""

    def test_timeframe_placeholder_is_filled_from_the_raw_pattern(self, tmp_path):
        src = _source(tmp_path, file_pattern="{symbol}_{timeframe}.csv")
        _write_raw(
            tmp_path / "raw",
            "timestamp,open,high,low,close",
            ["2024-07-01 08:00:00,1,2,0.5,1"],
        )
        assert src.prepare() == ["DAX"]
        assert sorted(p.name for p in src.path.iterdir()) == ["DAX_m15.csv"]

    def test_timeframe_map_is_applied_like_for_lookups(self, tmp_path):
        src = _source(
            tmp_path,
            file_pattern="{symbol}_{timeframe}.csv",
            timeframe_map={"m15": "MINUTE_15"},
        )
        _write_raw(
            tmp_path / "raw",
            "timestamp,open,high,low,close",
            ["2024-07-01 08:00:00,1,2,0.5,1"],
        )
        src.prepare()
        assert sorted(p.name for p in src.path.iterdir()) == ["DAX_MINUTE_15.csv"]
        assert src.get_file_path("DAX", "m15").exists()

    def test_no_literal_placeholder_in_any_output_name(self, tmp_path):
        src = _source(tmp_path, file_pattern="{symbol}_{timeframe}.csv")
        _write_raw(
            tmp_path / "raw",
            "timestamp,open,high,low,close",
            ["2024-07-01 08:00:00,1,2,0.5,1"],
        )
        src.prepare()
        assert not [p for p in src.path.iterdir() if "{" in p.name]

    def test_pattern_needing_a_timeframe_without_one_is_refused(self, tmp_path):
        src = _source(
            tmp_path,
            file_pattern="{symbol}_{timeframe}.csv",
            raw_pattern="{raw_symbol}.csv",
        )
        (tmp_path / "raw" / "DE40_DAX.csv").write_text(
            "timestamp,open,high,low,close\n2024-07-01 08:00:00,1,2,0.5,1\n"
        )
        assert src.prepare() == []
        assert not src.path.exists() or not list(src.path.iterdir())

    def test_pattern_without_timeframe_placeholder_is_unchanged(self, tmp_path):
        src = _source(tmp_path)
        _write_raw(
            tmp_path / "raw",
            "timestamp,open,high,low,close",
            ["2024-07-01 08:00:00,1,2,0.5,1"],
        )
        src.prepare()
        assert sorted(p.name for p in src.path.iterdir()) == ["DAX_MINUTE_15.csv"]


def test_prepared_dst_fallback_file_loads_with_a_unique_increasing_index(tmp_path):
    from fwbg.data.loader import load_data_aligned

    src = _source(tmp_path, timezone="Europe/Berlin")
    wall = [f"{h:02d}:00" for h in (1, 2, 2, 3)]
    _write_raw(
        tmp_path / "raw",
        "timestamp,open,high,low,close",
        [f"2024-10-27 {t}:00,1,2,0.5,1" for t in wall],
    )
    src.prepare()

    df = load_data_aligned(str(src.get_file_path("DAX")))

    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert list(df.index.hour) == [23, 0, 1, 2]
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
