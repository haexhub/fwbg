"""Data files are replaced atomically: a failed write never leaves a partial file."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from fwbg.data.atomic import write_csv_atomic, write_text_atomic

FRAME = pd.DataFrame({"T": ["2024-01-01 00:00:00"], "O": [1.0], "C": [2.0]})


def _partial_write_then_fail(self, target, **_kwargs):
    partial = "T,O,C\n2024-01-01 00:00:00,1.0,"
    if isinstance(target, (str, Path)):  # writing straight to the destination
        Path(target).write_text(partial)
    else:
        target.write(partial)
    raise OSError("disk full")


def test_csv_is_written_and_no_temp_file_remains(tmp_path):
    dest = tmp_path / "EURUSD_HOUR_1.csv"
    write_csv_atomic(FRAME, dest, index=False)

    assert pd.read_csv(dest).equals(FRAME)
    assert [p.name for p in tmp_path.iterdir()] == [dest.name]


def test_failed_write_keeps_the_previous_file_intact(tmp_path, monkeypatch):
    dest = tmp_path / "EURUSD_HOUR_1.csv"
    dest.write_text("old,content\n")
    monkeypatch.setattr(pd.DataFrame, "to_csv", _partial_write_then_fail)

    with pytest.raises(OSError, match="disk full"):
        write_csv_atomic(FRAME, dest, index=False)

    assert dest.read_text() == "old,content\n"
    assert list(tmp_path.glob("*.tmp")) == []
    assert [p.name for p in tmp_path.iterdir()] == [dest.name]


def test_failed_first_write_leaves_no_file_behind(tmp_path, monkeypatch):
    dest = tmp_path / "NEW_HOUR_1.csv"
    monkeypatch.setattr(pd.DataFrame, "to_csv", _partial_write_then_fail)

    with pytest.raises(OSError):
        write_csv_atomic(FRAME, dest, index=False)

    assert list(tmp_path.iterdir()) == []


def test_temp_file_is_invisible_to_csv_globs(tmp_path, monkeypatch):
    dest = tmp_path / "EURUSD_HOUR_1.csv"
    seen: list[list[str]] = []

    def _spy(self, handle, **_kwargs):
        seen.append([p.name for p in tmp_path.glob("*.csv")])
        handle.write("x\n")

    monkeypatch.setattr(pd.DataFrame, "to_csv", _spy)
    write_csv_atomic(FRAME, dest, index=False)

    assert seen == [[]]


def test_text_write_replaces_atomically(tmp_path):
    dest = tmp_path / "x.quality.json"
    dest.write_text("old")
    write_text_atomic(dest, '{"a": 1}')

    assert dest.read_text() == '{"a": 1}'
    assert [p.name for p in tmp_path.iterdir()] == [dest.name]


def test_dukascopy_download_does_not_clobber_an_existing_file_on_failure(
    tmp_path, monkeypatch
):
    pytest.importorskip("dukascopy_python")
    from fwbg.data import assets, dukascopy

    index = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    bars = pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0}, index=index
    )
    monkeypatch.setattr(dukascopy, "_fetch_with_progress", lambda *a, **k: bars)
    monkeypatch.setattr(assets, "save_asset_spread", lambda *a, **k: None)

    existing = tmp_path / "EURUSD_HOUR_1.csv"
    existing.write_text("old,content\n")
    monkeypatch.setattr(pd.DataFrame, "to_csv", _partial_write_then_fail)

    with pytest.raises(OSError, match="disk full"):
        dukascopy.download(
            tmp_path,
            ["EURUSD"],
            "HOUR_1",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

    assert existing.read_text() == "old,content\n"
    assert list(tmp_path.glob("*.tmp")) == []
