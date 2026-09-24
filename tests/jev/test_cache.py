import pandas as pd
import pytest

from scripts.jev.cache import PredictionCache


def test_cache_round_trip(tmp_path):
    path = tmp_path / "EURUSD_M1.csv"
    cache = PredictionCache(path)
    assert cache.pending_timestamps(all_timestamps=["t0", "t1", "t2"]) == ["t0", "t1", "t2"]

    cache.append("t0", {"is_long_win": 0.6, "is_short_win": 0.1})
    cache.append("t1", {"is_long_win": 0.4, "is_short_win": 0.3})

    assert cache.pending_timestamps(all_timestamps=["t0", "t1", "t2"]) == ["t2"]

    reloaded = PredictionCache(path)
    df = reloaded.as_dataframe()
    assert list(df.index) == ["t0", "t1"]
    assert df.loc["t0", "is_long_win"] == 0.6


def test_cache_survives_reopen_after_partial_run(tmp_path):
    path = tmp_path / "EURUSD_M1.csv"
    PredictionCache(path).append("t0", {"is_long_win": 0.6, "is_short_win": 0.1})

    resumed = PredictionCache(path)
    assert resumed.pending_timestamps(all_timestamps=["t0", "t1"]) == ["t1"]


def test_append_failure_does_not_corrupt_existing_cache(tmp_path, monkeypatch):
    path = tmp_path / "EURUSD_M1.csv"
    cache = PredictionCache(path)
    cache.append("t0", {"is_long_win": 0.6, "is_short_win": 0.1})
    original_content = path.read_text()

    def boom(self, *args, **kwargs):
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)

    with pytest.raises(OSError):
        cache.append("t1", {"is_long_win": 0.4, "is_short_win": 0.3})

    assert path.read_text() == original_content
