"""Cached exit-optimization results are keyed by everything that changes the result."""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import fwbg.core.data_sources as ds_mod
from fwbg.api import create_app
from fwbg.core.data_sources import CSVSourceConfig

URL = "/api/exploration/exit-optimization"


def write_bars(path, seed=3, n=400):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    open_ = close + rng.standard_normal(n) * 0.1
    frame = pd.DataFrame(
        {
            "T": pd.date_range("2025-01-01", periods=n, freq="h"),
            "O": open_,
            "H": np.maximum(open_, close) + 0.2,
            "L": np.minimum(open_, close) - 0.2,
            "C": close,
            "V": 100,
        }
    )
    frame.to_csv(path, index=False)


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("FWBG_TEST_RESULTS_DIR", str(tmp_path / "results"))
    source = CSVSourceConfig(name="t1", path=tmp_path / "data")
    source.path.mkdir()
    write_bars(source.path / "EURUSD_HOUR.csv")
    monkeypatch.setitem(ds_mod._DATA_SOURCES, "t1", source)
    with TestClient(create_app()) as client:
        client.data_dir = source.path
        client.cache_dir = tmp_path / "results" / "exploration"
        yield client


def post(api, **fields):
    body = {"source": "t1", "symbol": "EURUSD", "timeframe": "HOUR", **fields}
    response = api.post(URL, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def cached(api):
    return sorted(p.name for p in api.cache_dir.glob("*.json"))


def test_the_same_analysis_reuses_its_cache_file(api):
    first = post(api)
    second = post(api)

    assert first["cache_key"] == second["cache_key"]
    assert cached(api) == [f"{first['cache_key']}.json"]
    assert first["cache_key"].startswith("EURUSD_HOUR_")


@pytest.mark.parametrize(
    "change",
    [
        {"max_bars": 24},
        {"exit_params": {"atr_period": 20}},
        {"exit_strategy": "fixed_atr"},
    ],
)
def test_a_changed_parameter_gets_its_own_cache_file(api, change):
    base = post(api)
    other = post(api, **change)

    assert other["cache_key"] != base["cache_key"]
    assert cached(api) == sorted(
        [f"{base['cache_key']}.json", f"{other['cache_key']}.json"]
    )
    # The first result is still on disk, unchanged by the second run.
    stored = json.loads((api.cache_dir / f"{base['cache_key']}.json").read_text())
    assert stored["max_bars_forward"] == base["max_bars_forward"]
    assert stored["exit_params"] == base["exit_params"]


def test_changed_data_gets_a_new_cache_file(api):
    before = post(api)
    write_bars(api.data_dir / "EURUSD_HOUR.csv", seed=9, n=500)

    after = post(api)

    assert after["cache_key"] != before["cache_key"]
    assert len(cached(api)) == 2


def test_a_different_data_window_gets_its_own_cache_file(api):
    from fwbg.api.exploration import exit_optimization as module

    base = post(api)
    req = module.ExitOptimizationRequest(source="t1", symbol="EURUSD", timeframe="HOUR")
    path = str(api.data_dir / "EURUSD_HOUR.csv")

    narrowed = module._cache_key(
        req, path, {"start_date": None, "end_date": "2025-01-10"}
    )

    assert narrowed != base["cache_key"]
    assert (
        module._cache_key(req, path, {"start_date": None, "end_date": None})
        == base["cache_key"]
    )


def test_get_returns_the_most_recent_result_and_can_filter_by_timeframe(api):
    first = post(api)
    second = post(api, max_bars=24)

    latest = api.get(f"{URL}/EURUSD")
    assert latest.status_code == 200
    assert latest.json()["cache_key"] == second["cache_key"]
    assert latest.json()["analyzed_at"] >= first["analyzed_at"]

    assert api.get(f"{URL}/EURUSD", params={"timeframe": "NOPE"}).status_code == 404
    assert api.get(f"{URL}/GBPUSD").status_code == 404


def test_list_shows_every_cached_variant(api):
    post(api)
    post(api, max_bars=24)

    rows = api.get(URL).json()

    assert len(rows) == 2
    assert {row["max_bars_forward"] for row in rows} == {24, 48}
