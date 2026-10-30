"""The feature-discovery SSE stream must not block the API's event loop."""

import asyncio
import json
import threading

import httpx
import pandas as pd
import pytest

from fwbg.api import create_app
from fwbg.api import _paths as paths_mod
from fwbg.api import discovery as discovery_mod
from fwbg.core import data_sources as core_ds
from fwbg.core.data_sources import CSVSourceConfig


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    """Run ``run-1`` with 24 EURUSD trades on a registered CSV datasource."""
    results_dir = tmp_path / "results"
    run_dir = results_dir / "run-1"
    (run_dir / "grid_details" / "EURUSD").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"timeframe": "HOUR"}))
    (run_dir / "strategy.json").write_text(json.dumps({"datasource": "disc"}))
    trades = [
        {"entry_time": f"2024-01-01T{i:02d}:00:00", "pnl_raw": 1.0 if i % 2 else -1.0}
        for i in range(24)
    ]
    fold_results = {"walk_forward": {"fold_details": [{"test_trades_detail": trades}]}}
    (run_dir / "grid_details" / "EURUSD" / "fold_results.json").write_text(json.dumps(fold_results))

    source = CSVSourceConfig(name="disc", path=tmp_path / "data")
    source.path.mkdir()
    source.get_file_path("EURUSD", "HOUR").write_text("")
    monkeypatch.setattr(paths_mod, "get_test_results_dir", lambda: results_dir)
    monkeypatch.setitem(core_ds._DATA_SOURCES, "disc", source)
    return results_dir


def test_data_loading_does_not_block_the_event_loop(results_dir, monkeypatch):
    loading = threading.Event()
    loop_responded = threading.Event()
    responded_while_loading: list[bool] = []

    def _blocking_load(_path):
        # Holds the loader until a coroutine on the event loop answers. If the
        # loader ran on the loop itself, that coroutine could never run.
        loading.set()
        responded_while_loading.append(loop_responded.wait(timeout=2))
        return pd.DataFrame()

    monkeypatch.setattr(discovery_mod, "load_data_aligned", _blocking_load)

    async def _scenario():
        async def _respond_once_loading():
            await asyncio.to_thread(loading.wait, 5)
            loop_responded.set()

        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            _, response = await asyncio.gather(
                _respond_once_loading(),
                client.get("/api/runs/run-1/discovery/EURUSD"),
            )
        return response

    response = asyncio.run(_scenario())

    assert response.status_code == 200
    assert "event: error" in response.text
    assert responded_while_loading == [True]


def test_discovery_completes_and_caches_its_result(results_dir, monkeypatch):
    index = pd.date_range("2023-12-31", periods=60, freq="h")
    bars = pd.DataFrame(
        {"O": 1.0, "H": 1.1, "L": 0.9, "C": 1.0, "V": 1.0, "feat": range(60)},
        index=index,
    )
    monkeypatch.setattr(discovery_mod, "load_data_aligned", lambda _path: bars)
    monkeypatch.setattr(discovery_mod, "_get_available_indicators", lambda: [])

    async def _scenario():
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/runs/run-1/discovery/EURUSD")

    response = asyncio.run(_scenario())

    assert response.status_code == 200
    assert "event: done" in response.text
    cached = json.loads(discovery_mod._cache_path("run-1", "EURUSD").read_text())
    assert cached["total_trades"] == 24
    assert cached["results"][0]["feature"] == "feat"


def test_discovery_cache_filename_is_bounded_for_max_length_symbol(results_dir):
    symbol = "A" * 128

    path = discovery_mod._cache_path("run-1", symbol)

    assert len(path.name) <= 128
    assert path == discovery_mod._cache_path("run-1", symbol)
