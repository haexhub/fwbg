import pandas as pd

from scripts.fetch_jev_predictions import run_batch
from scripts.jev.client import JevProvider


def test_run_batch_calls_each_provider_once_per_bar(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {"O": [1.08, 1.081], "H": [1.082, 1.083], "L": [1.079, 1.080], "C": [1.081, 1.082]},
        index=["t0", "t1"],
    )
    calls = []

    def fake_ask(provider, state, questions):
        calls.append(provider.name)
        return {"is_long_win": 0.5, "is_short_win": 0.5}

    monkeypatch.setattr("scripts.fetch_jev_predictions.ask", fake_ask)

    provider = JevProvider(name="fake", base_url="http://x", api_key_env="X")
    run_batch(
        df,
        provider=provider,
        tp_pips=20,
        sl_pips=20,
        horizon_bars=30,
        cache_path=tmp_path / "out.csv",
    )

    assert calls == ["fake", "fake"]
    cached = pd.read_csv(tmp_path / "out.csv", index_col=0)
    assert list(cached.index) == ["t0", "t1"]


def test_run_batch_skips_already_cached_bars(tmp_path, monkeypatch):
    df = pd.DataFrame({"O": [1.0], "H": [1.0], "L": [1.0], "C": [1.0]}, index=["t0"])
    (tmp_path / "out.csv").write_text("timestamp,is_long_win,is_short_win\nt0,0.9,0.1\n")

    calls = []
    monkeypatch.setattr(
        "scripts.fetch_jev_predictions.ask",
        lambda *a, **k: calls.append(1) or {"is_long_win": 0.0, "is_short_win": 0.0},
    )

    provider = JevProvider(name="fake", base_url="http://x", api_key_env="X")
    run_batch(
        df,
        provider=provider,
        tp_pips=20,
        sl_pips=20,
        horizon_bars=30,
        cache_path=tmp_path / "out.csv",
    )

    assert calls == []  # t0 was already cached, no API call made
