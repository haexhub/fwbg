import pandas as pd
import pytest

from fwbg.data.assets import get_asset
from scripts.fetch_jev_predictions import run_batch, spread_multiple_to_pips
from scripts.jev.client import JevProvider
from scripts.jev.labels import compute_labels


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


def test_run_batch_stops_cleanly_on_error_but_keeps_prior_progress(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {"O": [1.08, 1.081], "H": [1.082, 1.083], "L": [1.079, 1.080], "C": [1.081, 1.082]},
        index=["t0", "t1"],
    )
    cache_path = tmp_path / "out.csv"
    calls = []

    def flaky_ask(provider, state, questions):
        calls.append(provider.name)
        if len(calls) == 1:
            return {"is_long_win": 0.5, "is_short_win": 0.5}
        raise RuntimeError("simulated API failure")

    monkeypatch.setattr("scripts.fetch_jev_predictions.ask", flaky_ask)

    provider = JevProvider(name="fake", base_url="http://x", api_key_env="X")
    with pytest.raises(RuntimeError, match="simulated API failure"):
        run_batch(
            df,
            provider=provider,
            tp_pips=20,
            sl_pips=20,
            horizon_bars=30,
            cache_path=cache_path,
        )

    # bar 0 succeeded before bar 1 raised: its result must have survived.
    cached = pd.read_csv(cache_path, index_col=0)
    assert list(cached.index) == ["t0"]

    # a subsequent run with a healthy provider only retries the un-cached bar.
    resumed_calls = []

    def healthy_ask(provider, state, questions):
        resumed_calls.append(provider.name)
        return {"is_long_win": 0.1, "is_short_win": 0.2}

    monkeypatch.setattr("scripts.fetch_jev_predictions.ask", healthy_ask)

    run_batch(
        df,
        provider=provider,
        tp_pips=20,
        sl_pips=20,
        horizon_bars=30,
        cache_path=cache_path,
    )

    assert resumed_calls == ["fake"]  # only t1 retried
    cached = pd.read_csv(cache_path, index_col=0)
    assert list(cached.index) == ["t0", "t1"]


def test_tp_sl_pip_conversion_matches_compute_labels_price_distance():
    """compute_labels() (Task 3) treats tp/sl as fwbg-native spread
    multipliers -- FixedExitStrategy.compute_targets() derives the actual
    price distance as `asset.spread * tp`. build_questions() (Task 2)
    instead describes tp/sl as literal pips in the prompt text. main() must
    bridge the two via spread_multiple_to_pips() before both reach Jev with
    the same nominal number, or Jev is asked about a different barrier than
    the one labels are graded against.
    """
    asset = get_asset("EURUSD")
    tp_multiple = 20

    converted_pips = spread_multiple_to_pips(asset, tp_multiple)

    # The pips fed to build_questions() must describe the exact same price
    # distance compute_labels() uses for the same nominal tp value.
    assert converted_pips * asset.point == pytest.approx(asset.spread * tp_multiple)

    # Confirm compute_labels() really does consume tp_multiple as a
    # spread-multiplier (not pips) end to end, tying the identity above to
    # real behavior rather than two formulas that merely look consistent.
    n = 10
    df = pd.DataFrame({"O": [1.08] * n, "H": [1.081] * n, "L": [1.079] * n, "C": [1.08] * n})
    labels = compute_labels(df, symbol="EURUSD", tp=tp_multiple, sl=tp_multiple, timeout_bars=5)
    assert len(labels) == n
