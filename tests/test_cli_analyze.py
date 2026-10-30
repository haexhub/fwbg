"""Tests for ``fwbg analyze`` (MFE/MAE exit analysis)."""

import json

import numpy as np
import pandas as pd
import pytest

from fwbg.cli._analyze import run_analyze


def _write_csv(path, start="2020-01-01", n=300):
    rng = np.random.default_rng(7)
    dates = pd.date_range(start, periods=n, freq="h")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    open_ = close + rng.standard_normal(n) * 0.1
    high = np.maximum(open_, close) + abs(rng.standard_normal(n)) * 0.2
    low = np.minimum(open_, close) - abs(rng.standard_normal(n)) * 0.2
    frame = pd.DataFrame({"T": dates, "O": open_, "H": high, "L": low, "C": close, "V": 100})
    frame.to_csv(path, index=False)
    return path


def _write_strategy(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_strategy_file_exit_strategies_drive_the_analysis(tmp_path):
    csv = _write_csv(tmp_path / "EURUSD_HOUR_1.csv")
    strategy = _write_strategy(
        tmp_path / "s.json",
        {"exit_strategies": [{"name": "trailing_atr", "params": {"atr_period": 7}}]},
    )
    out = tmp_path / "out"

    run_analyze([str(csv), "--strategy-file", str(strategy), "--output-dir", str(out)])

    result = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    assert result["exit_strategy"] == "trailing_atr"
    assert result["exit_params"] == {"atr_period": 7}


def test_strategy_file_with_removed_exit_keys_is_refused(tmp_path, capsys):
    csv = _write_csv(tmp_path / "EURUSD_HOUR_1.csv")
    strategy = _write_strategy(
        tmp_path / "s.json", {"exit_strategy": "atr_based", "exit_params": {"atr_period": 7}}
    )

    with pytest.raises(SystemExit) as exc:
        run_analyze([str(csv), "--strategy-file", str(strategy), "--output-dir", str(tmp_path)])

    assert exc.value.code == 2
    assert "exit_strategies" in capsys.readouterr().err
