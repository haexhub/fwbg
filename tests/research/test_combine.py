"""``fwbg combine``: a gated portfolio run over frozen strategy sleeves."""

import faulthandler
import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from fwbg.optimization import sleeves
from fwbg.optimization.sleeves import PortfolioSpecError, combine, load_spec, trade_returns
from fwbg.research.ledger import LEDGER_FILENAME, TrialKind, TrialStatus, read_trials
from fwbg.research.runs import GATE_ENV
from fwbg.results import storage

cli = importlib.import_module("fwbg.cli.main")
LAST = pd.Timestamp("2026-06-30")
CUTOFF = pd.Timestamp("2025-04-30")  # the S-0007 card of gate_register
PARAMS = {"lookback_days": 60, "min_days": 40, "target_volatility": 0.10, "max_scale": 5.0}


def trades(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """One unit-notional trade per business day; AAA is calm, BBB three times as volatile."""
    days = pd.bdate_range("2023-01-02", LAST)
    rng = np.random.default_rng(1 if symbol == "AAA" else 2)
    scale = 0.004 if symbol == "AAA" else 0.012
    returns = rng.normal(0.0008, scale, len(days))
    return [
        {"exit_time": f"{d.date()}T16:00:00", "entry_price": 100.0, "pnl_raw": 100.0 * r}
        for d, r in zip(days, returns)
        if start <= d < end
    ]


def test_trade_returns_are_net_per_unit_of_notional_by_exit_date():
    series = trade_returns(
        [
            {"exit_time": "2024-01-02T10:00:00", "entry_price": 200.0, "pnl_raw": 2.0},
            {"exit_time": "2024-01-02T15:00:00", "entry_price": 100.0, "pnl_raw": -0.5},
            {"exit_time": "2024-01-03T00:00:00", "entry_price": 50.0, "pnl_raw": 1.0},
        ]
    )
    assert series.to_dict() == {pd.Timestamp("2024-01-02"): 0.005, pd.Timestamp("2024-01-03"): 0.02}
    with pytest.raises(ValueError, match="dated"):
        trade_returns([{"exit_time": "2024-01-02", "pnl_raw": 1.0}])


def streams(end=LAST):
    return {s: trade_returns(trades(s, pd.Timestamp("2023-01-01"), end)) for s in ("AAA", "BBB")}


def test_weights_are_inverse_to_volatility_and_causal():
    returns, notional = combine(streams(), PARAMS, {"AAA": None, "BBB": None})
    ratio = (notional["AAA"] / notional["BBB"]).median()
    assert 2.5 < ratio < 3.5  # BBB is three times as volatile
    first = pd.bdate_range("2023-01-02", LAST)[PARAMS["min_days"]]
    assert returns.index[0] > first  # weights need min_days before them, then a portfolio vol
    # a change after a day does not move that day's weights or return
    later = streams()
    later["BBB"].loc["2025-01-02":] *= 5
    changed_returns, changed = combine(later, PARAMS, {"AAA": None, "BBB": None})
    day = pd.Timestamp("2025-01-02")
    pd.testing.assert_frame_equal(changed.loc[:day], notional.loc[:day])
    pd.testing.assert_series_equal(changed_returns.loc[: day - pd.Timedelta(days=1)],
                                   returns.loc[: day - pd.Timedelta(days=1)])


def test_max_notional_caps_a_sleeve():
    _, notional = combine(streams(), PARAMS, {"AAA": 0.5, "BBB": None})
    assert notional["AAA"].max() <= 0.5 and (notional["AAA"] == 0.5).any()


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def sleeve_files(tmp_path):
    for sym in ("AAA", "BBB"):
        write(tmp_path / f"{sym.lower()}.json",
              {"name": sym, "timeframe": "DAY_1", "assets": {"filter": [sym]}, "model": {}})
    return tmp_path


def spec(directory, **extra):
    data = {
        "name": "two sleeves",
        "sleeves": [
            {"name": "calm", "strategy_file": "aaa.json"},
            {"name": "wild", "strategy_file": "bbb.json", "max_notional": 4.0},
        ],
        **PARAMS,
        **extra,
    }
    return write(directory / "portfolio.json", data)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"sleeves": [{"strategy_file": "aaa.json"}]}, "two sleeves"),
        ({"leverage": 2}, "unknown portfolio keys"),
        ({"min_days": 300}, "min_days"),
        ({"sleeves": [{"strategy_file": "aaa.json"}, {"strategy_file": "aaa.json"}]}, "duplicate"),
        ({"sleeves": [{"strategy_file": "aaa.json"}, {"strategy_file": "nope.json"}]}, "cannot load"),
    ],
)
def test_bad_specs_are_refused(sleeve_files, change, match):
    def load(path):
        try:
            return json.loads(open(path).read())
        except OSError:
            return None

    with pytest.raises(PortfolioSpecError, match=match):
        load_spec(spec(sleeve_files, **change), load, dict)


def test_a_sleeve_must_not_set_its_own_window(sleeve_files):
    write(sleeve_files / "bbb.json", {"name": "B", "end_date": "2024-01-01"})
    with pytest.raises(PortfolioSpecError, match="end_date"):
        load_spec(spec(sleeve_files), lambda p: json.loads(open(p).read()), dict)


class Fakes:
    """process_symbol and the frozen holdout evaluation on synthetic trades."""

    def __init__(self, statuses):
        self.statuses = statuses
        self.developed = []
        self.looked = []
        self.no_frozen = set()

    def select(self, strategy, metadata, asset_filter=None):
        return [f"/data/{metadata['assets']['filter'][0]}_DAY_1.csv"]

    def process(self, csv_path, strategy):
        sym = sleeves.symbol_of(csv_path)
        self.developed.append((sym, strategy.end_date, strategy.holdout_cutoff))
        end = pd.Timestamp(strategy.end_date) if strategy.end_date else LAST
        return {
            "symbol": sym,
            "status": self.statuses[sym],
            "trades_detailed": trades(sym, pd.Timestamp("2023-01-01"), end),
            "unified_candidate": {"tp": 1.0},
        }

    def frozen(self, csv_path, strategy, development):
        sym = development["symbol"]
        if sym in self.no_frozen:
            return {"symbol": sym, "holdout": {"evaluated": False, "reason": "no frozen setting"}}
        self.looked.append((sym, strategy.holdout_cutoff))
        look = {"cutoff": strategy.holdout_cutoff, "evaluated": True}
        return look, trades(sym, pd.Timestamp(strategy.holdout_cutoff), LAST + pd.Timedelta(days=1))


@pytest.fixture
def gated(tmp_path, gate_register, monkeypatch, sleeve_files):
    monkeypatch.setenv(GATE_ENV, "on")
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(gate_register))
    monkeypatch.setattr(storage, "RESULTS_BASE_PATH", str(tmp_path / "results"))
    monkeypatch.setattr(faulthandler, "enable", lambda *args, **kwargs: None)
    fakes = Fakes({"AAA": "ok", "BBB": "not_significant"})
    monkeypatch.setattr(sleeves, "select_run_files", fakes.select)
    monkeypatch.setattr(sleeves, "process_symbol", fakes.process)
    monkeypatch.setattr(sleeves, "frozen_trades_after_cutoff", fakes.frozen)
    fakes.spec = spec(sleeve_files)
    fakes.register = gate_register
    fakes.results = tmp_path / "results"
    return fakes


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["fwbg", "combine", *args])
    cli.main()


def portfolio(results, run_id):
    return json.loads((results / run_id / "grid_details" / "PORTFOLIO" / "config.json").read_text())


def test_iteration_is_gated_clamped_to_the_cutoff_and_recorded(gated, monkeypatch):
    run_cli(monkeypatch, "--spec", str(gated.spec), "--research-id", "S-0007",
            "--note", "two sleeves", "--run-id", "p1")
    assert gated.developed == [("AAA", "2025-04-30", None), ("BBB", "2025-04-30", None)]
    assert gated.looked == []
    [trial] = read_trials(gated.register / LEDGER_FILENAME)
    assert trial.kind is TrialKind.ITERATION and trial.status is TrialStatus.OK
    metrics = trial.metrics["PORTFOLIO"]
    assert metrics["sharpe"] > 0 and metrics["trades"] > 0
    result = portfolio(gated.results, "p1")
    assert result["mean_notional"]["calm"] > result["mean_notional"]["wild"]
    assert pd.read_csv(gated.results / "p1" / "grid_details" / "PORTFOLIO" / "daily.csv")["date"].max() < "2025-04-30"


def test_holdout_evaluates_every_sleeve_once_and_reports_after_the_cutoff(gated, monkeypatch):
    run_cli(monkeypatch, "--spec", str(gated.spec), "--research-id", "S-0007",
            "--note", "look", "--run-id", "h1", "--holdout")
    # each sleeve develops before the cutoff; the not significant one is looked at too
    assert gated.developed == [("AAA", "2025-04-30", None), ("BBB", "2025-04-30", None)]
    assert gated.looked == [("AAA", "2025-04-30"), ("BBB", "2025-04-30")]
    [trial] = read_trials(gated.register / LEDGER_FILENAME)
    assert trial.kind is TrialKind.HOLDOUT and trial.status is TrialStatus.OK
    result = portfolio(gated.results, "h1")
    assert result["holdout"] == {"cutoff": "2025-04-30", "evaluated": True}
    daily = pd.read_csv(gated.results / "h1" / "grid_details" / "PORTFOLIO" / "daily.csv")
    assert daily["date"].min() >= "2025-04-30"
    assert result["metrics"]["trades"] == 2 * len(pd.bdate_range(CUTOFF, LAST))
    assert result["metrics"]["test_period_years"] < 1.3


def test_holdout_with_a_sleeve_that_cannot_be_evaluated_is_an_error(gated, monkeypatch):
    gated.no_frozen.add("BBB")
    run_cli(monkeypatch, "--spec", str(gated.spec), "--research-id", "S-0007",
            "--note", "look", "--run-id", "h2", "--holdout")
    result = portfolio(gated.results, "h2")
    assert result["status"] == "error" and "wild" in result["reason"]
    assert result["holdout"]["evaluated"] is True  # AAA's data after the cutoff was seen
    [trial] = read_trials(gated.register / LEDGER_FILENAME)
    assert trial.kind is TrialKind.HOLDOUT


def test_combine_without_a_research_id_is_refused(gated, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        run_cli(monkeypatch, "--spec", str(gated.spec), "--note", "x")
    assert exc.value.code == 2 and gated.developed == []
