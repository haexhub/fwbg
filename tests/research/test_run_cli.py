"""The research gate on the backtest CLI (``python -m fwbg.cli``)."""

import faulthandler
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from fwbg.data import config as data_config
from fwbg.optimization.resource_manager import SimplePoolManager
from fwbg.research.gate import Denial, DenialCode
from fwbg.research.ledger import LEDGER_FILENAME, TrialStatus, read_trials
from fwbg.research.pending import read_pending
from fwbg.research.runs import GATE_ENV, GateDenied, parse_check_output
from fwbg.results import storage
from fwbg.utils import progress
from tests.research.factories import make_run

# ``fwbg.cli`` re-exports ``main()``, which shadows the module of the same name.
cli = importlib.import_module("fwbg.cli.main")

PIPELINE = {"indicators": [{"name": "cusum_events"}]}


class FakeOptimizer:
    """Stands in for ``run_optimizer``: gates on EURUSD and writes an ok run."""

    def __init__(self, results):
        self.results = results
        self.calls = []
        self.passed = False
        self.error = None
        self.symbols = {"EURUSD": {}}
        self.result = {"ok": True}

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        metadata = kwargs["strategy_metadata"]
        if kwargs["before_run"] is not None:
            metadata = kwargs["before_run"](metadata, ["EURUSD"])
        self.passed = True
        make_run(self.results, kwargs["run_id"], symbols=self.symbols)
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def gated(tmp_path, gate_register, monkeypatch):
    monkeypatch.setenv(GATE_ENV, "on")
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(gate_register))
    results = tmp_path / "results"
    monkeypatch.setattr(storage, "RESULTS_BASE_PATH", str(results))
    # main() enables faulthandler on sys.stderr, which capsys replaces with a stream without fileno.
    monkeypatch.setattr(faulthandler, "enable", lambda *args, **kwargs: None)
    strategies = tmp_path / "strategies"
    (strategies / "pipelines").mkdir(parents=True)
    (strategies / "pipelines" / "cusum.json").write_text(json.dumps(PIPELINE), encoding="utf-8")
    (strategies / "configs").mkdir()
    strategy_file = strategies / "configs" / "cusum.json"
    strategy = {"name": "cusum", "timeframe": "HOUR_1", "pipeline": "cusum", "model": {}}
    strategy_file.write_text(json.dumps(strategy), encoding="utf-8")
    fake = FakeOptimizer(results)
    monkeypatch.setattr(cli, "run_optimizer", fake)
    return SimpleNamespace(register=gate_register, results=results, file=strategy_file, fake=fake)


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["fwbg", *args])
    cli.main()


def trials(register):
    return read_trials(register / LEDGER_FILENAME)


def test_run_without_research_id_is_refused(gated, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exited:
        run_cli(monkeypatch, "--strategy-file", str(gated.file), "--note", "x", "--run-id", "r1")
    assert exited.value.code == 2
    assert "[missing_research_id]" in capsys.readouterr().err
    assert trials(gated.register) == [] and read_pending(gated.register) == []
    assert not gated.fake.passed


def test_gated_run_is_recorded(gated, monkeypatch):
    run_cli(
        monkeypatch,
        *("--strategy-file", str(gated.file), "--research-id", "S-0007"),
        *("--note", "baseline", "--run-id", "r1"),
    )
    [trial] = trials(gated.register)
    assert trial.run_id == "r1" and trial.status is TrialStatus.OK
    [call] = gated.fake.calls
    assert call["run_id"] == "r1" and call["before_run"] is not None
    assert call["strategy_metadata"]["pipeline"] == PIPELINE
    assert read_pending(gated.register) == []


def test_marker_holds_the_absolute_run_dir(gated, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(storage, "RESULTS_BASE_PATH", "results")
    markers = []

    def spying_optimizer(**kwargs):
        gate = kwargs["before_run"]

        def before_run(metadata, symbols):
            effective = gate(metadata, symbols)
            markers.extend(read_pending(gated.register))
            return effective

        return gated.fake(**{**kwargs, "before_run": before_run})

    monkeypatch.setattr(cli, "run_optimizer", spying_optimizer)
    run_cli(
        monkeypatch,
        *("--strategy-file", str(gated.file), "--research-id", "S-0007"),
        *("--note", "baseline", "--run-id", "r1"),
    )
    assert [run.run_dir for run in markers] == [tmp_path / "results" / "r1"]


def test_failed_run_exits_1_and_is_recorded(gated, monkeypatch):
    gated.fake.symbols, gated.fake.result = {}, None
    with pytest.raises(SystemExit) as exited:
        run_cli(
            monkeypatch,
            *("--strategy-file", str(gated.file), "--research-id", "S-0007"),
            *("--note", "baseline", "--run-id", "r1"),
        )
    assert exited.value.code == 1
    [trial] = trials(gated.register)
    assert trial.run_id == "r1" and trial.status is TrialStatus.FAILED
    assert read_pending(gated.register) == []


def test_aborted_strategy_prompt_is_an_error(gated, monkeypatch):
    monkeypatch.setattr(cli, "prompt_strategy_metadata", lambda: None)
    with pytest.raises(SystemExit) as exited:
        run_cli(monkeypatch, "--strategy", "--research-id", "S-0007", "--note", "x")
    assert exited.value.code == 1
    assert gated.fake.calls == []


def test_check_reports_and_writes_nothing(gated, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exited:
        run_cli(
            monkeypatch,
            *("--strategy-file", str(gated.file), "--research-id", "S-0007"),
            *("--note", "baseline", "--run-id", "r1", "--check"),
        )
    assert exited.value.code == 0
    assert "Research gate: allowed" in capsys.readouterr().out
    assert trials(gated.register) == [] and read_pending(gated.register) == []
    assert not gated.fake.passed


def test_check_json_reports_an_allowed_run(gated, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exited:
        run_cli(
            monkeypatch,
            *("--strategy-file", str(gated.file), "--research-id", "S-0007"),
            *("--note", "baseline", "--run-id", "r1", "--check-json"),
        )
    assert exited.value.code == 0
    payload = parse_check_output(capsys.readouterr().out)
    assert payload is not None and payload["allowed"] is True
    assert payload["symbols"] == ["EURUSD"] and payload["window"]["end"] == "2025-04-30"
    assert trials(gated.register) == [] and read_pending(gated.register) == []
    assert not gated.fake.passed


def test_check_json_reports_a_refusal(gated, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exited:
        run_cli(
            monkeypatch,
            *("--strategy-file", str(gated.file), "--note", "x", "--run-id", "r1", "--check-json"),
        )
    assert exited.value.code == 2
    payload = parse_check_output(capsys.readouterr().out)
    assert payload == {
        "allowed": False,
        "denials": [
            {
                "code": "missing_research_id",
                "status": 422,
                "message": "every run needs a research id (S-NNNN)",
            }
        ],
    }
    assert not gated.fake.passed


def test_no_save_is_refused(gated, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exited:
        run_cli(
            monkeypatch,
            *("--strategy-file", str(gated.file), "--research-id", "S-0007"),
            *("--note", "baseline", "--no-save"),
        )
    assert exited.value.code == 2
    assert "--no-save runs cannot be recorded" in capsys.readouterr().err
    assert gated.fake.calls == []


def test_strategy_file_that_fails_to_load_is_an_error(gated, monkeypatch, tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exited:
        run_cli(
            monkeypatch,
            *("--strategy-file", str(broken), "--research-id", "S-0007", "--note", "x"),
        )
    assert exited.value.code == 1
    assert gated.fake.calls == []
    assert trials(gated.register) == [] and read_pending(gated.register) == []


def test_crashed_run_is_recorded_and_reraised(gated, monkeypatch):
    gated.fake.error = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        run_cli(
            monkeypatch,
            *("--strategy-file", str(gated.file), "--research-id", "S-0007"),
            *("--note", "baseline", "--run-id", "r1"),
        )
    [trial] = trials(gated.register)
    assert trial.run_id == "r1" and trial.status is TrialStatus.OK
    assert read_pending(gated.register) == []


def test_gate_off_runs_ungated_with_a_warning(gated, monkeypatch, capsys):
    monkeypatch.setenv(GATE_ENV, "off")
    run_cli(monkeypatch, "--strategy-file", str(gated.file), "--run-id", "r1")
    assert "neither gated nor recorded" in capsys.readouterr().err
    [call] = gated.fake.calls
    assert call["before_run"] is None
    assert trials(gated.register) == []


# --- the real run_optimizer ---------------------------------------------------


@pytest.fixture
def optimizer(tmp_path, monkeypatch):
    """The real ``run_optimizer`` on a data dir with one EURUSD file; globals restored."""
    monkeypatch.setenv(GATE_ENV, "on")
    for name in ("DATA_PATH", "TIMEFRAME", "tf_cfg", "OOS_SIZE", "WINDOW_SIZE", "RESAMPLE_FROM"):
        monkeypatch.setattr(data_config, name, getattr(data_config, name))
    monkeypatch.setattr(progress, "_progress_queue", progress._progress_queue)
    monkeypatch.setenv("TIMEFRAME", "HOUR_1")  # delenv of an unset name would not restore it
    results = tmp_path / "results"
    monkeypatch.setattr(storage, "RESULTS_BASE_PATH", str(results))
    data = tmp_path / "data"
    data.mkdir()
    (data / "EURUSD_HOUR_1.csv").write_text("time,O,H,L,C\n", encoding="utf-8")
    metadata = {"name": "x", "timeframe": "HOUR_1", "_cli_data_path": str(data)}
    return SimpleNamespace(results=results, metadata=metadata)


def test_run_optimizer_refuses_to_run_ungated(optimizer):
    with pytest.raises(RuntimeError, match=GATE_ENV):
        cli.run_optimizer(strategy_metadata=optimizer.metadata, run_id="r1")
    assert not optimizer.results.exists()


def test_run_optimizer_runs_with_the_gated_metadata(optimizer, monkeypatch):
    seen = []
    captured = []

    def before_run(metadata, symbols):
        seen.append(list(symbols))
        return {**metadata, "end_date": "2024-01-01"}

    def capture(self, func, items, progress_callback=None, result_callback=None):
        captured.append(func)
        return []

    monkeypatch.setattr(SimplePoolManager, "map_adaptive", capture)
    cli.run_optimizer(strategy_metadata=optimizer.metadata, run_id="r1", before_run=before_run)
    assert seen == [["EURUSD"]]
    assert captured[0].keywords["strategy"].end_date == "2024-01-01"
    strategy = json.loads((optimizer.results / "r1" / "strategy.json").read_text("utf-8"))
    assert strategy["end_date"] == "2024-01-01"


def test_run_optimizer_writes_nothing_when_the_gate_refuses(optimizer):
    def before_run(metadata, symbols):
        raise GateDenied([Denial(DenialCode.MISSING_NOTE, "no note")])

    with pytest.raises(GateDenied):
        cli.run_optimizer(strategy_metadata=optimizer.metadata, run_id="r1", before_run=before_run)
    assert not optimizer.results.exists()
