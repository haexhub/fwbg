import yaml

from fwbg.research.cli import main
from fwbg.research.ledger import LEDGER_FILENAME, read_trials
from tests.research.factories import make_run


def test_init_scan_and_apply_end_to_end(tmp_path, monkeypatch, git_env, capsys):
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setenv("FWBG_TEST_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("FWBG_STRATEGIES_DIR", str(tmp_path / "strategies" / "configs"))
    make_run(tmp_path / "results", "r1", name="orb-test")
    make_run(tmp_path / "results", "r2", name="m3_smoke_x")
    mapping = tmp_path / "mapping.yaml"

    assert main(["init"]) == 0
    assert main(["backfill", "scan", "--out", str(mapping)]) == 0
    assert "2 runs (1 counted, 0 unresolvable)" in capsys.readouterr().out
    cards = yaml.safe_load(mapping.read_text(encoding="utf-8"))["cards"]
    assert [card["title"] for card in cards] == ["orb-test", "m3_smoke_x"]

    assert main(["backfill", "apply", str(mapping)]) == 0
    assert "2 cards written, 2 trials appended, committed" in capsys.readouterr().out
    assert len(read_trials(tmp_path / "research" / LEDGER_FILENAME)) == 2


def test_scan_refuses_to_overwrite_a_mapping(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setenv("FWBG_TEST_RESULTS_DIR", str(tmp_path / "results"))
    make_run(tmp_path / "results", "r1")
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("keep me\n", encoding="utf-8")
    assert main(["backfill", "scan", "--out", str(mapping)]) == 1
    assert mapping.read_text(encoding="utf-8") == "keep me\n"


def test_apply_reports_mapping_errors_without_traceback(tmp_path, monkeypatch, git_env, capsys):
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setenv("FWBG_TEST_RESULTS_DIR", str(tmp_path / "results"))
    make_run(tmp_path / "results", "r1")
    main(["init"])
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("cards: []\n", encoding="utf-8")
    assert main(["backfill", "apply", str(mapping)]) == 1
    assert "Error: unmapped runs: r1" in capsys.readouterr().out
