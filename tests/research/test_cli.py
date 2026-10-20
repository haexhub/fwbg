import faulthandler
import json
import sys

import pytest
import yaml

from fwbg.cli.main import main as fwbg_main
from fwbg.research.cli import main
from fwbg.research.ledger import LEDGER_FILENAME, read_trials
from fwbg.research.mapping import load_mapping
from tests.research.factories import make_run


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """Point every workspace directory the CLI resolves into *tmp_path*."""
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setenv("FWBG_TEST_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("FWBG_STRATEGIES_DIR", str(tmp_path / "strategies" / "configs"))


def test_init_scan_and_apply_end_to_end(tmp_path, git_env, capsys):
    make_run(tmp_path / "results", "r1", name="orb-test")
    make_run(tmp_path / "results", "r2", name="m3_smoke_x")
    mapping = tmp_path / "mapping.yaml"

    assert main(["init"]) == 0
    assert main(["backfill", "scan", "--out", str(mapping)]) == 0
    out = capsys.readouterr().out
    assert "2 runs: 1 counted, 0 failed, 1 smoke, 0 unresolvable; 2 draft cards" in out
    cards = yaml.safe_load(mapping.read_text(encoding="utf-8"))["cards"]
    assert [card["title"] for card in cards] == ["orb-test", "m3_smoke_x"]

    assert main(["backfill", "apply", str(mapping)]) == 0
    assert "2 cards written, 2 trials appended, committed" in capsys.readouterr().out
    assert len(read_trials(tmp_path / "research" / LEDGER_FILENAME)) == 2


def test_init_twice_reports_an_existing_register(git_env, capsys):
    assert main(["init"]) == 0
    capsys.readouterr()
    assert main(["init"]) == 0
    assert "Already initialised" in capsys.readouterr().out


def test_scan_refuses_to_overwrite_a_mapping_before_scanning(tmp_path, capsys):
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("keep me\n", encoding="utf-8")
    missing = tmp_path / "missing"
    assert main(["backfill", "scan", "--out", str(mapping), "--results", str(missing)]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
    assert mapping.read_text(encoding="utf-8") == "keep me\n"


def test_scan_does_not_create_the_research_dir(tmp_path):
    make_run(tmp_path / "results", "r1")
    assert main(["backfill", "scan", "--out", str(tmp_path / "mapping.yaml")]) == 0
    assert not (tmp_path / "research").exists()


def test_scan_of_a_missing_results_dir_fails_without_creating_the_default(tmp_path, capsys):
    missing = tmp_path / "missing"
    out = tmp_path / "mapping.yaml"
    assert main(["backfill", "scan", "--out", str(out), "--results", str(missing)]) == 1
    assert capsys.readouterr().err.startswith("Error: ")
    assert not (tmp_path / "results").exists()
    assert not out.exists()


def test_draft_prose_keeps_its_line_breaks_when_edited(tmp_path):
    make_run(tmp_path / "results", "20260305_100000")
    mapping = tmp_path / "mapping.yaml"
    assert main(["backfill", "scan", "--out", str(mapping)]) == 0
    last = "    No conclusion documented.\n"
    edited = mapping.read_text(encoding="utf-8").replace(last, last + "    Rejected after v3.\n")

    (plan,) = load_mapping(edited)
    assert plan.body.endswith("No conclusion documented.\nRejected after v3.\n")
    assert plan.runs == ("20260305_100000",)


def test_scan_warns_about_runs_in_progress(tmp_path, capsys):
    running = make_run(tmp_path / "results", "r1")
    finished = make_run(tmp_path / "results", "r2")
    (running / "progress.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    (finished / "progress.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    assert main(["backfill", "scan", "--out", str(tmp_path / "mapping.yaml")]) == 0
    err = capsys.readouterr().err
    assert "may still be in progress" in err
    assert "r1" in err
    assert "r2" not in err


def test_apply_before_init_reports_not_initialised(tmp_path, capsys):
    make_run(tmp_path / "results", "r1")
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("cards: []\n", encoding="utf-8")
    assert main(["backfill", "apply", str(mapping)]) == 1
    assert "research register not initialised" in capsys.readouterr().err
    research = tmp_path / "research"
    assert not research.exists() or not any(research.iterdir())


def test_apply_reports_mapping_errors_without_traceback(tmp_path, git_env, capsys):
    make_run(tmp_path / "results", "r1")
    main(["init"])
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("cards: []\n", encoding="utf-8")
    assert main(["backfill", "apply", str(mapping)]) == 1
    assert "Error: unmapped runs: r1" in capsys.readouterr().err


@pytest.mark.parametrize(
    "content",
    [b"cards:\n  - id: S-0001\n    title: [unclosed\n", b"\xff\xfe not utf-8\n"],
    ids=["malformed-yaml", "not-utf-8"],
)
def test_apply_reports_an_unreadable_mapping(tmp_path, git_env, capsys, content):
    make_run(tmp_path / "results", "r1")
    main(["init"])
    mapping = tmp_path / "mapping.yaml"
    mapping.write_bytes(content)
    assert main(["backfill", "apply", str(mapping)]) == 1
    assert capsys.readouterr().err.startswith("Error: ")


def test_apply_reports_a_missing_mapping(tmp_path, git_env, capsys):
    make_run(tmp_path / "results", "r1")
    main(["init"])
    assert main(["backfill", "apply", str(tmp_path / "missing.yaml")]) == 1
    assert capsys.readouterr().err.startswith("Error: ")


@pytest.mark.parametrize(
    "template",
    ["min_sharpe: 1.0\nmin_sharpe: 9.0\n", "min_sharpe: 2026-02-30\n"],
    ids=["duplicate-key", "impossible-date"],
)
def test_apply_reports_an_unreadable_criteria_template(tmp_path, git_env, capsys, template):
    make_run(tmp_path / "results", "r1")
    main(["init"])
    (tmp_path / "research" / "templates" / "criteria-default.yaml").write_text(
        template, encoding="utf-8"
    )
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("cards:\n- id: S-0001\n  title: orb\n  runs: [r1]\n", encoding="utf-8")
    assert main(["backfill", "apply", str(mapping)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: ") and "criteria-default.yaml" in err
    assert "Traceback" not in err
    assert read_trials(tmp_path / "research" / LEDGER_FILENAME) == []


def test_fwbg_routes_research_commands(monkeypatch, capsys):
    # fwbg's main enables faulthandler on stderr, which needs a real file descriptor.
    monkeypatch.setattr(faulthandler, "enable", lambda: None)
    monkeypatch.setattr(sys, "argv", ["fwbg", "research", "backfill"])
    with pytest.raises(SystemExit) as exit_info:
        fwbg_main()
    assert exit_info.value.code == 2
    assert "fwbg research backfill" in capsys.readouterr().err
