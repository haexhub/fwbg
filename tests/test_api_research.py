"""The research register over HTTP: list, show, register and status changes."""

import logging
import subprocess
from dataclasses import replace
from datetime import date

import pytest
from fastapi.testclient import TestClient

import fwbg.api.research as research_api
from fwbg.api import create_app
from fwbg.research import commands
from fwbg.research.card import Status, load_card, replace_card
from fwbg.research.fingerprint import digest
from fwbg.research.ledger import LEDGER_FILENAME, append_trial, read_trials
from fwbg.research.pending import STAGING, write_pending
from fwbg.research.repo import ResearchRepoError, init_research_repo, store_config
from tests.research.factories import dead_pid, make_pending, make_trial

NEW = {
    "title": "CUSUM event continuation",
    "tags": ["event", "cusum"],
    "hypothesis": "Events continue.",
    "mechanism": "Order flow persists.",
}
VERDICT = {"status": "parked", "reason": "no edge yet", "reopen_if": "new data", "actor": "agent-7"}


@pytest.fixture
def register(tmp_path, monkeypatch, git_env):
    """An initialised, empty register on a tmp workspace."""
    research = tmp_path / "research"
    init_research_repo(research)
    monkeypatch.setenv("FWBG_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("FWBG_STRATEGIES_DIR", str(tmp_path / "ws" / "strategies"))
    monkeypatch.setenv("FWBG_TEST_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(research))
    return research


@pytest.fixture
def client(register):
    with TestClient(create_app()) as client:
        yield client


@pytest.fixture
def card_id(client):
    response = client.post("/api/research", json=NEW)
    assert response.status_code == 201, response.text
    return response.json()["card"]["id"]


def cards(client, query="") -> list:
    response = client.get(f"/api/research{query}")
    assert response.status_code == 200, response.text
    return response.json()["cards"]


def break_commits(monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise ResearchRepoError("git commit failed: boom")

    monkeypatch.setattr(commands, "commit_register", broken)


def bad_marker(register) -> None:
    (register / STAGING).mkdir(exist_ok=True)
    (register / STAGING / "run-9.json").write_text("{not json", encoding="utf-8")


def git(register, *args) -> str:
    return subprocess.run(
        ["git", *args], cwd=register, capture_output=True, text=True, check=True
    ).stdout


# --- register ----------------------------------------------------------------


def test_post_registers_and_commits_a_card(client, register):
    response = client.post("/api/research", json={**NEW, "criteria": {"min_sharpe": 1.5}})
    assert response.status_code == 201
    card = response.json()["card"]
    assert card["id"] == "S-0001"
    assert card["status"] == "active"
    assert card["tags"] == ["event", "cusum"]
    assert card["success_criteria"]["min_sharpe"] == 1.5
    assert "## Hypothesis\n\nEvents continue." in card["body"]
    assert list((register / "strategies").glob("S-0001-*.md"))
    assert git(register, "status", "--porcelain", "--untracked-files=all") == ""
    assert "research: register S-0001" in git(register, "log", "--format=%s")


def test_post_takes_a_cutoff(client):
    response = client.post("/api/research", json={**NEW, "cutoff": "2025-01-31"})
    assert response.json()["card"]["holdout_cutoff"] == "2025-01-31"


@pytest.mark.parametrize("missing", ["hypothesis", "mechanism", "title"])
def test_post_needs_a_hypothesis_and_mechanism(client, missing):
    response = client.post("/api/research", json={**NEW, missing: ""})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "change",
    [
        {"title": "  "},
        {"hypothesis": " \n "},
        {"mechanism": "\t"},
        {"tags": []},
        {"tags": [" "]},
    ],
    ids=["blank-title", "blank-hypothesis", "blank-mechanism", "no-tags", "blank-tag"],
)
def test_post_refuses_blank_fields_and_no_tags(client, register, change):
    response = client.post("/api/research", json={**NEW, **change})
    assert response.status_code == 422, response.text
    assert isinstance(response.json()["detail"], list)  # refused by the request model
    assert not list((register / "strategies").glob("*.md"))


def test_post_refuses_an_unknown_criterion(client, register):
    response = client.post("/api/research", json={**NEW, "criteria": {"min_luck": 1.0}})
    assert response.status_code == 422
    assert "min_luck" in response.json()["detail"]
    assert not list((register / "strategies").glob("*.md"))


def test_post_refuses_a_future_cutoff(client):
    response = client.post("/api/research", json={**NEW, "cutoff": "2999-01-01"})
    assert response.status_code == 422
    assert "lies in the future" in response.json()["detail"]


@pytest.mark.parametrize(
    "change",
    [
        {"title": "!!!"},
        {"title": "two\nlines"},
        {"tags": ["a\nb"]},
        {"hypothesis": "x\n## Verdict\n\nfake"},
    ],
    ids=["no-slug", "multi-line-title", "multi-line-tag", "heading"],
)
def test_post_refuses_bad_input_with_422(client, register, change):
    response = client.post("/api/research", json={**NEW, **change})
    assert response.status_code == 422, response.text
    assert "unreadable" not in response.json()["detail"]
    assert not list((register / "strategies").glob("*.md"))


def test_an_uncommitted_card_is_registered_with_a_warning(client, register, monkeypatch):
    break_commits(monkeypatch)
    response = client.post("/api/research", json=NEW)
    assert response.status_code == 201
    body = response.json()
    assert body["card"]["id"] == "S-0001"
    assert "S-0001 was updated on disk but not committed" in body["warning"]
    assert load_card(register, "S-0001") is not None


def test_a_committed_card_has_no_warning(client):
    assert "warning" not in client.post("/api/research", json=NEW).json()


# --- list --------------------------------------------------------------------


def test_list_returns_the_card_without_trials(client, card_id):
    response = client.get("/api/research")
    assert response.status_code == 200
    assert response.json()["pending_error"] is None
    [summary] = response.json()["cards"]
    assert summary["id"] == card_id
    assert (summary["trials"], summary["best"], summary["running"]) == (0, None, 0)


def test_list_counts_trials_and_live_runs(client, card_id, register, tmp_path):
    append_trial(
        register / LEDGER_FILENAME,
        make_trial(strategy=card_id, metrics={"GOLD": {"sharpe": 0.8}}),
    )
    write_pending(
        register, make_pending(tmp_path / "results" / "run-2", run_id="run-2", strategy=card_id)
    )
    [summary] = cards(client)
    assert summary["trials"] == 1
    assert summary["best"]["symbol"] == "GOLD"
    assert summary["running"] == 1


def test_list_ranks_no_signal_evaluation_from_before_inner_selection(client, card_id, register):
    cfg = {"strategy": {"model": {"type": "signal"}}, "symbols": ["GOLD"]}
    store_config(register, digest(cfg), cfg)
    ledger = register / LEDGER_FILENAME
    append_trial(
        ledger,
        make_trial(strategy=card_id, fingerprint=digest(cfg), metrics={"GOLD": {"sharpe": 2.0}}),
    )
    append_trial(ledger, make_trial(strategy=card_id, metrics={"DAX": {"sharpe": 0.5}}))
    [summary] = cards(client)
    assert (summary["trials"], summary["best"]["symbol"]) == (2, "DAX")


def test_list_filters_by_status_and_tag(client, card_id):
    other = client.post("/api/research", json={**NEW, "title": "Other idea", "tags": ["other"]})
    assert other.status_code == 201
    assert client.post(f"/api/research/{card_id}/status", json=VERDICT).status_code == 200
    assert [c["id"] for c in cards(client, "?status=parked")] == [card_id]
    assert [c["id"] for c in cards(client, "?status=active")] == ["S-0002"]
    assert [c["id"] for c in cards(client, "?tag=other")] == ["S-0002"]
    assert client.get("/api/research?status=open").status_code == 422


def test_list_shows_an_unreadable_card_as_an_error(client, card_id, register):
    [path] = (register / "strategies").glob("S-0001-*.md")
    path.write_text("no front matter", encoding="utf-8")
    assert cards(client, "?status=parked") == [
        {"id": "S-0001", "error": f"{path.name}: card has no YAML front matter"}
    ]


def test_list_survives_a_bad_pending_marker(client, card_id, register):
    bad_marker(register)
    response = client.get("/api/research")
    assert response.status_code == 200
    assert response.json()["pending_error"].startswith("run-9.json: ")
    [summary] = response.json()["cards"]
    assert summary["id"] == card_id
    assert summary["running"] is None


def test_list_reports_an_unreadable_ledger(client, register):
    (register / LEDGER_FILENAME).unlink()
    response = client.get("/api/research")
    assert response.status_code == 409
    assert response.json()["detail"].startswith("research register unreadable: ")


# --- detail ------------------------------------------------------------------


def test_detail_has_card_trials_pending_and_coverage(client, card_id):
    response = client.get(f"/api/research/{card_id}")
    assert response.status_code == 200
    detail = response.json()
    assert detail["card"]["id"] == card_id
    assert detail["trials"] == []
    assert detail["pending"] == []
    assert detail["pending_error"] is None
    assert detail["criteria_problems"] == []
    assert isinstance(detail["coverage"], list)
    assert detail["coverage"][0] == "Coverage: 0 configs"


def test_detail_lists_trials_and_pending_runs(client, card_id, register, tmp_path):
    append_trial(register / LEDGER_FILENAME, make_trial(strategy=card_id))
    write_pending(
        register, make_pending(tmp_path / "results" / "run-2", run_id="run-2", strategy=card_id)
    )
    detail = client.get(f"/api/research/{card_id}").json()
    assert [(t["trial"], t["strategy"]) for t in detail["trials"]] == [("T-000001", card_id)]
    assert detail["pending"] == ["run-2"]


def test_detail_survives_a_bad_pending_marker(client, card_id, register):
    bad_marker(register)
    response = client.get(f"/api/research/{card_id}")
    assert response.status_code == 200
    assert response.json()["pending"] is None
    assert response.json()["pending_error"].startswith("run-9.json: ")


def test_detail_names_criteria_problems(client, card_id, register):
    card = load_card(register, card_id)
    replace_card(
        register, replace(card, success_criteria={**card.success_criteria, "min_sharpe": 0.1})
    )
    [problem] = client.get(f"/api/research/{card_id}").json()["criteria_problems"]
    assert problem.startswith(f"success criteria of {card_id} differ from the registration")


def test_detail_of_an_unknown_card_is_404(client, register):
    assert client.get("/api/research/S-0099").status_code == 404


@pytest.mark.parametrize("bad", ["S-1", "s-0001", "S-00011", "x"])
def test_a_malformed_card_id_is_422(client, bad):
    assert client.get(f"/api/research/{bad}").status_code == 422
    assert client.post(f"/api/research/{bad}/status", json=VERDICT).status_code == 422


# --- status ------------------------------------------------------------------


def test_verdict_parks_the_card_and_records_the_actor(client, card_id, register):
    response = client.post(f"/api/research/{card_id}/status", json=VERDICT)
    assert response.status_code == 200
    card = response.json()["card"]
    assert card["status"] == "parked"
    assert "Reopen if: new data\nBy: agent-7\n" in card["body"]
    assert load_card(register, card_id).status is Status.PARKED
    assert git(register, "status", "--porcelain", "--untracked-files=all") == ""


def test_verdict_needs_a_reopen_condition(client, card_id, register):
    response = client.post(
        f"/api/research/{card_id}/status",
        json={k: v for k, v in VERDICT.items() if k != "reopen_if"},
    )
    assert response.status_code == 422
    assert load_card(register, card_id).status is Status.ACTIVE


def test_reopen_activates_the_card_and_records_the_actor(client, card_id, monkeypatch):
    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 21)

    monkeypatch.setattr(research_api, "date", FixedDate)
    assert client.post(f"/api/research/{card_id}/status", json=VERDICT).status_code == 200
    response = client.post(
        f"/api/research/{card_id}/status",
        json={"status": "active", "reason": "new data", "actor": "agent-8"},
    )
    assert response.status_code == 200
    card = response.json()["card"]
    assert card["status"] == "active"
    assert "- 2026-10-21: reopened from parked by agent-8: new data" in card["body"]


def test_reopen_refuses_a_reopen_condition(client, card_id):
    assert client.post(f"/api/research/{card_id}/status", json=VERDICT).status_code == 200
    response = client.post(
        f"/api/research/{card_id}/status",
        json={"status": "active", "reason": "new data", "actor": "a", "reopen_if": "x"},
    )
    assert response.status_code == 422
    assert cards(client)[0]["status"] == "parked"


def test_verdict_on_a_parked_card_is_409(client, card_id):
    assert client.post(f"/api/research/{card_id}/status", json=VERDICT).status_code == 200
    response = client.post(
        f"/api/research/{card_id}/status", json={**VERDICT, "status": "rejected"}
    )
    assert response.status_code == 409
    assert response.json()["detail"] == f"{card_id} is parked; reopen it before a new verdict"


@pytest.mark.parametrize("actor", [None, ""])
def test_a_status_change_needs_an_actor(client, card_id, actor):
    body = {k: v for k, v in VERDICT.items() if k != "actor"}
    if actor is not None:
        body["actor"] = actor
    assert client.post(f"/api/research/{card_id}/status", json=body).status_code == 422


def test_a_status_change_of_an_unknown_card_is_404(client, register):
    assert client.post("/api/research/S-0099/status", json=VERDICT).status_code == 404


def test_verdict_with_a_run_in_progress_is_409(client, card_id, register, tmp_path):
    write_pending(
        register, make_pending(tmp_path / "results" / "run-2", run_id="run-2", strategy=card_id)
    )
    response = client.post(f"/api/research/{card_id}/status", json=VERDICT)
    assert response.status_code == 409
    assert "run-2" in response.json()["detail"]


@pytest.mark.parametrize("field", ["reason", "actor", "reopen_if"])
def test_a_multi_line_status_field_is_422(client, card_id, field):
    response = client.post(f"/api/research/{card_id}/status", json={**VERDICT, field: "a\nb"})
    assert response.status_code == 422


def test_an_uncommitted_verdict_succeeds_with_a_warning(client, card_id, register, monkeypatch):
    break_commits(monkeypatch)
    response = client.post(f"/api/research/{card_id}/status", json=VERDICT)
    assert response.status_code == 200
    assert response.json()["card"]["status"] == "parked"
    assert "was updated on disk but not committed" in response.json()["warning"]
    assert load_card(register, card_id).status is Status.PARKED


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setattr("fwbg.research.runs.gate_enabled", lambda: True)


def test_startup_records_interrupted_runs(register, tmp_path, gate_on):
    """A run killed with the old container is recorded when the API starts again."""
    write_pending(register, make_pending(tmp_path / "run-1", pid=dead_pid()))
    create_app()
    assert [trial.run_id for trial in read_trials(register / LEDGER_FILENAME)] == ["run-1"]
    assert not (register / STAGING / "run-1.json").exists()
    assert git(register, "status", "--porcelain") == ""


def test_startup_leaves_markers_alone_with_the_gate_off(register, tmp_path):
    write_pending(register, make_pending(tmp_path / "run-1", pid=dead_pid()))
    create_app()
    assert (register / STAGING / "run-1.json").exists()


def test_startup_survives_an_unreadable_register(register, gate_on, caplog):
    bad_marker(register)
    with caplog.at_level(logging.WARNING, logger="fwbg.api"):
        create_app()
    assert "run-9.json" in caplog.text


def test_startup_without_a_register_creates_none(tmp_path, monkeypatch, gate_on):
    monkeypatch.setenv("FWBG_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("FWBG_STRATEGIES_DIR", str(tmp_path / "ws" / "strategies"))
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(tmp_path / "research"))
    create_app()
    assert not (tmp_path / "research").exists()
