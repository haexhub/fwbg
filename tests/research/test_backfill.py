import subprocess
from dataclasses import replace
from datetime import date

import pytest

from fwbg.research.backfill import BackfillError, apply_mapping, draft_mapping, parse_mapping
from fwbg.research.card import Status, load_card
from fwbg.research.gate import DenialCode, RunRequest, check
from fwbg.research.ledger import LEDGER_FILENAME, TrialKind, read_trials
from fwbg.research.legacy import scan_runs
from fwbg.research.repo import init_research_repo
from tests.research.factories import make_run

CRITERIA = {"min_sharpe": 1.0, "min_trades": 100.0}
TODAY = date(2026, 10, 20)


@pytest.fixture
def register(tmp_path, git_env):
    research = tmp_path / "research"
    init_research_repo(research)
    return research


@pytest.fixture
def runs(tmp_path):
    results = tmp_path / "results"
    make_run(results, "r1", name="orb-test", timestamp="2026-03-05T10:00:00")
    make_run(results, "r2", name="orb-test", timestamp="2026-03-06T09:00:00")
    make_run(results, "r3", name="pdh retest", timestamp="2026-02-22T08:00:00")
    return scan_runs(results, tmp_path / "presets")


def mapping_for(runs, **card_overrides):
    data = draft_mapping(runs)
    for card in data["cards"]:
        card.update(card_overrides.get(card["id"], {}))
    return parse_mapping(data)


def git_log(path):
    return subprocess.run(
        ["git", "log", "--format=%s"], cwd=path, capture_output=True, text=True, check=True
    ).stdout.splitlines()


def test_draft_proposes_one_parked_card_per_name_in_order_of_first_run(runs):
    cards = draft_mapping(runs)["cards"]
    assert [(c["id"], c["title"], c["runs"]) for c in cards] == [
        ("S-0001", "pdh retest", ["r3"]),
        ("S-0002", "orb-test", ["r1", "r2"]),
    ]
    assert {c["status"] for c in cards} == {"parked"}
    assert cards[0]["slug"] == "pdh-retest"


def test_apply_writes_cards_trials_configs_and_one_commit(runs, register):
    result = apply_mapping(mapping_for(runs), runs, register, CRITERIA, TODAY)
    assert (result.cards_written, result.trials_appended, result.committed) == (2, 3, True)
    card = load_card(register, "S-0002")
    assert card.status is Status.PARKED
    assert card.holdout_cutoff == date(2026, 3, 7)
    assert card.holdout_contaminated is False
    assert card.created == TODAY
    trials = read_trials(register / LEDGER_FILENAME)
    assert [t.run_id for t in trials] == ["r3", "r1", "r2"]
    assert {t.kind for t in trials} == {TrialKind.LEGACY}
    assert all(t.holdout_cutoff == load_card(register, t.strategy).holdout_cutoff for t in trials)
    assert all(t.criteria_hash == card.criteria_hash for t in trials)
    # Names are not part of the fingerprint: all three runs share one stored config.
    assert len(list((register / "configs").glob("*.json"))) == 1
    assert git_log(register)[0].startswith("chore: backfill legacy research (2 cards, 3 trials)")


def test_applying_again_adds_nothing(runs, register):
    plans = mapping_for(runs)
    apply_mapping(plans, runs, register, CRITERIA, TODAY)
    result = apply_mapping(plans, runs, register, CRITERIA, date(2026, 10, 21))
    assert (result.cards_written, result.trials_appended, result.committed) == (0, 0, False)
    assert len(read_trials(register / LEDGER_FILENAME)) == 3


def test_new_runs_are_added_to_existing_cards_only_if_the_cutoff_holds(tmp_path, runs, register):
    apply_mapping(mapping_for(runs), runs, register, CRITERIA, TODAY)
    make_run(tmp_path / "results", "r4", name="orb-test", timestamp="2026-04-01T10:00:00")
    more = scan_runs(tmp_path / "results", tmp_path / "presets")
    with pytest.raises(BackfillError, match="S-0002 exists with a different cutoff"):
        apply_mapping(mapping_for(more), more, register, CRITERIA, TODAY)


def test_every_scanned_run_must_be_mapped_exactly_once(runs, register):
    data = draft_mapping(runs)
    data["cards"][1]["runs"] = ["r1", "r1", "ghost"]
    with pytest.raises(BackfillError) as excinfo:
        apply_mapping(parse_mapping(data), runs, register, CRITERIA, TODAY)
    message = str(excinfo.value)
    assert "unknown runs: ghost" in message
    assert "mapped more than once: r1" in message
    assert "unmapped runs: r2" in message
    assert read_trials(register / LEDGER_FILENAME) == []


def test_criteria_override_and_rejected_status(runs, register):
    plans = mapping_for(
        runs, **{"S-0001": {"status": "rejected", "criteria": {"max_drawdown": 0.6}}}
    )
    apply_mapping(plans, runs, register, CRITERIA, TODAY)
    card = load_card(register, "S-0001")
    assert card.status is Status.REJECTED
    assert dict(card.success_criteria) == {"max_drawdown": 0.6}


def test_study_trials_are_recorded_once_with_a_synthetic_fingerprint(runs, register):
    study = {
        "variant": "config-4",
        "symbols": ["eurusd"],
        "date": date(2026, 9, 25),
        "note": "narrative state, alternated key order",
        "metrics": {"eurusd": {"accuracy": 0.29}},
    }
    plans = mapping_for(runs, **{"S-0001": {"studies": [study]}})
    apply_mapping(plans, runs, register, CRITERIA, TODAY)
    apply_mapping(plans, runs, register, CRITERIA, TODAY)
    trials = [t for t in read_trials(register / LEDGER_FILENAME) if t.run_id.startswith("study:")]
    assert [t.run_id for t in trials] == ["study:S-0001:config-4"]
    assert trials[0].metrics == {"EURUSD": {"accuracy": 0.29}}
    assert load_card(register, "S-0001").holdout_cutoff == date(2026, 9, 26)


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"id": "X-1"}, "invalid id"),
        ({"status": "active"}, "parked or rejected"),
        ({"slug": "Bad Slug"}, "invalid slug"),
        ({"runs": [], "studies": []}, "at least one run"),
        ({"criteria": {"min_sharpe": "high"}}, "must be a number"),
        ({"extra": 1}, "unknown keys: extra"),
        ({"studies": [{"variant": "v", "symbols": ["A"], "date": "soon", "note": "n"}]}, "date"),
        (
            {
                "studies": [
                    {"variant": "v", "symbols": ["A"], "date": date(2026, 1, 1), "note": "a\nb"}
                ]
            },
            "one line",
        ),
        (
            {
                "studies": [
                    {
                        "variant": "v",
                        "symbols": ["A"],
                        "date": date(2026, 1, 1),
                        "note": "n",
                        "metrics": {"A": {"x": float("nan")}},
                    }
                ]
            },
            "finite",
        ),
    ],
    ids=["id", "status", "slug", "empty", "criteria", "unknown", "date", "note", "nan"],
)
def test_parse_mapping_rejects_invalid_cards(runs, change, fragment):
    data = draft_mapping(runs)
    data["cards"][0].update(change)
    with pytest.raises(BackfillError, match=fragment):
        parse_mapping(data)


def test_duplicate_card_ids_are_rejected(runs):
    data = draft_mapping(runs)
    data["cards"][1]["id"] = "S-0001"
    with pytest.raises(BackfillError, match="duplicate card ids: S-0001"):
        parse_mapping(data)


def test_gate_recognises_a_backfilled_config_with_the_same_window(tmp_path, git_env):
    results = tmp_path / "results"
    make_run(
        results, "r1", timestamp="2026-03-05T10:00:00", strategy_extra={"end_date": "2026-03-01"}
    )
    runs = scan_runs(results, tmp_path / "presets")
    research = tmp_path / "research"
    init_research_repo(research)
    apply_mapping(parse_mapping(draft_mapping(runs)), runs, research, CRITERIA, TODAY)
    card = replace(load_card(research, "S-0001"), status=Status.ACTIVE)
    request = RunRequest(
        research_id="S-0001",
        strategy_metadata=dict(runs[0].config["strategy"]),
        symbols=runs[0].symbols,
        note="same config, same window",
    )
    decision = check(card, request, read_trials(research / LEDGER_FILENAME))
    assert {d.code for d in decision.denials} == {DenialCode.DUPLICATE_FINGERPRINT}


def test_open_ended_legacy_runs_differ_from_clamped_iterations(runs, register):
    # A legacy run without end_date saw data up to its run date; a new iteration is
    # clamped to the cutoff, so it evaluates a different window and is not a duplicate.
    apply_mapping(mapping_for(runs), runs, register, CRITERIA, TODAY)
    run = next(r for r in runs if r.run_id == "r1")
    card = replace(load_card(register, "S-0002"), status=Status.ACTIVE)
    request = RunRequest(
        research_id="S-0002",
        strategy_metadata=dict(run.config["strategy"]),
        symbols=run.symbols,
        note="clamped re-run",
    )
    assert check(card, request, read_trials(register / LEDGER_FILENAME)).allowed
