import copy
import json
import os
import subprocess
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pytest
import yaml

from fwbg.research.backfill import (
    BackfillError,
    apply_mapping,
    draft_mapping,
    load_mapping,
    parse_mapping,
)
from fwbg.research.card import (
    ResearchCard,
    Status,
    criteria_hash,
    load_card,
    parse_card,
    write_card,
)
from fwbg.research.gate import DenialCode, RunRequest, check
from fwbg.research.ledger import LEDGER_FILENAME, TrialKind, TrialStatus, read_trials
from fwbg.research.legacy import scan_runs
from fwbg.research.repo import ResearchRepoError, commit_register, init_research_repo
from tests.research.factories import make_card, make_run

CRITERIA = {"min_sharpe": 1.0, "min_trades": 100.0}
TODAY = date(2026, 10, 20)
STUDY = {
    "variant": "config-4",
    "symbols": ["eurusd"],
    "date": date(2026, 2, 1),
    "note": "narrative state",
}


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


def git_status(path):
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def register_files(path):
    """Every file of the register outside ``.git``, with its content."""
    return {
        file.relative_to(path).as_posix(): file.read_bytes()
        for file in sorted(path.rglob("*"))
        if file.is_file() and ".git" not in file.relative_to(path).parts
    }


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
    assert git_log(register)[0] == "chore: backfill legacy research"


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
    assert dict(card.success_criteria) == {**CRITERIA, "max_drawdown": 0.6}
    assert dict(load_card(register, "S-0002").success_criteria) == CRITERIA


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
        ({"criteria": {"min_sharpe": float("inf")}}, "finite"),
        ({"criteria": {"min_trades": 10**400}}, "out of range"),
        ({"studies": [{**STUDY, "date": datetime(2026, 2, 1, 10)}]}, r"studies\[1\]\.date"),
        ({"studies": [{**STUDY, "note": "   "}]}, "one line"),
        ({"studies": [{**STUDY, "note": "trailing break\n"}]}, "one line"),
        ({"studies": [{**STUDY, "metrics": {"a": {"x": 1}, "A": {"y": 2}}}]}, "ignoring case"),
        ({"tags": ["ok", 3]}, r"S-0001\.tags: .*\['ok', 3\]"),
        ({"studies": [STUDY, {**STUDY, "note": "copy"}]}, "duplicate study variants: config-4"),
    ],
    ids=[
        "id",
        "status",
        "slug",
        "empty",
        "criteria",
        "unknown",
        "date",
        "note",
        "nan",
        "inf-criterion",
        "huge-criterion",
        "datetime",
        "blank-note",
        "note-break",
        "symbol-case",
        "tags",
        "variant-twice",
    ],
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


@pytest.mark.parametrize(
    "spoil",
    [
        lambda cards: cards[1].update(criteria={"max_drawdown": float("inf")}),
        lambda cards: cards[1].update(criteria={"min_trades": 10**400}),
        lambda cards: cards[1].update(studies=[{**STUDY, "date": datetime(2026, 3, 1, 10)}]),
        # A card file is read with newline translation, so a carriage return cannot survive.
        lambda cards: cards[1].update(body="## Verdict\r\n"),
        lambda cards: cards.append(
            {
                "id": "S-0003",
                "title": "study only",
                "studies": [{**STUDY, "date": datetime(2026, 3, 1, 10)}],
            }
        ),
    ],
    ids=[
        "inf-criterion",
        "huge-criterion",
        "datetime-study",
        "carriage-return",
        "study-only-datetime",
    ],
)
def test_a_defect_in_a_later_card_leaves_the_register_untouched(runs, register, spoil):
    data = draft_mapping(runs)
    spoil(data["cards"])
    before = register_files(register)
    with pytest.raises(BackfillError):
        apply_mapping(parse_mapping(data), runs, register, CRITERIA, TODAY)
    assert register_files(register) == before
    assert git_status(register) == ""


def test_a_trial_the_ledger_cannot_store_is_refused_before_writing(runs, register):
    # json.loads accepts NaN, so a run's config.json can carry one into the trial.
    spoiled = [replace(r, code={"dirty": float("nan")}) if r.run_id == "r2" else r for r in runs]
    before = register_files(register)
    with pytest.raises(BackfillError, match="r2"):
        apply_mapping(mapping_for(spoiled), spoiled, register, CRITERIA, TODAY)
    assert register_files(register) == before


def test_a_conflicting_existing_card_is_found_before_writing(runs, register):
    existing = make_card(id="S-0002", status=Status.PARKED, holdout_cutoff=date(2026, 1, 1))
    write_card(register, existing, "orb-test")
    commit_register(register, ["strategies"], "test: existing card")
    before = register_files(register)
    with pytest.raises(BackfillError, match="S-0002 exists with a different cutoff"):
        apply_mapping(mapping_for(runs), runs, register, CRITERIA, TODAY)
    assert register_files(register) == before
    assert git_status(register) == ""


def test_existing_cards_keep_their_criteria(runs, register):
    apply_mapping(mapping_for(runs), runs, register, CRITERIA, TODAY)
    with pytest.raises(BackfillError, match="S-0001 exists with a different cutoff or criteria"):
        apply_mapping(mapping_for(runs), runs, register, {**CRITERIA, "min_sharpe": 1.5}, TODAY)


def test_dates_after_today_are_refused(runs, register):
    future_study = mapping_for(
        runs, **{"S-0001": {"studies": [{**STUDY, "date": date(2062, 9, 25)}]}}
    )
    with pytest.raises(BackfillError, match=r"S-0001: dated after today .*study:S-0001:config-4"):
        apply_mapping(future_study, runs, register, CRITERIA, TODAY)
    with pytest.raises(BackfillError, match=r"S-0002: dated after today \(2026-03-05\): r2$"):
        apply_mapping(mapping_for(runs), runs, register, CRITERIA, date(2026, 3, 5))
    assert read_trials(register / LEDGER_FILENAME) == []


def test_load_mapping_reads_a_yaml_draft(runs):
    text = yaml.safe_dump(draft_mapping(runs), sort_keys=False, allow_unicode=True)
    assert load_mapping(text) == parse_mapping(draft_mapping(runs))


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (
            "cards:\n"
            "- id: S-0001\n"
            "  title: jev\n"
            "  studies:\n"
            "  - {variant: a, symbols: [EURUSD], date: 2026-09-25, note: first}\n"
            "  studies:\n"
            "  - {variant: b, symbols: [EURUSD], date: 2026-09-26, note: second}\n",
            "line 6, column 3: duplicate key 'studies'",
        ),
        ("cards:\n- {id: S-0001, title: x, studies: [{date: 2026-02-30}]}\n", "out of range"),
        ("cards: [\n", "line 2, column 1"),
    ],
    ids=["duplicate-key", "impossible-date", "syntax"],
)
def test_load_mapping_reports_yaml_errors(text, fragment):
    with pytest.raises(BackfillError, match=fragment):
        load_mapping(text)


def test_trials_record_what_each_run_left(tmp_path, register):
    workspace = tmp_path / "workspace"
    (workspace / "configs").mkdir(parents=True)
    (workspace / "pipelines").mkdir()
    (workspace / "pipelines" / "orb_simple_v1.json").write_text(
        json.dumps({"indicators": [{"name": "opening_range"}]}), encoding="utf-8"
    )
    results = tmp_path / "results"
    moved = {"pipeline": "orb_simple_v1", "_strategy_dir": "/nonexistent/configs"}
    make_run(
        results,
        "ok",
        name="pdh retest",
        timestamp="2026-02-01T08:00:00",
        symbols={"GOLD": {"unified": {"sharpe": 0.5, "trades": 40}}},
        strategy_extra={**moved, "end_date": "2026-01-31"},
    )
    make_run(
        results,
        "failed",
        name="pdh retest",
        timestamp="2026-02-02T08:00:00",
        symbols={"GOLD": None},
    )
    make_run(results, "smoke", name="m3_smoke_1", timestamp="2026-02-03T08:00:00")
    make_run(
        results,
        "unres",
        name="pdh retest",
        timestamp="2026-02-04T08:00:00",
        strategy_extra={**moved, "pipeline": "missing_preset"},
    )
    runs = scan_runs(results, workspace)
    apply_mapping(mapping_for(runs), runs, register, CRITERIA, TODAY)

    trials = {t.run_id: t for t in read_trials(register / LEDGER_FILENAME)}
    assert list(trials) == ["ok", "failed", "unres", "smoke"]
    contents = {
        run_id: (t.strategy, t.kind, t.status, t.counted, t.ts, dict(t.window), dict(t.code))
        for run_id, t in trials.items()
    }
    code = {"commit": "abc123", "branch": "main", "dirty": True}
    open_window = {"start": None, "end": None}
    assert contents == {
        "ok": (
            "S-0001",
            TrialKind.LEGACY,
            TrialStatus.OK,
            True,
            "2026-02-01T08:00:00",
            {"start": None, "end": "2026-01-31"},
            code,
        ),
        "failed": (
            "S-0001",
            TrialKind.LEGACY,
            TrialStatus.FAILED,
            False,
            "2026-02-02T08:00:00",
            open_window,
            code,
        ),
        "unres": (
            "S-0001",
            TrialKind.LEGACY,
            TrialStatus.OK,
            True,
            "2026-02-04T08:00:00",
            open_window,
            code,
        ),
        "smoke": (
            "S-0002",
            TrialKind.SMOKE,
            TrialStatus.OK,
            False,
            "2026-02-03T08:00:00",
            open_window,
            code,
        ),
    }
    assert trials["ok"].note == "legacy run: pdh retest (presets resolved from the workspace)"
    assert trials["failed"].note == "legacy run: pdh retest"
    assert trials["unres"].note.startswith("legacy run: pdh retest (configuration unresolvable: ")
    assert "missing_preset" in trials["unres"].note
    assert trials["smoke"].note == "legacy run: m3_smoke_1"
    assert trials["ok"].metrics == {
        "GOLD": {"sharpe": 0.5, "trades": 40, "p_value": None, "fold_stability": None}
    }
    assert trials["failed"].metrics == {}
    for run in runs:
        assert trials[run.run_id].fingerprint == run.fingerprint

    stored = {path.stem for path in (register / "configs").glob("*.json")}
    unresolvable = next(run for run in runs if run.run_id == "unres")
    assert unresolvable.fingerprint.removeprefix("sha256:") not in stored
    assert stored == {
        run.fingerprint.removeprefix("sha256:") for run in runs if run.config is not None
    }

    strategies = register / "strategies"
    assert sorted(p.name for p in strategies.iterdir()) == [
        "S-0001-pdh-retest.md",
        "S-0002-m3-smoke-1.md",
    ]
    card = parse_card((strategies / "S-0001-pdh-retest.md").read_text(encoding="utf-8"))
    assert card == ResearchCard(
        id="S-0001",
        title="pdh retest",
        status=Status.PARKED,
        tags=(),
        created=TODAY,
        holdout_cutoff=date(2026, 2, 5),
        holdout_contaminated=False,
        success_criteria=CRITERIA,
        registered_criteria_hash=criteria_hash(CRITERIA),
        criteria_amendments=(),
        body="## Hypothesis\n\n## Verdict\n\nNo conclusion documented.\n",
    )


def test_a_failed_commit_is_completed_by_applying_the_same_mapping_again(tmp_path, runs, register):
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "pre-commit").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    (hooks / "pre-commit").chmod(0o755)
    git_config = Path(os.environ["GIT_CONFIG_GLOBAL"])
    git_config.write_text(f"[core]\n\thooksPath = {hooks}\n", encoding="utf-8")
    plans = mapping_for(runs)
    with pytest.raises(ResearchRepoError, match="commit"):
        apply_mapping(plans, runs, register, CRITERIA, TODAY)

    git_config.write_text("", encoding="utf-8")
    result = apply_mapping(plans, runs, register, CRITERIA, TODAY)
    assert (result.cards_written, result.trials_appended, result.committed) == (0, 0, True)
    assert [t.run_id for t in read_trials(register / LEDGER_FILENAME)] == ["r3", "r1", "r2"]
    assert git_status(register) == ""
    assert git_log(register) == [
        "chore: backfill legacy research",
        "chore: initialise research register",
    ]


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        (
            {0: {"runs": ["r3", "r1"]}, 1: {"runs": ["r2"]}},
            "r1 is recorded on S-0002 but mapped to S-0001",
        ),
        ({1: {"id": "S-0050"}}, "r1 is recorded on S-0002 but mapped to S-0050"),
        (
            {0: {"studies": []}, 1: {"studies": [STUDY]}},
            "study:S-0001:config-4 is recorded on S-0001 but missing from the mapping",
        ),
        (
            {0: {"studies": [{**STUDY, "variant": "config-4b"}]}},
            "study:S-0001:config-4 is recorded on S-0001 but missing from the mapping",
        ),
    ],
    ids=["move-run", "renumber-card", "move-study", "rename-variant"],
)
def test_regrouping_after_an_apply_is_refused(runs, register, changes, fragment):
    first = draft_mapping(runs)
    first["cards"][0]["studies"] = [STUDY]
    apply_mapping(parse_mapping(first), runs, register, CRITERIA, TODAY)
    regrouped = copy.deepcopy(first)
    for index, change in changes.items():
        regrouped["cards"][index].update(change)
    before = register_files(register)
    with pytest.raises(BackfillError, match=fragment) as excinfo:
        apply_mapping(parse_mapping(regrouped), runs, register, CRITERIA, TODAY)
    assert "reset the register" in str(excinfo.value)
    assert register_files(register) == before


def test_a_recorded_run_cannot_leave_the_mapping(runs, register):
    apply_mapping(mapping_for(runs), runs, register, CRITERIA, TODAY)
    remaining = [run for run in runs if run.run_id != "r1"]
    with pytest.raises(
        BackfillError, match="r1 is recorded on S-0002 but missing from the mapping"
    ):
        apply_mapping(mapping_for(remaining), remaining, register, CRITERIA, TODAY)


@pytest.mark.parametrize(
    "change",
    [
        {"title": "pdh retest v2"},
        {"slug": "pdh-v2"},
        {"status": "rejected"},
        {"tags": ["pdh"]},
        {"body": "## Verdict\n\nNo edge.\n"},
    ],
    ids=["title", "slug", "status", "tags", "body"],
)
def test_existing_cards_are_not_edited_through_the_mapping(runs, register, change):
    apply_mapping(mapping_for(runs), runs, register, CRITERIA, TODAY)
    field = next(iter(change))
    with pytest.raises(BackfillError, match=f"S-0001: the mapping changes the {field} "):
        apply_mapping(mapping_for(runs, **{"S-0001": change}), runs, register, CRITERIA, TODAY)


def test_configs_of_recorded_runs_are_stored_again(runs, register):
    plans = mapping_for(runs)
    apply_mapping(plans, runs, register, CRITERIA, TODAY)
    (stored,) = (register / "configs").glob("*.json")
    content = stored.read_bytes()
    stored.unlink()
    apply_mapping(plans, runs, register, CRITERIA, TODAY)
    assert stored.read_bytes() == content
