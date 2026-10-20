import os
from datetime import date

import pytest

from fwbg.research.cli import main
from fwbg.research.card import CriteriaAmendment, Status, criteria_hash, write_card
from fwbg.research.commands import CommandError
from fwbg.research.fingerprint import digest
from fwbg.research.ledger import LEDGER_FILENAME, TrialKind, TrialStatus, append_trial
from fwbg.research.pending import write_pending
from fwbg.research.repo import store_config
from fwbg.research.views import coverage, list_lines, show_text
from tests.research.factories import make_card, make_trial
from tests.research.test_pending import make_pending

BODY = "## Hypothesis\n\nEvents continue.\n\n## Verdict\n\n## Reports\n"


def config(range_bars=None, symbols=("EURUSD",), **strategy):
    indicators = [
        {"name": "cusum", "params": {} if range_bars is None else {"range_bars": range_bars}}
    ]
    return {
        "strategy": {"timeframe": "HOUR_1", "pipeline": {"indicators": indicators}, **strategy},
        "symbols": list(symbols),
    }


@pytest.fixture
def register(strategies_dir):
    """A register on disk (no git needed for reading) with cards S-0007 and S-0008."""
    (strategies_dir / "configs").mkdir()
    (strategies_dir / LEDGER_FILENAME).touch()
    write_card(strategies_dir, make_card(body=BODY), "cusum-event-continuation")
    write_card(
        strategies_dir,
        make_card(id="S-0008", title="Opening range", status=Status.PARKED, tags=("orb",)),
        "opening-range",
    )
    return strategies_dir


def add_trial(register, cfg=None, **overrides):
    if cfg is not None:
        overrides["fingerprint"] = digest(cfg)
        store_config(register, overrides["fingerprint"], cfg)
    return append_trial(register / LEDGER_FILENAME, make_trial(**overrides))


# --- list --------------------------------------------------------------------


def test_list_shows_one_line_per_card_sorted_by_id(register):
    add_trial(register, metrics={"EURUSD": {"sharpe": 0.4}})
    add_trial(register, metrics={"GBPUSD": {"sharpe": 0.9}, "EURUSD": {"sharpe": 0.1}})
    add_trial(register, metrics={"EURUSD": {"sharpe": 0.2}})
    add_trial(register, status=TrialStatus.FAILED)
    assert list_lines(register) == [
        "S-0007  active    3 trials  best Sharpe 0.90 (T-000002 GBPUSD)  "
        "holdout unused  CUSUM event continuation",
        "S-0008  parked    0 trials  best Sharpe -                       "
        "holdout unused  Opening range",
    ]


def test_list_shows_holdout_state(register):
    add_trial(register, kind=TrialKind.HOLDOUT, metrics={"EURUSD": {"sharpe": 2.0}})
    write_card(
        register,
        make_card(id="S-0009", title="Dirty", holdout_contaminated=True),
        "dirty",
    )
    lines = list_lines(register)
    assert "1 trial   best Sharpe -" in lines[0]
    assert "holdout used 1x" in lines[0]
    assert "holdout contaminated" in lines[2]


def test_list_filters_by_status_and_tag(register):
    assert [line[:6] for line in list_lines(register, status=Status.PARKED)] == ["S-0008"]
    assert [line[:6] for line in list_lines(register, tag="event")] == ["S-0007"]
    assert list_lines(register, status=Status.ACTIVE, tag="orb") == []


def test_list_reports_an_unreadable_card_instead_of_aborting(register):
    (register / "strategies" / "S-0009-broken.md").write_text("no front matter", encoding="utf-8")
    (register / "strategies" / "notes.md").write_text("not a card", encoding="utf-8")
    lines = list_lines(register)
    assert [line[:6] for line in lines] == ["S-0007", "S-0008", "S-0009"]
    assert lines[2] == "S-0009  unreadable: S-0009-broken.md: card has no YAML front matter"


def test_list_truncates_long_titles(register):
    write_card(register, make_card(id="S-0009", title="x" * 200), "long")
    line = list_lines(register)[2]
    assert len(line) == 120
    assert line.endswith("x…")


# --- show --------------------------------------------------------------------


def test_show_prints_header_criteria_body_and_timeline(register):
    add_trial(register, metrics={"GBPUSD": {"sharpe": 0.9}, "EURUSD": {"sharpe": 0.1}})
    add_trial(register, run_id="run-2", status=TrialStatus.FAILED, note="crashed early")
    text = show_text(register, "S-0007")
    lines = text.splitlines()
    assert lines[0] == "S-0007  CUSUM event continuation"
    assert "status: active  tags: event  created: 2026-10-19" in lines
    assert "holdout cutoff: 2025-04-30  contaminated: no  holdout unused" in lines
    assert "Criteria (registered):" in lines
    assert "  min_sharpe  1" in lines
    assert "  min_trades  100" in lines
    assert "## Hypothesis" in lines
    assert "## Reports" in lines
    assert "Runs in progress: none" in lines
    assert "Trials: 1 counted (2 symbol evaluations), 1 failed or smoke" in lines
    assert "  T-000001  2026-10-19  iteration ok     EURUSD,GBPUSD       0.90  baseline" in lines
    assert (
        "  T-000002  2026-10-19  iteration failed -                      -  crashed early" in lines
    )


def test_show_states_amended_criteria(register):
    criteria = {"min_sharpe": 0.8, "min_trades": 100.0}
    amendments = (
        CriteriaAmendment(date(2026, 10, 20), "first", criteria_hash({"min_sharpe": 0.9})),
        CriteriaAmendment(date(2026, 11, 2), "second", criteria_hash(criteria)),
    )
    card = make_card(id="S-0009", success_criteria=criteria, criteria_amendments=amendments)
    write_card(register, card, "amended")
    assert "Criteria (amended 2x (latest 2026-11-02)):" in show_text(register, "S-0009")


def test_show_flags_hand_edited_criteria(register):
    write_card(register, make_card(id="S-0009", success_criteria={"min_sharpe": 5.0}), "edited")
    assert "Criteria (edited by hand - runs are refused):" in show_text(register, "S-0009")


def test_show_lists_live_runs_of_the_card(register, tmp_path):
    write_pending(register, make_pending(tmp_path / "run-9", run_id="run-9", pid=os.getpid()))
    write_pending(register, make_pending(tmp_path / "x", run_id="other", strategy="S-0008"))
    lines = show_text(register, "S-0007").splitlines()
    assert "Runs in progress:" in lines
    assert f"  run-9  started 2026-10-19T10:00  pid {os.getpid()}  baseline" in lines
    assert not any("other" in line for line in lines)


def test_show_coverage_leaves_out_trials_without_a_stored_config(register):
    add_trial(register, cfg=config(10))
    add_trial(register, cfg=config(20))
    add_trial(register, run_id="study:S-0007:a", fingerprint="sha256:" + "1" * 64)
    lines = show_text(register, "S-0007").splitlines()
    assert "Coverage: 2 configs (1 trial without a stored config left out)" in lines
    assert any(
        line.startswith("  strategy.pipeline.indicators[0].params.range_bars") for line in lines
    )


def test_show_of_an_unknown_card_is_an_error(register):
    with pytest.raises(CommandError, match="no research card S-0042"):
        show_text(register, "S-0042")


# --- coverage ----------------------------------------------------------------


RANGE_BARS = "strategy.pipeline.indicators[0].params.range_bars"


def row(path, values):
    return f"{path:<{len(RANGE_BARS)}}  {values}"


def test_coverage_lists_varied_paths_and_the_window():
    configs = [
        config(20, ("EURUSD",)),
        config(10, ("EURUSD", "GBPUSD"), end_date="2025-04-30"),
        config(None, ("EURUSD",), end_date="2025-04-30"),
    ]
    assert coverage(configs) == [
        row("symbols", '["EURUSD","GBPUSD"], ["EURUSD"]'),
        row("strategy.timeframe", '"HOUR_1"'),
        row("strategy.start_date", "-"),
        row("strategy.end_date", '"2025-04-30", -'),
        row(RANGE_BARS, "10, 20, -"),
    ]


def test_coverage_caps_the_values_shown():
    lines = coverage([config(value) for value in range(13)])
    assert lines[-1] == row(RANGE_BARS, "0, 1, 2, 3, 4, 5, 6, 7, 8, 9 … (+3 more)")


def test_coverage_keeps_lines_short():
    lines = coverage([config("x" * 30 + str(value)) for value in range(5)])
    assert len(lines[-1]) <= 118
    assert lines[-1] == row(RANGE_BARS, '"' + "x" * 30 + '0" … (+4 more)')


def test_coverage_of_no_configs_is_empty():
    assert coverage([]) == []


# --- CLI ---------------------------------------------------------------------


def test_cli_lists_and_shows_cards(register, monkeypatch, capsys):
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(register))
    assert main(["list", "--status", "parked"]) == 0
    assert capsys.readouterr().out.startswith("S-0008  parked")
    assert main(["list", "--tag", "event"]) == 0
    assert capsys.readouterr().out.startswith("S-0007  active")
    assert main(["show", "S-0007"]) == 0
    assert capsys.readouterr().out.startswith("S-0007  CUSUM event continuation\n")


def test_cli_show_of_an_unknown_card_exits_1(register, monkeypatch, capsys):
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(register))
    assert main(["show", "S-0042"]) == 1
    assert "Error: no research card S-0042" in capsys.readouterr().err


def test_list_aligns_columns_across_rows(register):
    for _ in range(12):
        add_trial(register, metrics={"PORTFOLIO": {"sharpe": 0.5}})
    lines = list_lines(register)
    assert lines[0].index("best") == lines[1].index("best")
    assert lines[0].index("holdout") == lines[1].index("holdout")
