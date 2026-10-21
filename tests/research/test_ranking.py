"""Evaluations that chose their setting on test folds count in N but are not ranked."""

import logging
from datetime import date

import pytest

from fwbg.research.card import Status, load_card
from fwbg.research.commands import write_verdict
from fwbg.research.fingerprint import digest
from fwbg.research.ledger import LEDGER_FILENAME, TrialKind, append_trial
from fwbg.research.repo import config_path, store_config
from fwbg.research.stats import (
    INDICATOR_GRID,
    INNER_SELECTION_PROTOCOL,
    SIGNAL_MODEL,
    best_result,
    deflated_best,
    evaluations,
    return_samples,
    unrankable_configs,
    unranked_evaluations,
    unranked_note,
)
from fwbg.research.views import card_summary, list_lines, show_text
from tests.research.factories import make_card, make_trial

TODAY = date(2026, 10, 20)
INPUTS = {"sr_per_trade": 0.1, "n_returns": 101, "skew": 0.0, "kurtosis": 3.0}
SIGNAL = "sha256:" + "a" * 64
ML = "sha256:" + "b" * 64
UNSTORED = "sha256:" + "c" * 64
GRID = "sha256:" + "f" * 64
UNRANKABLE = {SIGNAL: SIGNAL_MODEL, GRID: INDICATOR_GRID}


def config(model_type: str, grid: dict | None = None) -> dict:
    strategy: dict = {"timeframe": "HOUR_1", "model": {"type": model_type}}
    if grid is not None:
        strategy["optimization"] = {"indicator_grid": grid}
    return {"strategy": strategy, "symbols": ["X"]}


def stored(register, model_type: str, grid: dict | None = None) -> str:
    cfg = config(model_type, grid)
    fingerprint = digest(cfg)
    store_config(register, fingerprint, cfg)
    return fingerprint


def row(sr: float, protocol: int | None = None) -> dict:
    values: dict = {"sharpe": sr * 10, **INPUTS, "sr_per_trade": sr}
    if protocol is not None:
        values["selection_protocol"] = protocol
    return values


def verdict_body(register) -> str:
    card = load_card(register, "S-0007")
    assert card is not None
    return card.body


# --- stats -------------------------------------------------------------------


def test_the_protocol_is_the_inner_selection():
    assert INNER_SELECTION_PROTOCOL == 2


def test_unmarked_unrankable_rows_count_in_n_but_are_not_ranked():
    trials = [
        make_trial(trial_id="T-000001", fingerprint=SIGNAL, metrics={"A": row(0.9)}),
        make_trial(trial_id="T-000002", fingerprint=SIGNAL, metrics={"A": row(0.8, 1)}),
        make_trial(trial_id="T-000003", fingerprint=SIGNAL, metrics={"A": row(0.1, 2)}),
        make_trial(trial_id="T-000004", fingerprint=ML, metrics={"A": row(0.2)}),
        make_trial(trial_id="T-000005", fingerprint=UNSTORED, metrics={"A": row(0.3)}),
        make_trial(trial_id="T-000006", fingerprint=GRID, metrics={"A": row(0.7)}),
        make_trial(trial_id="T-000007", fingerprint=GRID, metrics={"A": row(0.05, 3)}),
    ]
    best = best_result(trials, UNRANKABLE)
    assert best is not None and best.trial_id == "T-000005"
    ranked = [s.trial_id for s in return_samples(trials, UNRANKABLE)]
    assert ranked == ["T-000003", "T-000004", "T-000005", "T-000007"]
    result = deflated_best(trials, UNRANKABLE)
    assert result is not None
    assert (result.best.trial_id, result.samples, result.n_trials) == ("T-000005", 4, 7)
    assert evaluations(trials) == 7
    assert unranked_evaluations(trials, UNRANKABLE) == {SIGNAL_MODEL: 2, INDICATOR_GRID: 1}
    # Nothing known to be unrankable: everything is ranked.
    unfiltered = best_result(trials, {})
    assert unfiltered is not None and unfiltered.trial_id == "T-000001"
    assert unranked_evaluations(trials, {}) == {}


def test_unranked_rows_are_counted_per_development_symbol():
    sharpe_only = {"sharpe": 0.5, "sr_per_trade": None}
    trials = [
        make_trial(fingerprint=SIGNAL, metrics={"A": row(0.2), "B": row(0.1, 2), "C": INPUTS}),
        make_trial(fingerprint=SIGNAL, metrics={"D": sharpe_only}),
        make_trial(fingerprint=SIGNAL, kind=TrialKind.HOLDOUT, metrics={"A": row(0.2)}),
    ]
    assert unranked_evaluations(trials, UNRANKABLE) == {SIGNAL_MODEL: 3}


def test_metric_less_rows_count_in_n_but_not_as_unranked():
    # E.g. a run that ended before any fold result: nothing to rank, so nothing left out.
    empty = {"sharpe": None, "sr_per_trade": None, "selection_protocol": None}
    trials = [make_trial(fingerprint=SIGNAL, metrics={"A": empty, "B": {}})]
    assert evaluations(trials) == 2
    assert unranked_evaluations(trials, UNRANKABLE) == {}


@pytest.mark.parametrize(
    ("counts", "note"),
    [
        ({}, None),
        (
            {SIGNAL_MODEL: 1},
            "1 pre-protocol-2 evaluation not ranked (counted in N): 1 signal-model",
        ),
        (
            {INDICATOR_GRID: 2, SIGNAL_MODEL: 3},
            "5 pre-protocol-2 evaluations not ranked (counted in N): "
            "3 signal-model, 2 indicator-grid",
        ),
    ],
)
def test_unranked_note(counts, note):
    assert unranked_note(counts) == note


def test_unrankable_configs_reads_model_type_and_indicator_grid(gate_register, caplog):
    signal = stored(gate_register, "signal")
    signal_grid = stored(gate_register, "signal", {"st": {"period": [10, 21]}})
    ml = stored(gate_register, "xgboost")
    # Two variants: one per indicator (expand_indicator_grid sums over indicators).
    ml_grid = stored(gate_register, "xgboost", {"a": {"n": [1]}, "b": {"m": [[2]]}})
    ml_product = stored(gate_register, "xgboost_rrr", {"or": {"bars": [4, 8], "pre": [[0]]}})
    ml_single = stored(gate_register, "xgboost", {"or": {"bars": [4], "pre": [[0]]}})
    ml_empty = stored(gate_register, "xgboost", {})
    corrupt = "sha256:" + "d" * 64
    config_path(gate_register, corrupt).write_text("{not json", encoding="utf-8")
    odd = "sha256:" + "e" * 64
    config_path(gate_register, odd).write_text("[1]", encoding="utf-8")
    fingerprints = (
        signal,
        signal,
        signal_grid,
        ml,
        ml_grid,
        ml_product,
        ml_single,
        ml_empty,
        corrupt,
        odd,
        UNSTORED,
    )
    trials = [make_trial(fingerprint=fingerprint) for fingerprint in fingerprints]
    with caplog.at_level(logging.WARNING, logger="fwbg.research.stats"):
        found = unrankable_configs(gate_register, trials)
    assert found == {
        signal: SIGNAL_MODEL,
        signal_grid: SIGNAL_MODEL,  # both reasons: counted once, as a signal model
        ml_grid: INDICATOR_GRID,
        ml_product: INDICATOR_GRID,
    }
    assert f"configs/{'d' * 64}.json" in caplog.text


# --- verdict, show, list, API summary ----------------------------------------


def test_verdict_names_the_unranked_evaluations_per_reason(gate_register):
    ledger = gate_register / LEDGER_FILENAME
    signal = stored(gate_register, "signal")
    grid = stored(gate_register, "xgboost", {"or": {"bars": [4, 8]}})
    ml = stored(gate_register, "xgboost")
    append_trial(ledger, make_trial(fingerprint=signal, metrics={"A": row(0.9), "B": row(0.5)}))
    append_trial(ledger, make_trial(fingerprint=grid, metrics={"A": row(0.8)}))
    append_trial(ledger, make_trial(fingerprint=ml, metrics={"A": row(0.1), "B": row(0.2)}))
    write_verdict(gate_register, "S-0007", Status.PARKED, "x", "y", TODAY)
    body = verdict_body(gate_register)
    assert (
        "\n3 pre-protocol-2 evaluations not ranked (counted in N): 2 signal-model, "
        "1 indicator-grid.\nBest development result: T-000003 B, "
    ) in body
    assert "Deflated Sharpe ratio: " in body and "for T-000003 B (" in body
    assert "N=5 symbol evaluations" in body


def test_verdict_with_only_unranked_rows_has_no_best_result(gate_register):
    signal = stored(gate_register, "signal")
    append_trial(
        gate_register / LEDGER_FILENAME,
        make_trial(fingerprint=signal, metrics={"A": row(0.9), "B": row(0.5)}),
    )
    write_verdict(gate_register, "S-0007", Status.PARKED, "x", "y", TODAY)
    body = verdict_body(gate_register)
    assert "(2 symbol evaluations)" in body
    assert "2 pre-protocol-2 evaluations not ranked (counted in N): 2 signal-model." in body
    assert "Best development result: none with a Sharpe ratio." in body
    assert "Deflated Sharpe ratio: too few trials with return statistics (0 of 2 needed)." in body


def test_verdict_without_unranked_rows_does_not_mention_them(gate_register):
    ml = stored(gate_register, "xgboost", {"or": {"bars": [4]}})
    append_trial(gate_register / LEDGER_FILENAME, make_trial(fingerprint=ml, metrics={"A": row(1)}))
    write_verdict(gate_register, "S-0007", Status.PARKED, "x", "y", TODAY)
    assert "not ranked" not in verdict_body(gate_register)


def test_verdict_survives_a_corrupt_config(gate_register):
    corrupt = "sha256:" + "d" * 64
    config_path(gate_register, corrupt).write_text("{not json", encoding="utf-8")
    append_trial(
        gate_register / LEDGER_FILENAME, make_trial(fingerprint=corrupt, metrics={"A": row(0.9)})
    )
    write_verdict(gate_register, "S-0007", Status.PARKED, "x", "y", TODAY)
    assert "Best development result: T-000001 A, " in verdict_body(gate_register)


def test_show_and_list_leave_unranked_rows_out(gate_register):
    grid = stored(gate_register, "xgboost", {"or": {"bars": [4, 8]}})
    ml = stored(gate_register, "xgboost")
    ledger = gate_register / LEDGER_FILENAME
    append_trial(ledger, make_trial(fingerprint=grid, metrics={"A": row(0.9)}))
    append_trial(ledger, make_trial(fingerprint=ml, metrics={"B": row(0.1)}))
    note = "1 pre-protocol-2 evaluation not ranked (counted in N): 1 indicator-grid"
    assert note in show_text(gate_register, "S-0007").splitlines()
    assert "best Sharpe 1.00 (T-000002 B)" in list_lines(gate_register)[0]


def test_card_summary_ranks_without_unrankable_rows():
    own = [
        make_trial(trial_id="T-000001", fingerprint=GRID, metrics={"A": row(0.9)}),
        make_trial(trial_id="T-000002", fingerprint=ML, metrics={"B": row(0.1)}),
    ]
    summary = card_summary(make_card(), own, running=0, unrankable=UNRANKABLE)
    assert summary["best"]["trial"] == "T-000002"
