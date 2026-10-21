"""Signal-model evaluations from before inner selection count in N but are not ranked."""

import logging
from datetime import date

from fwbg.research.card import Status, load_card
from fwbg.research.commands import write_verdict
from fwbg.research.fingerprint import digest
from fwbg.research.ledger import LEDGER_FILENAME, TrialKind, append_trial
from fwbg.research.repo import store_config
from fwbg.research.stats import (
    CURRENT_PROTOCOL,
    best_result,
    deflated_best,
    evaluations,
    return_samples,
    signal_configs,
    unranked_evaluations,
)
from fwbg.research.views import card_summary, list_lines, show_text
from tests.research.factories import make_card, make_trial

TODAY = date(2026, 10, 20)
INPUTS = {"sr_per_trade": 0.1, "n_returns": 101, "skew": 0.0, "kurtosis": 3.0}
SIGNAL = "sha256:" + "a" * 64
ML = "sha256:" + "b" * 64
UNSTORED = "sha256:" + "c" * 64
NOTE = "signal-model evaluations predate selection protocol 2 (counted in N, not ranked)"


def config(model_type: str) -> dict:
    return {"strategy": {"timeframe": "HOUR_1", "model": {"type": model_type}}, "symbols": ["X"]}


def stored(register, model_type: str) -> str:
    cfg = config(model_type)
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


def test_the_current_protocol_is_the_inner_selection():
    assert CURRENT_PROTOCOL == 2


def test_unmarked_signal_rows_count_in_n_but_are_not_ranked():
    trials = [
        make_trial(trial_id="T-000001", fingerprint=SIGNAL, metrics={"A": row(0.9)}),
        make_trial(trial_id="T-000002", fingerprint=SIGNAL, metrics={"A": row(0.8, 1)}),
        make_trial(trial_id="T-000003", fingerprint=SIGNAL, metrics={"A": row(0.1, 2)}),
        make_trial(trial_id="T-000004", fingerprint=ML, metrics={"A": row(0.2)}),
        make_trial(trial_id="T-000005", fingerprint=UNSTORED, metrics={"A": row(0.3)}),
    ]
    signal = frozenset({SIGNAL})
    best = best_result(trials, signal)
    assert best is not None and best.trial_id == "T-000005"
    assert [s.trial_id for s in return_samples(trials, signal)] == [
        "T-000003",
        "T-000004",
        "T-000005",
    ]
    result = deflated_best(trials, signal)
    assert result is not None
    assert (result.best.trial_id, result.samples, result.n_trials) == ("T-000005", 3, 5)
    assert evaluations(trials) == 5
    assert unranked_evaluations(trials, signal) == 2
    # Without stored signal configs everything stays rankable.
    unfiltered = best_result(trials)
    assert unfiltered is not None and unfiltered.trial_id == "T-000001"
    assert unranked_evaluations(trials) == 0


def test_unranked_rows_are_counted_per_development_symbol():
    sharpe_only = {"sharpe": 0.5, "sr_per_trade": None}
    trials = [
        make_trial(fingerprint=SIGNAL, metrics={"A": row(0.2), "B": row(0.1, 2), "C": INPUTS}),
        make_trial(fingerprint=SIGNAL, metrics={"D": sharpe_only}),
        make_trial(fingerprint=SIGNAL, kind=TrialKind.HOLDOUT, metrics={"A": row(0.2)}),
    ]
    assert unranked_evaluations(trials, {SIGNAL}) == 3


def test_metric_less_signal_rows_count_in_n_but_not_as_unranked():
    # E.g. a run that ended before any fold result: nothing to rank, so nothing left out.
    empty = {"sharpe": None, "sr_per_trade": None, "selection_protocol": None}
    trials = [make_trial(fingerprint=SIGNAL, metrics={"A": empty, "B": {}})]
    assert evaluations(trials) == 2
    assert unranked_evaluations(trials, {SIGNAL}) == 0


def test_signal_configs_reads_the_model_type_of_stored_configs(gate_register, caplog):
    signal = stored(gate_register, "signal")
    ml = stored(gate_register, "xgboost")
    corrupt = "sha256:" + "d" * 64
    (gate_register / "configs" / f"{'d' * 64}.json").write_text("{not json", encoding="utf-8")
    odd = "sha256:" + "e" * 64
    (gate_register / "configs" / f"{'e' * 64}.json").write_text("[1]", encoding="utf-8")
    trials = [
        make_trial(fingerprint=fingerprint)
        for fingerprint in (signal, signal, ml, corrupt, odd, UNSTORED)
    ]
    with caplog.at_level(logging.WARNING, logger="fwbg.research.stats"):
        assert signal_configs(gate_register, trials) == {signal}
    assert f"configs/{'d' * 64}.json" in caplog.text


# --- verdict, show, list, API summary ----------------------------------------


def test_verdict_names_the_unranked_signal_evaluations(gate_register):
    ledger = gate_register / LEDGER_FILENAME
    signal = stored(gate_register, "signal")
    ml = stored(gate_register, "xgboost")
    append_trial(ledger, make_trial(fingerprint=signal, metrics={"A": row(0.9), "B": row(0.5)}))
    append_trial(ledger, make_trial(fingerprint=ml, metrics={"A": row(0.1), "B": row(0.2)}))
    write_verdict(gate_register, "S-0007", Status.PARKED, "x", "y", TODAY)
    body = verdict_body(gate_register)
    assert f"\n2 {NOTE}.\nBest development result: T-000002 B, " in body
    assert "Deflated Sharpe ratio: " in body and "for T-000002 B (" in body
    assert "N=4 symbol evaluations" in body


def test_verdict_with_only_unranked_rows_has_no_best_result(gate_register):
    signal = stored(gate_register, "signal")
    append_trial(
        gate_register / LEDGER_FILENAME,
        make_trial(fingerprint=signal, metrics={"A": row(0.9), "B": row(0.5)}),
    )
    write_verdict(gate_register, "S-0007", Status.PARKED, "x", "y", TODAY)
    body = verdict_body(gate_register)
    assert "(2 symbol evaluations)" in body
    assert f"2 {NOTE}." in body
    assert "Best development result: none with a Sharpe ratio." in body
    assert "Deflated Sharpe ratio: too few trials with return statistics (0 of 2 needed)." in body


def test_verdict_without_unranked_rows_does_not_mention_them(gate_register):
    ml = stored(gate_register, "xgboost")
    append_trial(gate_register / LEDGER_FILENAME, make_trial(fingerprint=ml, metrics={"A": row(1)}))
    write_verdict(gate_register, "S-0007", Status.PARKED, "x", "y", TODAY)
    assert "predate selection protocol" not in verdict_body(gate_register)


def test_verdict_survives_a_corrupt_config(gate_register):
    (gate_register / "configs" / f"{'d' * 64}.json").write_text("{not json", encoding="utf-8")
    append_trial(
        gate_register / LEDGER_FILENAME,
        make_trial(fingerprint="sha256:" + "d" * 64, metrics={"A": row(0.9)}),
    )
    write_verdict(gate_register, "S-0007", Status.PARKED, "x", "y", TODAY)
    body = verdict_body(gate_register)
    assert "Best development result: T-000001 A, " in body


def test_show_and_list_leave_unranked_rows_out(gate_register):
    signal = stored(gate_register, "signal")
    ml = stored(gate_register, "xgboost")
    ledger = gate_register / LEDGER_FILENAME
    append_trial(ledger, make_trial(fingerprint=signal, metrics={"A": row(0.9)}))
    append_trial(ledger, make_trial(fingerprint=ml, metrics={"B": row(0.1)}))
    assert f"1 {NOTE}" in show_text(gate_register, "S-0007").splitlines()
    assert "best Sharpe 1.00 (T-000002 B)" in list_lines(gate_register)[0]


def test_card_summary_ranks_without_unmarked_signal_rows():
    own = [
        make_trial(trial_id="T-000001", fingerprint=SIGNAL, metrics={"A": row(0.9)}),
        make_trial(trial_id="T-000002", fingerprint=ML, metrics={"B": row(0.1)}),
    ]
    summary = card_summary(make_card(), own, running=0, signal={SIGNAL})
    assert summary["best"]["trial"] == "T-000002"
