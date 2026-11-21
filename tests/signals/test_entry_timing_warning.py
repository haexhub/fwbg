"""entry_delay=0 on a rule that reads an unshifted signal column warns.

An unshifted signal column at row i uses bar i's close. With ``entry_delay=0`` the
order fills at that same close, so the decision is only executable as a
market-on-close style order. That stays a warning: the backtest allows it.
"""

import logging

from fwbg.core.config import ExitStrategyConfig
from fwbg.core.context import SimulationContext
from fwbg.signals.timing import (
    close_decision_columns,
    entry_delay_warnings,
    rule_columns,
    warn_close_decisions,
)

IBS = {"name": "ibs", "params": {}}
ORB = {"name": "opening_range", "params": {"sessions": [8], "range_bars": 1}}
MACD = {"name": "macd", "params": {}}
TOM = {"name": "turn_of_month", "params": {}}


def rules(*columns, nested=False):
    conditions = [{"type": "signal_active", "column": c} for c in columns]
    if nested:
        conditions = [{"type": "group", "operator": "OR", "conditions": conditions}]
    return {"long": {"operator": "AND", "conditions": conditions}}


def make_ctx(indicators, signal_rules=None, entry_delay=None, **overrides):
    params = {} if entry_delay is None else {"entry_delay": entry_delay}
    fields = dict(
        symbol="TESTUSD",
        asset_class="FOREX",
        spread=0.0002,
        point=0.0001,
        indicator_plugins=indicators,
        signal_rules=signal_rules,
        exit_strategies=[ExitStrategyConfig(name="fixed", params=params)],
    )
    fields.update(overrides)
    return SimulationContext(**fields)


def test_entry_delay_zero_on_an_unshifted_signal_warns():
    ctx = make_ctx([IBS], rules("ibs_down"), entry_delay=0)
    (message,) = entry_delay_warnings(ctx)
    assert "entry_delay=0" in message
    assert "ibs_down" in message
    assert "market-on-close" in message


def test_default_entry_delay_does_not_warn():
    assert entry_delay_warnings(make_ctx([IBS], rules("ibs_down"))) == []


def test_entry_delay_one_does_not_warn():
    assert entry_delay_warnings(make_ctx([IBS], rules("ibs_down"), entry_delay=1)) == []


def test_any_exit_strategy_with_entry_delay_zero_triggers():
    ctx = make_ctx([IBS], rules("ibs_down"))
    ctx.exit_strategies = [
        ExitStrategyConfig(name="fixed", params={"entry_delay": 1}),
        ExitStrategyConfig(name="fixed", params={"entry_delay": 0}),
    ]
    assert len(entry_delay_warnings(ctx)) == 1


def test_short_rule_names_and_nested_groups_are_resolved():
    ctx = make_ctx([ORB], rules("breakout_up", nested=True), entry_delay=0)
    (message,) = entry_delay_warnings(ctx)
    assert "rb1_orb_s08_breakout_up" in message


def test_shifted_columns_do_not_warn():
    ctx = make_ctx([MACD], rules("macd_above_zero"), entry_delay=0)
    assert entry_delay_warnings(ctx) == []


def test_calendar_signal_needs_no_close_and_does_not_warn():
    ctx = make_ctx([TOM], rules("tom_entry_long"), entry_delay=0)
    assert entry_delay_warnings(ctx) == []


def test_signal_columns_not_used_by_a_rule_do_not_warn():
    ctx = make_ctx([IBS, MACD], rules("macd_above_zero"), entry_delay=0)
    assert entry_delay_warnings(ctx) == []


def test_signal_model_required_features_count_as_entry_rule():
    ctx = make_ctx(
        [IBS], None, entry_delay=0, model_type="signal", required_features=["ibs_down"]
    )
    assert len(entry_delay_warnings(ctx)) == 1


def test_required_features_of_a_ml_model_are_not_an_entry_rule():
    ctx = make_ctx([IBS], None, entry_delay=0, required_features=["ibs_down"])
    assert entry_delay_warnings(ctx) == []


def test_warning_is_logged_not_raised(caplog):
    ctx = make_ctx([IBS], rules("ibs_down"), entry_delay=0)
    with caplog.at_level(logging.WARNING, logger="fwbg.signals.timing"):
        messages = warn_close_decisions(ctx, "TEST")
    assert len(messages) == 1
    assert any("entry_delay=0" in r.getMessage() for r in caplog.records)


def test_helpers():
    nested = rules("a", "b", nested=True)
    nested["long"]["conditions"].append(
        {"type": "col_compare", "column_a": "c", "column_b": "a", "op": ">"}
    )
    assert rule_columns(nested) == ["a", "b", "c"]
    assert close_decision_columns([IBS], rules("other"), ()) == []
    assert close_decision_columns([IBS], rules("ibs_down"), ()) == ["ibs_down"]
    # a short name resolves by suffix, as the rule evaluator does
    assert close_decision_columns([IBS], rules("down"), ()) == ["ibs_down"]
