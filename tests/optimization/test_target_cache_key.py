"""Target dedup key: equal only when compute_targets would produce equal output."""
from types import SimpleNamespace

from fwbg.optimization.grid_search import _target_cache_key


def _ctx(**overrides):
    ctx = dict(
        exit_strategy="atr_based", exit_params={"atr_period": 14},
        exit_modifier=None, exit_modifier_params=None,
        entry_modifier=None, entry_modifier_params=None,
    )
    ctx.update(overrides)
    return SimpleNamespace(**ctx)


def test_exit_modifier_without_params_changes_key():
    assert _target_cache_key(2.0, 1.0, 0, _ctx()) != _target_cache_key(
        2.0, 1.0, 0, _ctx(exit_modifier="trailing_stop")
    )


def test_list_params_are_hashable_and_order_independent():
    a = _ctx(entry_modifier="scale_in",
             entry_modifier_params={"levels": [0.5], "qty_multiplier": 1})
    b = _ctx(entry_modifier="scale_in",
             entry_modifier_params={"qty_multiplier": 1.0, "levels": [0.5]})

    assert {_target_cache_key(2.0, 1.0, 0, a)} == {_target_cache_key(2, 1, 0, b)}


def test_different_levels_differ():
    a = _ctx(entry_modifier="scale_in", entry_modifier_params={"levels": [0.5]})
    b = _ctx(entry_modifier="scale_in", entry_modifier_params={"levels": [0.3]})

    assert _target_cache_key(2.0, 1.0, 0, a) != _target_cache_key(2.0, 1.0, 0, b)
