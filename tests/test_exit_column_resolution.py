"""resolve_role_column: how an exit finds the column it reads, and what it says when it cannot."""

import logging

import pytest

from fwbg.core.columns import AmbiguousColumnError, resolve_role_column


def _is_stop(column: str) -> bool:
    return column.endswith("_sl_dist")


def test_a_missing_plain_name_falls_back_with_a_warning(caplog):
    columns = ["rb4_orb_s00_sl_dist", "feat"]
    with caplog.at_level(logging.WARNING, logger="fwbg.core.columns"):
        chosen = resolve_role_column(columns, "rb4_orb_s99_sl_dist", _is_stop, "sl_dist_column")
    assert chosen == "rb4_orb_s00_sl_dist"
    assert "rb4_orb_s99_sl_dist" in caplog.text
    assert "auto-detect" in caplog.text


def test_the_error_lists_at_most_eight_candidates_and_the_total():
    columns = [f"rb{rb}_orb_s{s:02d}_sl_dist" for rb in (1, 2) for s in range(10)]
    with pytest.raises(AmbiguousColumnError) as err:
        resolve_role_column(columns, None, _is_stop, "sl_dist_column")
    message = str(err.value)
    assert "(20 in total)" in message
    assert sum(c in message for c in columns) == 8


def test_a_pattern_is_suggested_only_when_it_names_one_column():
    one_session = ["rb4_orb_s00_sl_dist", "rb4_orb_s00_rl50_sl_dist"]
    with pytest.raises(AmbiguousColumnError) as err:
        resolve_role_column(one_session, None, _is_stop, "sl_dist_column")
    assert "'*_orb_s00_sl_dist'" in str(err.value)

    two_ranges = ["rb4_orb_s00_sl_dist", "rb8_orb_s00_sl_dist"]
    with pytest.raises(AmbiguousColumnError) as err:
        resolve_role_column(two_ranges, None, _is_stop, "sl_dist_column")
    assert "pattern such as" not in str(err.value)
