"""Asset filters in ``select_run_files`` match whole symbols, not substrings of the file path."""

from __future__ import annotations

import pytest

from fwbg.core.config import StrategyConfig
from fwbg.optimization.run_files import file_symbol, select_run_files

SYMBOLS = ["DB", "HDB", "TS", "TSM", "E", "GE", "PFE", "SAP.DE", "BRK-B"]


@pytest.fixture
def data_dir(tmp_path):
    for sym in SYMBOLS:
        (tmp_path / f"{sym}_DAY_1.csv").write_text("T,O,H,L,C,V\n")
    return tmp_path


def _select(data_dir, assets, asset_filter=None):
    meta = {"name": "t", "timeframe": "DAY_1", "assets": assets, "_cli_data_path": str(data_dir)}
    files = select_run_files(StrategyConfig.from_dict(meta), meta, asset_filter)
    return sorted(file_symbol(f) for f in files or [])


def test_file_symbol_is_the_name_before_the_first_underscore():
    assert file_symbol("/data/x/SAP.DE_DAY_1.csv") == "SAP.DE"
    assert file_symbol("/data/x/brk-b_MINUTE_15.csv") == "BRK-B"


@pytest.mark.parametrize(("listed", "expected"), [
    (["DB"], ["DB"]),          # not HDB
    (["TS"], ["TS"]),          # not TSM
    (["E"], ["E"]),            # not GE, PFE or every other symbol containing an E
    (["sap.de", "BRK-B"], ["BRK-B", "SAP.DE"]),
])
def test_strategy_filter_matches_whole_symbols(data_dir, listed, expected):
    assert _select(data_dir, {"filter": listed}) == expected


def test_exclude_matches_whole_symbols(data_dir):
    assert _select(data_dir, {"exclude": ["E", "DB"]}) == sorted(s for s in SYMBOLS if s not in ("E", "DB"))


def test_cli_filter_matches_whole_symbols(data_dir):
    assert _select(data_dir, {}, asset_filter=["GE"]) == ["GE"]
