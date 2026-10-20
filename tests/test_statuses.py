import re
from pathlib import Path

from fwbg.statuses import EXECUTED_STATUSES, NOT_EXECUTED_STATUSES

PROCESS_SOURCE = Path(__file__).parents[1] / "src" / "fwbg" / "optimization" / "process.py"


def test_every_status_process_symbol_returns_is_classified():
    source = PROCESS_SOURCE.read_text(encoding="utf-8")
    produced = set(re.findall(r'"status":\s*"([a-z_]+)"', source))
    assert produced, "the pattern no longer finds statuses; update this test"
    assert produced <= EXECUTED_STATUSES | NOT_EXECUTED_STATUSES
    assert not EXECUTED_STATUSES & NOT_EXECUTED_STATUSES
