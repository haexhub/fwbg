# Research Register — Step 1: Core Implementation Plan

> Execute task by task. Every task is test-first and ends with a commit.
> Design: [2026-10-19-research-register-design.md](2026-10-19-research-register-design.md).

**Goal:** Build the pure core of the research register — config fingerprints,
research cards, the trial ledger and the pre-run gate — plus the research
repository layout and `code.patch` capture for runs with uncommitted changes.

**Architecture:** A new package `fwbg.research` holds domain logic only: it
takes paths and plain data and never resolves workspace locations itself.
`gate.check` is a pure function; loading the card and ledger and acting on
the decision is left to the CLI/API wiring of step 3. `code.patch` capture
lives in `fwbg.results.code_patch` and is hooked into the existing run
directory creation.

**Tech stack:** Python ≥ 3.11, stdlib (`dataclasses`, `enum`, `fcntl`,
`hashlib`, `json`, `subprocess`), PyYAML (already a dependency), pytest.

**Out of scope (later steps):** CLI/API wiring, `fwbg research` commands,
writing `configs/<fingerprint>.json`, auto-commits after runs, backfill,
verdicts and DSR, holdout evaluation mechanics.

---

## Design refinements made while planning

The design left these open or stated them loosely; this plan settles them,
and the design doc has been updated to match.

1. **Fingerprint input.** The CLI already merges every override
   (`timeframe`, `start_date`, `end_date`, `cost_multiplier`,
   `_cli_data_path`) into `strategy_metadata` before calling
   `run_optimizer` (`src/fwbg/cli/main.py`, `main()`). The fingerprint hashes
   that dict **after preset resolution** (`fwbg.core.presets.resolve_presets`,
   the same resolution `StrategyConfig.from_dict` uses), minus descriptive,
   scheduling, location (`_strategy_dir`) and bookkeeping (`_refs`) keys,
   plus the resolved symbols. Unresolved preset names are refused. There is
   a single `fingerprint_version: 1`.
   Known ceiling: defaults are not resolved, so a config that spells out a
   default and one that omits it hash differently.
2. **Criteria hash.** The card stores
   `registered_criteria_hash`, written by the registration command from
   `success_criteria` and never accepted as a hand-edited value. The current
   hash is computed from `success_criteria`; amendments live in the front
   matter as `criteria_amendments: [{date, reason, criteria_hash}]` so the
   gate can read them without parsing prose. The current hash must equal
   the **latest** amendment's hash, or the registration hash when there are
   no amendments; without amendments it must also equal the hash recorded
   by the card's last trial.
3. **Holdout cutoff anchor.** Every trial records the `holdout_cutoff` it
   ran under, and the gate refuses any card whose cutoff differs from the
   one its trials recorded (`CUTOFF_MOVED`): the cutoff is frozen once
   trials ran. (A first version allowed moving it earlier — wrong, since
   that puts already-seen data into the holdout; fixed after review.)
4. **Ledger locking.** `append_trial` takes an exclusive `flock` on the
   ledger file itself and assigns the trial id under that lock. Step 3 must
   serialise ledger append plus research-repo commit with a **separate**
   lock file (e.g. `.register.lock`); locking `trials.jsonl` itself from the
   same process would deadlock against `append_trial`.
5. **`code.patch` module.** `src/fwbg/results/storage.py` is already over
   the 500-line limit, so the capture logic goes into its own module and
   `storage.py` gains only a two-line hook. Splitting `storage.py` is a
   separate change.

## Before you start

- Work in a new worktree on a topic branch:
  `git worktree add .worktrees/research-register-core -b feat/research-register-core origin/develop`
  and run every command from that directory.
- The shared virtualenv's editable install points at the primary checkout.
  Prefix every Python command with `PYTHONPATH=src` so the worktree's
  sources win:
  ```bash
  PY=../../.venv/bin/python
  PYTHONPATH=src $PY -c "import fwbg; print(fwbg.__file__)"   # must print a path inside the worktree
  ```
- Tests use `tmp_path`; nothing touches `~/fwbg`.

---

### Task 1: Package and config fingerprint

> **Amended after code review (done).** The code blocks below are the first
> iteration. Review showed that hashing the raw CLI dict depends on file
> location (`_strategy_dir`) and on preset *names* rather than content. The
> merged result differs: preset loading moved to `fwbg.core.presets`, which
> gained `PRESET_SECTIONS` and `resolve_presets()` (also used by
> `StrategyConfig.from_dict`); `fingerprint.py` excludes `_strategy_dir` and
> `_refs`, refuses unresolved preset names, hashes
> `{"strategy": …, "symbols": […]}` with `symbols: tuple[str, ...]`, and is
> pinned by a golden digest test. The source of truth is the code on the
> branch, not these blocks.

**Files:**
- Create: `src/fwbg/research/__init__.py`
- Create: `src/fwbg/research/fingerprint.py`
- Create: `tests/research/__init__.py` (empty)
- Test: `tests/research/test_fingerprint.py`

**Step 1: Write the failing tests**

`tests/research/test_fingerprint.py`:
```python
import pytest

from fwbg.research.fingerprint import canonical_json, fingerprint

BASE = {
    "name": "CUSUM",
    "description": "first try",
    "tags": ["event"],
    "timeframe": "DAY_1",
    "pipeline": {"indicators": [{"name": "cusum_events", "params": {"threshold": 1.5}}]},
    "cost_multiplier": 2.0,
}
SYMBOLS = ["GOLD", "EURUSD"]


def test_descriptive_keys_do_not_change_fingerprint():
    renamed = {
        **BASE,
        "name": "other",
        "description": "second try",
        "tags": ["x"],
        "hypothesis": "h",
        "resources": {"max_concurrent_assets": 4},
    }
    assert fingerprint(renamed, SYMBOLS) == fingerprint(BASE, SYMBOLS)


def test_parameter_change_changes_fingerprint():
    changed = {
        **BASE,
        "pipeline": {"indicators": [{"name": "cusum_events", "params": {"threshold": 2.0}}]},
    }
    assert fingerprint(changed, SYMBOLS) != fingerprint(BASE, SYMBOLS)


@pytest.mark.parametrize(
    "override",
    [
        {"end_date": "2025-04-30"},
        {"start_date": "2020-01-01"},
        {"cost_multiplier": 3.0},
        {"timeframe": "HOUR_1"},
        {"_cli_data_path": "/data/other"},
    ],
)
def test_run_overrides_change_fingerprint(override):
    assert fingerprint({**BASE, **override}, SYMBOLS) != fingerprint(BASE, SYMBOLS)


def test_key_order_is_irrelevant():
    reordered = dict(reversed(list(BASE.items())))
    assert fingerprint(reordered, SYMBOLS) == fingerprint(BASE, SYMBOLS)


def test_symbol_order_and_case_are_irrelevant():
    assert fingerprint(BASE, ["eurusd", "GOLD"]) == fingerprint(BASE, SYMBOLS)


def test_symbol_set_changes_fingerprint():
    assert fingerprint(BASE, ["GOLD"]) != fingerprint(BASE, SYMBOLS)


def test_integral_float_and_int_hash_alike():
    assert fingerprint({**BASE, "cost_multiplier": 2}, SYMBOLS) == fingerprint(BASE, SYMBOLS)


def test_bool_is_not_confused_with_int():
    assert canonical_json({"flag": True}) != canonical_json({"flag": 1})


def test_fingerprint_format():
    value = fingerprint(BASE, SYMBOLS)
    assert value.startswith("sha256:")
    assert len(value) == len("sha256:") + 64
```

**Step 2: Run to verify failure**

Run: `PYTHONPATH=src $PY -m pytest tests/research/test_fingerprint.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'fwbg.research'`.

**Step 3: Implement**

`src/fwbg/research/__init__.py`:
```python
"""Research register: research cards, trial ledger, config fingerprints and the pre-run gate."""
```

`src/fwbg/research/fingerprint.py`:
```python
"""Config fingerprints for the research register.

A fingerprint identifies what a run evaluated: the strategy metadata as the
CLI hands it to ``run_optimizer`` (and stores it as ``strategy.json``),
including CLI overrides, plus the resolved symbol list. Descriptive keys are
dropped so renaming or re-tagging a config does not make it look new.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

FINGERPRINT_VERSION = 1

DESCRIPTIVE_KEYS = frozenset(
    {
        "name",
        "description",
        "category",
        "tags",
        "hypothesis",
        "expected_outcome",
        "baseline_run",
        "changes",
        "resources",
    }
)
SYMBOLS_KEY = "__symbols__"


def _normalize(value: Any) -> Any:
    # JSON files write 2, the CLI passes 2.0 — both must hash alike.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize *value* deterministically: sorted keys, no whitespace."""
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value: Any) -> str:
    """Return ``sha256:<hex>`` over the canonical JSON of *value*."""
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_config(
    strategy_metadata: Mapping[str, Any], symbols: Iterable[str]
) -> dict[str, Any]:
    """Return the part of a run configuration that determines its result."""
    config = {
        key: value for key, value in strategy_metadata.items() if key not in DESCRIPTIVE_KEYS
    }
    config[SYMBOLS_KEY] = sorted({symbol.upper() for symbol in symbols})
    return config


def fingerprint(strategy_metadata: Mapping[str, Any], symbols: Iterable[str]) -> str:
    """Identical result-relevant configuration, identical fingerprint."""
    return digest(canonical_config(strategy_metadata, symbols))
```

**Step 4: Run to verify pass**

Run: `PYTHONPATH=src $PY -m pytest tests/research/test_fingerprint.py -q`
Expected: 13 passed.

**Step 5: Commit**

```bash
git add src/fwbg/research tests/research/__init__.py tests/research/test_fingerprint.py
git commit -m "feat(research): add config fingerprints"
```

---

### Task 2: Research cards

**Files:**
- Create: `src/fwbg/research/card.py`
- Test: `tests/research/test_card.py`

**Step 1: Write the failing tests**

`tests/research/test_card.py`:
```python
from datetime import date
from pathlib import Path

import pytest

from fwbg.research.card import CardError, Status, criteria_hash, load_card, parse_card

CARD = """\
---
id: S-0007
title: CUSUM event continuation
status: active
tags: [event, daily]
created: 2026-10-19
holdout_cutoff: "2025-04-30"
success_criteria:
  min_sharpe: 1.0
  min_trades: 100
registered_criteria_hash: sha256:5272a26ad425169031b39ae95c05abe1e2c3ebb664faaae0508eef108b989970
criteria_amendments:
  - date: 2026-10-20
    reason: daily needs fewer trades
    criteria_hash: sha256:c22558e72bca2f92a641914684ced8140937d9fe44d872db8f2526a71139f7e2
---
## Hypothesis
Structural breaks continue.
"""


def test_parses_front_matter_and_body():
    card = parse_card(CARD)
    assert card.id == "S-0007"
    assert card.status is Status.ACTIVE
    assert card.tags == ("event", "daily")
    assert card.created == date(2026, 10, 19)
    assert card.holdout_cutoff == date(2025, 4, 30)
    assert card.holdout_contaminated is False
    assert card.success_criteria == {"min_sharpe": 1.0, "min_trades": 100.0}
    assert card.registered_criteria_hash == (
        "sha256:5272a26ad425169031b39ae95c05abe1e2c3ebb664faaae0508eef108b989970"
    )
    assert card.criteria_amendments[0].criteria_hash == (
        "sha256:c22558e72bca2f92a641914684ced8140937d9fe44d872db8f2526a71139f7e2"
    )
    assert card.body.startswith("## Hypothesis")


@pytest.mark.parametrize(
    ("broken", "fragment"),
    [
        (CARD.replace("---\nid", "id", 1), "front matter"),
        (CARD.replace("title: CUSUM event continuation\n", ""), "title"),
        (CARD.replace("status: active", "status: maybe"), "maybe"),
        (CARD.replace('holdout_cutoff: "2025-04-30"', "holdout_cutoff: someday"), "holdout_cutoff"),
        (CARD.replace("  min_sharpe: 1.0\n  min_trades: 100\n", ""), "success_criteria"),
        (CARD.replace("min_trades: 100", "min_trades: yes"), "min_trades"),
        (CARD.replace("tags: [event, daily]", "tags: event"), "tags"),
        (CARD.replace("id: S-0007", "id: 7"), "invalid research id"),
    ],
)
def test_rejects_malformed_cards(broken, fragment):
    with pytest.raises(CardError, match=fragment):
        parse_card(broken)


def test_criteria_hash_ignores_key_order_but_not_values():
    assert criteria_hash({"a": 1.0, "b": 2.0}) == criteria_hash({"b": 2.0, "a": 1.0})
    assert criteria_hash({"a": 1.0}) != criteria_hash({"a": 1.5})


def write_card(research_dir: Path, filename: str, text: str = CARD) -> None:
    folder = research_dir / "strategies"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(text, encoding="utf-8")


def test_load_card_by_id(tmp_path):
    write_card(tmp_path, "S-0007-cusum-continuation.md")
    assert load_card(tmp_path, "S-0007").title == "CUSUM event continuation"


def test_load_card_returns_none_when_absent(tmp_path):
    write_card(tmp_path, "S-0007-cusum-continuation.md")
    assert load_card(tmp_path, "S-0008") is None


def test_load_card_does_not_match_longer_ids(tmp_path):
    write_card(tmp_path, "S-00071-other.md")
    assert load_card(tmp_path, "S-0007") is None


def test_load_card_refuses_ambiguous_ids(tmp_path):
    write_card(tmp_path, "S-0007-a.md")
    write_card(tmp_path, "S-0007-b.md")
    with pytest.raises(CardError, match="ambiguous"):
        load_card(tmp_path, "S-0007")


def test_load_card_refuses_mismatched_id(tmp_path):
    write_card(tmp_path, "S-0008-copy.md")
    with pytest.raises(CardError, match="declares id S-0007"):
        load_card(tmp_path, "S-0008")


def test_load_card_refuses_malformed_id(tmp_path):
    with pytest.raises(CardError, match="invalid research id"):
        load_card(tmp_path, "../S-0007")
```

**Step 2: Run to verify failure**

Run: `PYTHONPATH=src $PY -m pytest tests/research/test_card.py -q`
Expected: collection error, `No module named 'fwbg.research.card'`.

**Step 3: Implement**

`src/fwbg/research/card.py`:
```python
"""Research cards: one Markdown file with YAML front matter per research strategy."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path

import yaml

from fwbg.research.fingerprint import digest

CARD_ID_RE = re.compile(r"S-\d{4}")
_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---[ \t]*\n?(.*)\Z", re.DOTALL)


class CardError(Exception):
    """A research card is malformed, ambiguous or does not match its file."""


class Status(str, Enum):
    ACTIVE = "active"
    PARKED = "parked"
    REJECTED = "rejected"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class CriteriaAmendment:
    date: date
    reason: str
    criteria_hash: str


@dataclass(frozen=True)
class ResearchCard:
    id: str
    title: str
    status: Status
    tags: tuple[str, ...]
    created: date
    holdout_cutoff: date
    holdout_contaminated: bool
    success_criteria: Mapping[str, float]
    registered_criteria_hash: str
    criteria_amendments: tuple[CriteriaAmendment, ...]
    body: str

    @property
    def criteria_hash(self) -> str:
        return criteria_hash(self.success_criteria)


def criteria_hash(criteria: Mapping[str, float]) -> str:
    """Hash of a set of success criteria, independent of key order."""
    return digest(dict(criteria))


def parse_card(text: str) -> ResearchCard:
    """Parse a card file; raise :class:`CardError` on any defect."""
    match = _FRONT_MATTER_RE.match(text)
    if match is None:
        raise CardError("card has no YAML front matter")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise CardError(f"invalid YAML front matter: {exc}") from exc
    if not isinstance(meta, Mapping):
        raise CardError("front matter must be a mapping")
    try:
        return ResearchCard(
            id=_card_id(meta["id"]),
            title=str(meta["title"]),
            status=Status(meta["status"]),
            tags=_tags(meta.get("tags")),
            created=_as_date(meta["created"], "created"),
            holdout_cutoff=_as_date(meta["holdout_cutoff"], "holdout_cutoff"),
            holdout_contaminated=bool(meta.get("holdout_contaminated", False)),
            success_criteria=_criteria(meta["success_criteria"]),
            registered_criteria_hash=str(meta["registered_criteria_hash"]),
            criteria_amendments=tuple(
                _amendment(item) for item in meta.get("criteria_amendments") or ()
            ),
            body=match.group(2),
        )
    except KeyError as exc:
        raise CardError(f"missing field: {exc.args[0]}") from exc
    except ValueError as exc:
        raise CardError(str(exc)) from exc


def load_card(research_dir: Path, card_id: str) -> ResearchCard | None:
    """Load card *card_id* from ``strategies/``; ``None`` if no file exists for it."""
    _card_id(card_id)
    folder = research_dir / "strategies"
    paths = sorted({*folder.glob(f"{card_id}-*.md"), *folder.glob(f"{card_id}.md")})
    if not paths:
        return None
    if len(paths) > 1:
        raise CardError(f"ambiguous research id {card_id}: {[p.name for p in paths]}")
    card = parse_card(paths[0].read_text(encoding="utf-8"))
    if card.id != card_id:
        raise CardError(f"{paths[0].name} declares id {card.id}, expected {card_id}")
    return card


def _card_id(value: object) -> str:
    if not isinstance(value, str) or not CARD_ID_RE.fullmatch(value):
        raise CardError(f"invalid research id {value!r} (expected S-NNNN)")
    return value


def _tags(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CardError(f"tags must be a list, got {value!r}")
    return tuple(str(tag) for tag in value)


def _as_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise CardError(f"{field_name}: not an ISO date: {value!r}") from exc
    raise CardError(f"{field_name}: expected a date, got {value!r}")


def _criteria(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise CardError("success_criteria must be a non-empty mapping")
    criteria: dict[str, float] = {}
    for key, threshold in value.items():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise CardError(f"success_criteria.{key}: expected a number, got {threshold!r}")
        criteria[str(key)] = float(threshold)
    return criteria


def _amendment(value: object) -> CriteriaAmendment:
    if not isinstance(value, Mapping):
        raise CardError(f"criteria_amendments: expected a mapping, got {value!r}")
    try:
        return CriteriaAmendment(
            date=_as_date(value["date"], "criteria_amendments.date"),
            reason=str(value["reason"]),
            criteria_hash=str(value["criteria_hash"]),
        )
    except KeyError as exc:
        raise CardError(f"criteria_amendments: missing {exc.args[0]}") from exc
```

Note: `CardError` deliberately does not subclass `ValueError`, so the
`except ValueError` in `parse_card` only wraps foreign errors (e.g. an
unknown `Status`) and lets `CardError` from the helpers pass through.

**Step 4: Run to verify pass**

Run: `PYTHONPATH=src $PY -m pytest tests/research/test_card.py -q`
Expected: 16 passed.

**Step 5: Commit**

```bash
git add src/fwbg/research/card.py tests/research/test_card.py
git commit -m "feat(research): parse and load research cards"
```

---

### Task 3: Trial ledger

> **Amended after code review (done).** The code blocks below are the first
> iteration. Review found that `ensure_ascii=False` plus `str.splitlines()`
> let a note containing U+2028/U+2029/NEL make the ledger permanently
> unreadable, and that most record fields were not validated on read. The
> merged result writes pure-ASCII lines and splits on `"\n"` only, refuses a
> trial that would not read back unchanged before writing, validates every
> field of every stored line (exact key set, `T-NNNNNN` ids unique, card id,
> `DIGEST_RE` hashes shared with `card.py`, finite metrics, no NaN literals),
> turns decoding errors and torn last lines into `LedgerError`, and treats a
> **missing ledger as an error** (appends never create the file;
> `init_research_repo` does). The source of truth is the code on the branch.

**Files:**
- Create: `src/fwbg/research/ledger.py`
- Create: `tests/research/factories.py`
- Test: `tests/research/test_ledger.py`

**Step 1: Write the test factories and failing tests**

`tests/research/factories.py`:
```python
"""Builders for research-register test data."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from fwbg.research.card import ResearchCard, Status, criteria_hash
from fwbg.research.ledger import Trial, TrialKind, TrialStatus, append_trial


def make_card(**overrides: Any) -> ResearchCard:
    values: dict[str, Any] = {
        "id": "S-0007",
        "title": "CUSUM event continuation",
        "status": Status.ACTIVE,
        "tags": ("event",),
        "created": date(2026, 10, 19),
        "holdout_cutoff": date(2025, 4, 30),
        "holdout_contaminated": False,
        "success_criteria": {"min_sharpe": 1.0, "min_trades": 100.0},
        "registered_criteria_hash": criteria_hash({"min_sharpe": 1.0, "min_trades": 100.0}),
        "criteria_amendments": (),
        "body": "",
    }
    values.update(overrides)
    return ResearchCard(**values)


def make_trial(**overrides: Any) -> Trial:
    values: dict[str, Any] = {
        "strategy": "S-0007",
        "run_id": "run-1",
        "kind": TrialKind.ITERATION,
        "status": TrialStatus.OK,
        "ts": "2026-10-19T10:00:00",
        "fingerprint": "sha256:" + "0" * 64,
        "fingerprint_version": 1,
        "criteria_hash": make_card().criteria_hash,
        "holdout_cutoff": date(2025, 4, 30),
        "code": {"commit": "abc123", "dirty": False, "patch": None},
        "window": {"start": None, "end": "2025-04-30"},
        "note": "baseline",
    }
    values.update(overrides)
    return Trial(**values)


def append_many(ledger: str, worker: int, count: int) -> None:
    """Process target for concurrency tests: append *count* trials."""
    for index in range(count):
        append_trial(Path(ledger), make_trial(run_id=f"w{worker}-{index}"))
```

`append_many` lives here rather than in the test module because the
concurrency test uses the `spawn` start method, whose child processes must
import the target by module path. `fork` would warn (and can deadlock) in
pytest's multi-threaded process.

`tests/research/test_ledger.py`:
```python
import multiprocessing

import pytest

from fwbg.research.ledger import LedgerError, TrialKind, TrialStatus, append_trial, read_trials
from tests.research.factories import append_many, make_trial


def test_missing_ledger_has_no_trials(tmp_path):
    assert read_trials(tmp_path / "trials.jsonl") == []


def test_append_assigns_sequential_ids_and_round_trips(tmp_path):
    ledger = tmp_path / "trials.jsonl"
    first = append_trial(ledger, make_trial(metrics={"GOLD": {"sharpe": 0.14, "trades": 37}}))
    second = append_trial(ledger, make_trial(run_id="run-2"))
    assert (first.trial_id, second.trial_id) == ("T-000001", "T-000002")
    assert read_trials(ledger) == [first, second]


def test_append_refuses_trials_that_already_have_an_id(tmp_path):
    with pytest.raises(ValueError, match="assigns trial ids"):
        append_trial(tmp_path / "trials.jsonl", make_trial(trial_id="T-000009"))


def test_malformed_line_is_reported_with_its_number(tmp_path):
    ledger = tmp_path / "trials.jsonl"
    append_trial(ledger, make_trial())
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    with pytest.raises(LedgerError, match="line 2"):
        read_trials(ledger)


def test_malformed_holdout_cutoff_is_a_ledger_error(tmp_path):
    ledger = tmp_path / "trials.jsonl"
    line = make_trial(trial_id="T-000001").to_json().replace("2025-04-30", "someday", 1)
    ledger.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="line 1"):
        read_trials(ledger)


def test_append_repairs_missing_trailing_newline(tmp_path):
    ledger = tmp_path / "trials.jsonl"
    append_trial(ledger, make_trial())
    ledger.write_text(ledger.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    append_trial(ledger, make_trial(run_id="run-2"))
    assert [t.trial_id for t in read_trials(ledger)] == ["T-000001", "T-000002"]


@pytest.mark.parametrize(
    ("kind", "status", "counted"),
    [
        (TrialKind.ITERATION, TrialStatus.OK, True),
        (TrialKind.HOLDOUT, TrialStatus.OK, True),
        (TrialKind.LEGACY, TrialStatus.OK, True),
        (TrialKind.ITERATION, TrialStatus.FAILED, False),
        (TrialKind.SMOKE, TrialStatus.OK, False),
    ],
)
def test_counted_trials(kind, status, counted):
    assert make_trial(kind=kind, status=status).counted is counted


def test_concurrent_appends_lose_no_lines(tmp_path):
    ledger = tmp_path / "trials.jsonl"
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(target=append_many, args=(str(ledger), worker, 25)) for worker in range(4)
    ]
    for process in workers:
        process.start()
    for process in workers:
        process.join(timeout=60)
    assert [process.exitcode for process in workers] == [0, 0, 0, 0]
    trials = read_trials(ledger)
    assert len(trials) == 100
    assert len({t.trial_id for t in trials}) == 100
    assert len({t.run_id for t in trials}) == 100
```

**Step 2: Run to verify failure**

Run: `PYTHONPATH=src $PY -m pytest tests/research/test_ledger.py -q`
Expected: collection error, `No module named 'fwbg.research.ledger'`.

**Step 3: Implement**

`src/fwbg/research/ledger.py`:
```python
"""Append-only trial ledger (``trials.jsonl``) of the research register."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

LEDGER_FILENAME = "trials.jsonl"
TRIAL_ID_PREFIX = "T-"

Metrics = Mapping[str, Mapping[str, float | int | None]]


class LedgerError(Exception):
    """The ledger contains a line that is not a valid trial."""


class TrialKind(str, Enum):
    ITERATION = "iteration"
    HOLDOUT = "holdout"
    LEGACY = "legacy"
    SMOKE = "smoke"


class TrialStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"


@dataclass(frozen=True)
class Trial:
    """One evaluation; ``trial_id`` is assigned by :func:`append_trial`."""

    strategy: str
    run_id: str | None
    kind: TrialKind
    status: TrialStatus
    ts: str
    fingerprint: str
    fingerprint_version: int
    criteria_hash: str
    holdout_cutoff: date
    code: Mapping[str, Any]
    window: Mapping[str, str | None]
    note: str | None
    reason: str | None = None
    metrics: Metrics = field(default_factory=dict)
    trial_id: str | None = None

    @property
    def counted(self) -> bool:
        """Whether this trial counts towards N in multiple-testing corrections."""
        return self.status is TrialStatus.OK and self.kind is not TrialKind.SMOKE

    def to_json(self) -> str:
        record = {
            "trial": self.trial_id,
            "strategy": self.strategy,
            "run_id": self.run_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "ts": self.ts,
            "fingerprint": self.fingerprint,
            "fingerprint_version": self.fingerprint_version,
            "criteria_hash": self.criteria_hash,
            "holdout_cutoff": self.holdout_cutoff.isoformat(),
            "code": dict(self.code),
            "window": dict(self.window),
            "note": self.note,
            "reason": self.reason,
            "metrics": {symbol: dict(values) for symbol, values in self.metrics.items()},
        }
        return json.dumps(record, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, line: str) -> Trial:
        record = json.loads(line)
        return cls(
            trial_id=record["trial"],
            strategy=record["strategy"],
            run_id=record["run_id"],
            kind=TrialKind(record["kind"]),
            status=TrialStatus(record["status"]),
            ts=record["ts"],
            fingerprint=record["fingerprint"],
            fingerprint_version=record["fingerprint_version"],
            criteria_hash=record["criteria_hash"],
            holdout_cutoff=date.fromisoformat(record["holdout_cutoff"]),
            code=record["code"],
            window=record["window"],
            note=record["note"],
            reason=record.get("reason"),
            metrics=record.get("metrics", {}),
        )


def read_trials(ledger_path: Path) -> list[Trial]:
    """All trials in ledger order; a ledger that does not exist yet has none."""
    if not ledger_path.exists():
        return []
    with ledger_path.open(encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        return _parse(handle.read())


def append_trial(ledger_path: Path, trial: Trial) -> Trial:
    """Assign the next trial id and append *trial*; safe across processes."""
    if trial.trial_id is not None:
        raise ValueError("append_trial assigns trial ids; got a trial that already has one")
    # ponytail: re-reads the whole ledger under one exclusive lock to find the
    # next id — O(n) per append, fine for thousands of trials; switch to a
    # counter file if the ledger grows far beyond that.
    with ledger_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        text = handle.read()
        stored = replace(trial, trial_id=_next_id(_parse(text)))
        if text and not text.endswith("\n"):
            handle.write("\n")
        handle.write(stored.to_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return stored


def _parse(text: str) -> list[Trial]:
    trials = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            trials.append(Trial.from_json(line))
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerError(f"{LEDGER_FILENAME} line {number}: {exc!r}") from exc
    return trials


def _next_id(trials: list[Trial]) -> str:
    numbers = [int(t.trial_id.removeprefix(TRIAL_ID_PREFIX)) for t in trials if t.trial_id]
    return f"{TRIAL_ID_PREFIX}{max(numbers, default=0) + 1:06d}"
```

`append_trial` releases the lock when the `with` block closes the file.
Metrics must not contain NaN (`allow_nan=False`); step 3 converts NaN to
`None` when it extracts metrics from `results.json`.

**Step 4: Run to verify pass**

Run: `PYTHONPATH=src $PY -m pytest tests/research/test_ledger.py -q`
Expected: 12 passed.

**Step 5: Commit**

```bash
git add src/fwbg/research/ledger.py tests/research/factories.py tests/research/test_ledger.py
git commit -m "feat(research): add append-only trial ledger"
```

---

### Task 4: Pre-run gate

> **Amended after code review (done).** The code blocks below are the first
> iteration. Review found fail-open gaps: the cutoff rule ran the wrong way,
> `end_date` was checked on its first ten characters only, any historical
> amendment satisfied the criteria check, and most behaviour had no pinning
> test. The merged gate freezes the cutoff, requires strict `YYYY-MM-DD`
> windows (start < end; holdout runs must end after the cutoff), accepts only
> the latest amendment, counts every holdout trial as a look, denies holdout
> runs on contaminated cards (`HOLDOUT_CONTAMINATED`), turns unhashable
> configs into `INVALID_CONFIG` (422) and checks card/request id agreement.
> The source of truth is the code on the branch.

**Files:**
- Create: `src/fwbg/research/gate.py`
- Test: `tests/research/test_gate.py`

**Step 1: Write the failing tests**

`tests/research/test_gate.py`:
```python
from datetime import date

import pytest

from fwbg.research.card import CriteriaAmendment, Status, criteria_hash
from fwbg.research.fingerprint import fingerprint
from fwbg.research.gate import DenialCode, RunRequest, check
from fwbg.research.ledger import TrialKind, TrialStatus
from tests.research.factories import make_card, make_trial

METADATA = {"name": "cusum", "timeframe": "DAY_1", "pipeline": {"indicators": []}}
SYMBOLS = ("GOLD",)
OTHER_FINGERPRINT = "sha256:" + "1" * 64


def request(**overrides) -> RunRequest:
    values = {
        "research_id": "S-0007",
        "strategy_metadata": METADATA,
        "symbols": SYMBOLS,
        "note": "baseline",
    }
    values.update(overrides)
    return RunRequest(**values)


def codes(decision) -> set[DenialCode]:
    return {denial.code for denial in decision.denials}


def test_allows_first_iteration_and_clamps_end_date_to_cutoff():
    decision = check(make_card(), request(), [])
    assert decision.allowed
    assert decision.kind is TrialKind.ITERATION
    assert decision.effective_metadata["end_date"] == "2025-04-30"
    assert decision.fingerprint == fingerprint(decision.effective_metadata, SYMBOLS)
    assert "end_date" not in METADATA


def test_missing_research_id_is_a_request_error():
    decision = check(None, request(research_id=None), [])
    assert codes(decision) == {DenialCode.MISSING_RESEARCH_ID}
    assert decision.denials[0].code.http_status == 422
    assert decision.fingerprint is None


def test_unknown_card_is_a_request_error():
    assert codes(check(None, request(), [])) == {DenialCode.CARD_NOT_FOUND}


@pytest.mark.parametrize("note", [None, "", "   "])
def test_note_is_required(note):
    assert codes(check(make_card(), request(note=note), [])) == {DenialCode.MISSING_NOTE}


@pytest.mark.parametrize("status", [Status.PARKED, Status.REJECTED, Status.CANDIDATE])
def test_only_active_cards_accept_runs(status):
    decision = check(make_card(status=status), request(), [])
    assert codes(decision) == {DenialCode.NOT_ACTIVE}
    assert decision.denials[0].code.http_status == 409


def test_collects_all_denials():
    decision = check(make_card(status=Status.PARKED), request(note=""), [])
    assert codes(decision) == {DenialCode.NOT_ACTIVE, DenialCode.MISSING_NOTE}


def test_end_date_before_cutoff_is_kept():
    metadata = {**METADATA, "end_date": "2024-12-31"}
    decision = check(make_card(), request(strategy_metadata=metadata), [])
    assert decision.allowed
    assert decision.effective_metadata["end_date"] == "2024-12-31"


def test_end_date_after_cutoff_is_refused():
    metadata = {**METADATA, "end_date": "2025-05-01"}
    assert codes(check(make_card(), request(strategy_metadata=metadata), [])) == {
        DenialCode.AFTER_CUTOFF
    }


def test_malformed_end_date_is_a_request_error():
    metadata = {**METADATA, "end_date": "soon"}
    decision = check(make_card(), request(strategy_metadata=metadata), [])
    assert codes(decision) == {DenialCode.INVALID_WINDOW}
    assert decision.denials[0].code.http_status == 422


def test_first_holdout_look_is_free_and_unclamped():
    decision = check(make_card(), request(holdout=True), [])
    assert decision.allowed
    assert decision.kind is TrialKind.HOLDOUT
    assert "end_date" not in decision.effective_metadata


def test_second_holdout_look_needs_a_reason():
    used = [make_trial(trial_id="T-000003", kind=TrialKind.HOLDOUT, fingerprint=OTHER_FINGERPRINT)]
    assert codes(check(make_card(), request(holdout=True), used)) == {DenialCode.HOLDOUT_REPEAT}
    assert check(make_card(), request(holdout=True, reason="exit bug fixed"), used).allowed


def test_duplicate_fingerprint_is_refused_with_pointer_to_earlier_trial():
    first = check(make_card(), request(), [])
    earlier = [make_trial(trial_id="T-000001", fingerprint=first.fingerprint)]
    decision = check(make_card(), request(), earlier)
    assert codes(decision) == {DenialCode.DUPLICATE_FINGERPRINT}
    assert "T-000001" in decision.denials[0].message


def test_duplicate_under_another_research_id_is_refused_too():
    first = check(make_card(), request(), [])
    earlier = [make_trial(strategy="S-0001", trial_id="T-000001", fingerprint=first.fingerprint)]
    assert codes(check(make_card(), request(), earlier)) == {DenialCode.DUPLICATE_FINGERPRINT}


def test_reason_allows_rerunning_an_identical_config():
    first = check(make_card(), request(), [])
    earlier = [make_trial(trial_id="T-000001", fingerprint=first.fingerprint)]
    assert check(make_card(), request(reason="simulator bugfix"), earlier).allowed


def test_failed_trial_does_not_block_a_rerun():
    first = check(make_card(), request(), [])
    earlier = [
        make_trial(trial_id="T-000001", fingerprint=first.fingerprint, status=TrialStatus.FAILED)
    ]
    assert check(make_card(), request(), earlier).allowed


def test_changed_criteria_without_amendment_are_refused():
    changed = make_card(success_criteria={"min_sharpe": 0.5, "min_trades": 100.0})
    trials = [make_trial(trial_id="T-000001")]
    assert codes(check(changed, request(), trials)) == {DenialCode.CRITERIA_CHANGED}


def test_amended_criteria_are_accepted():
    new_criteria = {"min_sharpe": 0.5, "min_trades": 100.0}
    amendment = CriteriaAmendment(date(2026, 10, 20), "too strict", criteria_hash(new_criteria))
    amended = make_card(success_criteria=new_criteria, criteria_amendments=(amendment,))
    assert check(amended, request(), [make_trial(trial_id="T-000001")]).allowed


def test_first_trial_uses_the_registered_criteria_hash():
    changed = make_card(registered_criteria_hash="sha256:other")
    assert codes(check(changed, request(), [])) == {DenialCode.CRITERIA_CHANGED}


def test_amended_criteria_are_accepted_before_the_first_trial():
    new_criteria = {"min_sharpe": 0.5, "min_trades": 100.0}
    amendment = CriteriaAmendment(date(2026, 10, 20), "too strict", criteria_hash(new_criteria))
    amended = make_card(
        success_criteria=new_criteria,
        registered_criteria_hash="sha256:old",
        criteria_amendments=(amendment,),
    )
    assert check(amended, request(), []).allowed


def test_moving_the_cutoff_later_after_trials_is_refused():
    moved = make_card(holdout_cutoff=date(2025, 6, 30))
    assert codes(check(moved, request(), [make_trial(trial_id="T-000001")])) == {
        DenialCode.CUTOFF_MOVED
    }


def test_moving_the_cutoff_earlier_is_allowed():
    moved = make_card(holdout_cutoff=date(2025, 1, 31))
    assert check(moved, request(), [make_trial(trial_id="T-000001")]).allowed


def test_note_must_be_one_line():
    assert codes(check(make_card(), request(note="baseline\nextra"), [])) == {
        DenialCode.MISSING_NOTE
    }


def test_blank_reason_does_not_bypass_duplicate_protection():
    first = check(make_card(), request(), [])
    earlier = [make_trial(trial_id="T-000001", fingerprint=first.fingerprint)]
    assert codes(check(make_card(), request(reason=" "), earlier)) == {
        DenialCode.DUPLICATE_FINGERPRINT
    }
```

**Step 2: Run to verify failure**

Run: `PYTHONPATH=src $PY -m pytest tests/research/test_gate.py -q`
Expected: collection error, `No module named 'fwbg.research.gate'`.

**Step 3: Implement**

`src/fwbg/research/gate.py`:
```python
"""Pre-run gate of the research register.

``check`` is pure: the caller loads the card and the ledger, calls ``check``,
and starts the run with ``Decision.effective_metadata`` only if
``Decision.allowed``. The CLI run path and ``POST /api/runs/start`` share it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

from fwbg.research.card import ResearchCard, Status
from fwbg.research.fingerprint import fingerprint
from fwbg.research.ledger import Trial, TrialKind, TrialStatus


class DenialCode(str, Enum):
    MISSING_RESEARCH_ID = "missing_research_id"
    CARD_NOT_FOUND = "card_not_found"
    MISSING_NOTE = "missing_note"
    INVALID_WINDOW = "invalid_window"
    NOT_ACTIVE = "not_active"
    CRITERIA_CHANGED = "criteria_changed"
    CUTOFF_MOVED = "cutoff_moved"
    AFTER_CUTOFF = "after_cutoff"
    HOLDOUT_REPEAT = "holdout_repeat"
    DUPLICATE_FINGERPRINT = "duplicate_fingerprint"

    @property
    def http_status(self) -> int:
        """422 for an incomplete or malformed request, 409 for a conflict with the register."""
        return 422 if self in _REQUEST_ERRORS else 409


_REQUEST_ERRORS = frozenset(
    {
        DenialCode.MISSING_RESEARCH_ID,
        DenialCode.CARD_NOT_FOUND,
        DenialCode.MISSING_NOTE,
        DenialCode.INVALID_WINDOW,
    }
)


@dataclass(frozen=True)
class Denial:
    code: DenialCode
    message: str


@dataclass(frozen=True)
class RunRequest:
    """A run after overrides, preset resolution and canonical asset resolution.

    ``strategy_metadata`` must already be passed through
    ``fwbg.core.presets.resolve_presets`` (the fingerprint refuses unresolved
    preset names). ``symbols`` contains the uppercase, sorted concrete symbols
    produced by ``AssetRegistry``. CLI/API wiring must resolve asset classes
    before constructing this request; unresolved names never reach the gate.
    """

    research_id: str | None
    strategy_metadata: Mapping[str, Any]
    symbols: tuple[str, ...]
    note: str | None
    reason: str | None = None
    holdout: bool = False


@dataclass(frozen=True)
class Decision:
    denials: tuple[Denial, ...]
    kind: TrialKind
    effective_metadata: Mapping[str, Any]
    fingerprint: str | None

    @property
    def allowed(self) -> bool:
        return not self.denials


def check(card: ResearchCard | None, request: RunRequest, trials: Sequence[Trial]) -> Decision:
    """Decide whether *request* may run, given its card and the whole ledger."""
    kind = TrialKind.HOLDOUT if request.holdout else TrialKind.ITERATION
    metadata = dict(request.strategy_metadata)
    if not request.research_id:
        return _refuse(
            DenialCode.MISSING_RESEARCH_ID, "every run needs a research id (S-NNNN)", kind, metadata
        )
    if card is None:
        return _refuse(
            DenialCode.CARD_NOT_FOUND, f"no research card {request.research_id}", kind, metadata
        )

    own_trials = [trial for trial in trials if trial.strategy == card.id]
    denials: list[Denial] = []
    if not _valid_note(request.note):
        denials.append(
            Denial(
                DenialCode.MISSING_NOTE,
                "every run needs a non-empty one-line note: what changed and why",
            )
        )
    if card.status is not Status.ACTIVE:
        denials.append(
            Denial(
                DenialCode.NOT_ACTIVE,
                f"{card.id} is {card.status.value}; reopen it with a reason before running",
            )
        )
    denials.extend(_criteria_denials(card, own_trials))
    denials.extend(_cutoff_denials(card, own_trials))
    window_denial = _apply_window(card, metadata, request.holdout)
    if window_denial is not None:
        denials.append(window_denial)
    has_reason = _has_text(request.reason)
    if request.holdout and not has_reason:
        denials.extend(_holdout_denials(card, own_trials))
    run_fingerprint = fingerprint(metadata, request.symbols)
    if not has_reason:
        denials.extend(_duplicate_denials(run_fingerprint, trials))
    return Decision(tuple(denials), kind, metadata, run_fingerprint)


def _has_text(value: str | None) -> bool:
    return bool(value and value.strip())


def _valid_note(note: str | None) -> bool:
    return note is not None and _has_text(note) and "\n" not in note and "\r" not in note


def _refuse(code: DenialCode, message: str, kind: TrialKind, metadata: dict[str, Any]) -> Decision:
    return Decision((Denial(code, message),), kind, metadata, None)


def _criteria_denials(card: ResearchCard, own_trials: Sequence[Trial]) -> list[Denial]:
    current = card.criteria_hash
    if not own_trials:
        if current == card.registered_criteria_hash:
            return []
        if any(amendment.criteria_hash == current for amendment in card.criteria_amendments):
            return []
        return [
            Denial(
                DenialCode.CRITERIA_CHANGED,
                f"success criteria of {card.id} differ from the registration baseline "
                "and no amendment records the change",
            )
        ]
    last = own_trials[-1]
    if current == last.criteria_hash:
        return []
    if any(amendment.criteria_hash == current for amendment in card.criteria_amendments):
        return []
    return [
        Denial(
            DenialCode.CRITERIA_CHANGED,
            f"success criteria of {card.id} differ from those of {last.trial_id} "
            "and no amendment records the change",
        )
    ]


def _cutoff_denials(card: ResearchCard, own_trials: Sequence[Trial]) -> list[Denial]:
    """The holdout may shrink after trials ran, never grow into data they could not see."""
    if not own_trials:
        return []
    earliest = min(trial.holdout_cutoff for trial in own_trials)
    if card.holdout_cutoff <= earliest:
        return []
    return [
        Denial(
            DenialCode.CUTOFF_MOVED,
            f"holdout cutoff of {card.id} moved to {card.holdout_cutoff} after trials ran "
            f"with {earliest}; it may only move earlier",
        )
    ]


def _apply_window(card: ResearchCard, metadata: dict[str, Any], holdout: bool) -> Denial | None:
    """Clamp an open-ended iteration to the holdout cutoff; refuse windows past it."""
    end = metadata.get("end_date")
    if end is None:
        if not holdout:
            metadata["end_date"] = card.holdout_cutoff.isoformat()
        return None
    try:
        end_day = date.fromisoformat(str(end)[:10])
    except ValueError:
        return Denial(DenialCode.INVALID_WINDOW, f"end_date {end!r} is not an ISO date")
    if end_day > card.holdout_cutoff and not holdout:
        return Denial(
            DenialCode.AFTER_CUTOFF,
            f"end_date {end_day} lies after the holdout cutoff {card.holdout_cutoff} "
            f"of {card.id}; only a holdout run may see that data",
        )
    return None


def _holdout_denials(card: ResearchCard, own_trials: Sequence[Trial]) -> list[Denial]:
    used = [
        str(trial.trial_id)
        for trial in own_trials
        if trial.kind is TrialKind.HOLDOUT and trial.status is TrialStatus.OK
    ]
    if not used:
        return []
    return [
        Denial(
            DenialCode.HOLDOUT_REPEAT,
            f"the holdout of {card.id} was already used ({', '.join(used)}); "
            "a further look needs a reason",
        )
    ]


def _duplicate_denials(run_fingerprint: str, trials: Sequence[Trial]) -> list[Denial]:
    earlier = [
        trial
        for trial in trials
        if trial.fingerprint == run_fingerprint and trial.status is TrialStatus.OK
    ]
    if not earlier:
        return []
    listed = ", ".join(
        f"{trial.trial_id} ({trial.strategy}, run {trial.run_id}, {trial.ts[:10]})"
        for trial in earlier
    )
    return [
        Denial(
            DenialCode.DUPLICATE_FINGERPRINT,
            f"identical configuration already ran: {listed}; pass a reason to run it again",
        )
    ]
```

**Step 4: Run to verify pass**

Run: `PYTHONPATH=src $PY -m pytest tests/research/test_gate.py -q`
Expected: 27 passed.

**Step 5: Commit**

```bash
git add src/fwbg/research/gate.py tests/research/test_gate.py
git commit -m "feat(research): add pre-run gate"
```

---

### Task 5: Research directory and repository

> **Amended after code review (done).** The code blocks below are the first
> iteration. Review found that a failed first commit left the register
> "initialised" forever, that re-running init silently recreated a deleted
> ledger (resetting the trial history), that `git add --all` committed
> whatever already sat in the directory, and that inherited `GIT_DIR` /
> `GIT_INDEX_FILE` (exported by git hooks) made init write into another
> repository. The merged `init_research_repo` decides by "has a commit",
> retries a failed first commit, refuses to recreate a missing ledger in an
> initialised register, stages only `.gitignore`, the template and
> `trials.jsonl`, uses `--initial-branch=main`, and strips git's
> repository-local environment variables (`GIT_LOCAL_ENV_VARS`) from every
> git call; the `git_env` fixture strips them too. The source of truth is
> the code on the branch.

**Files:**
- Modify: `src/fwbg/api/workspace.py` (module docstring list; new `get_research_dir()` after `get_test_results_dir()`)
- Create: `src/fwbg/research/repo.py`
- Modify: `tests/conftest.py` (add `git_env` fixture)
- Test: `tests/research/test_repo.py`

`get_research_dir()` belongs to the workspace module (edge), not to
`fwbg.research` (domain): research functions take the directory as a
parameter. It is deliberately not added to `init_workspace()`; the register
is created explicitly by `init_research_repo`.

**Step 1: Add the shared git fixture**

Append to `tests/conftest.py` (add `import pytest` to the imports):
```python
@pytest.fixture
def git_env(tmp_path_factory, monkeypatch):
    """Deterministic git: fixed identity, no user or system configuration."""
    config = tmp_path_factory.mktemp("gitconfig") / "config"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for role in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{role}_NAME", "fwbg tests")
        monkeypatch.setenv(f"GIT_{role}_EMAIL", "tests@fwbg.invalid")
```

**Step 2: Write the failing tests**

`tests/research/test_repo.py`:
```python
import subprocess

import pytest
import yaml

from fwbg.api.workspace import get_research_dir
from fwbg.research.ledger import LEDGER_FILENAME
from fwbg.research.repo import (
    CRITERIA_TEMPLATE,
    DEFAULT_CRITERIA_YAML,
    ResearchRepoError,
    init_research_repo,
)


def commit_count(path) -> int:
    result = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    )
    return int(result.stdout.strip())


def test_init_creates_layout_template_ledger_and_repo(tmp_path, git_env):
    research = tmp_path / "research"
    assert init_research_repo(research) is True
    for name in ("strategies", "configs", "templates", "reports"):
        assert (research / name).is_dir()
    assert (research / CRITERIA_TEMPLATE).read_text(encoding="utf-8") == DEFAULT_CRITERIA_YAML
    assert (research / LEDGER_FILENAME).read_text(encoding="utf-8") == ""
    assert commit_count(research) == 1


def test_init_is_idempotent_and_keeps_edits(tmp_path, git_env):
    research = tmp_path / "research"
    init_research_repo(research)
    (research / CRITERIA_TEMPLATE).write_text("min_sharpe: 2.0\n", encoding="utf-8")
    assert init_research_repo(research) is False
    assert (research / CRITERIA_TEMPLATE).read_text(encoding="utf-8") == "min_sharpe: 2.0\n"
    assert commit_count(research) == 1


def test_init_without_git_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "no-binaries"))
    with pytest.raises(ResearchRepoError, match="git init"):
        init_research_repo(tmp_path / "research")


def test_default_criteria_are_numbers():
    criteria = yaml.safe_load(DEFAULT_CRITERIA_YAML)
    assert criteria
    assert all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in criteria.values()
    )


def test_research_dir_defaults_to_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("FWBG_RESEARCH_DIR", raising=False)
    monkeypatch.setenv("FWBG_WORKSPACE", str(tmp_path))
    assert get_research_dir() == tmp_path / "research"


def test_research_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(tmp_path / "elsewhere"))
    assert get_research_dir() == tmp_path / "elsewhere"
```

**Step 3: Run to verify failure**

Run: `PYTHONPATH=src $PY -m pytest tests/research/test_repo.py -q`
Expected: collection error, `cannot import name 'get_research_dir'`.

**Step 4: Implement**

In `src/fwbg/api/workspace.py`, add to the module docstring list:
```
  FWBG_RESEARCH_DIR     → overrides {workspace}/research
```
and after `get_test_results_dir()`:
```python
def get_research_dir() -> Path:
    """Return the research register directory (a private git repository)."""
    return _workspace_subdir("FWBG_RESEARCH_DIR", "research")
```

`src/fwbg/research/repo.py`:
```python
"""Layout and git repository of the research register."""

from __future__ import annotations

import subprocess
from pathlib import Path

from fwbg.research.ledger import LEDGER_FILENAME

CRITERIA_TEMPLATE = Path("templates") / "criteria-default.yaml"
_DIRECTORIES = ("strategies", "configs", "templates", "reports")

DEFAULT_CRITERIA_YAML = """\
# Starting values for the success criteria of new research cards.
# Hand-curated; never derive these values from past results.
# Each card sets its own values at registration — max_drawdown in particular
# depends on the strategy and its position sizing.
min_sharpe: 1.0
min_trades: 100
max_drawdown: 0.40
min_fold_stability: 0.6
max_p_value: 0.05
"""


class ResearchRepoError(Exception):
    """The research register's git repository could not be created."""


def init_research_repo(research_dir: Path) -> bool:
    """Create the register layout and its git repository.

    Idempotent: existing files are never overwritten. Returns ``True`` if this
    call created the git repository.
    """
    research_dir.mkdir(parents=True, exist_ok=True)
    for name in _DIRECTORIES:
        (research_dir / name).mkdir(exist_ok=True)
    template = research_dir / CRITERIA_TEMPLATE
    if not template.exists():
        template.write_text(DEFAULT_CRITERIA_YAML, encoding="utf-8")
    (research_dir / LEDGER_FILENAME).touch()
    if (research_dir / ".git").exists():
        return False
    _git(research_dir, "init", "--quiet")
    _git(research_dir, "add", "--all")
    _git(research_dir, "commit", "--quiet", "--message", "chore: initialise research register")
    return True


def _git(cwd: Path, *args: str) -> None:
    try:
        subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=30
        )
    except subprocess.CalledProcessError as exc:
        raise ResearchRepoError(f"git {args[0]} failed: {exc.stderr.strip()}") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ResearchRepoError(f"git {args[0]} failed: {exc}") from exc
```

The template holds starting values, not a policy: every card carries its
own `success_criteria`. `max_drawdown` defaults to 0.40, the operator's
drawdown target from the swing study, and is expected to be set per
strategy. The step 2 backfill mapping therefore needs a per-card criteria
override, so legacy cards do not all inherit the template unchanged.

**Step 5: Run to verify pass**

Run: `PYTHONPATH=src $PY -m pytest tests/research/test_repo.py -q`
Expected: 6 passed.

**Step 6: Commit**

```bash
git add src/fwbg/api/workspace.py src/fwbg/research/repo.py tests/conftest.py tests/research/test_repo.py
git commit -m "feat(research): add research directory and repository setup"
```

---

### Task 6: `code.patch` for runs with uncommitted changes

> **Amended after code review (done).** The code blocks below are the first
> iteration. Review found that porcelain `git diff` follows user config
> (`diff.external`, `diff.noprefix`, `color.diff`, textconv) and silently
> wrote unappliable patches, that quoted names and `--no-index` exit codes
> dropped untracked files without a warning, and that git could take the
> index lock during a run. The merged module diffs with `--binary
> --no-ext-diff --no-textconv --no-color --src-prefix=a/ --dst-prefix=b/`,
> lists untracked files with `-z`, fails loudly on an empty per-file diff,
> works on bytes end to end, and runs git through the shared
> `fwbg.core.git_env.isolated_git_env()` (no repository-local variables,
> `GIT_OPTIONAL_LOCKS=0`), which `fwbg.research.repo` uses too. The source
> of truth is the code on the branch.

**Files:**
- Create: `src/fwbg/results/code_patch.py`
- Modify: `src/fwbg/results/storage.py` (`create_run_directory`, right after `git_info = get_git_info()`)
- Test: `tests/test_code_patch.py`

**Step 1: Write the failing tests**

`tests/test_code_patch.py`:
```python
import json
import subprocess
from pathlib import Path

import pytest

from fwbg.results import storage
from fwbg.results.code_patch import (
    PATCH_FILENAME,
    CodePatchError,
    capture_code_patch,
    write_code_patch,
)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path, git_env):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(root, "init", "--quiet")
    git(root, "add", "--all")
    git(root, "commit", "--quiet", "--message", "initial")
    return root


@pytest.fixture
def no_repo(tmp_path, git_env, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    return outside


def make_dirty(root: Path) -> None:
    (root / "src" / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")
    (root / "src" / "new_module.py").write_text("NEW = True\n", encoding="utf-8")
    (root / "notes.txt").write_text("scratch\n", encoding="utf-8")


def test_clean_repo_has_empty_patch(repo):
    assert capture_code_patch(repo) == ""


def test_patch_covers_tracked_changes_and_untracked_sources_only(repo):
    make_dirty(repo)
    patch = capture_code_patch(repo)
    assert "src/tracked.py" in patch
    assert "src/new_module.py" in patch
    assert "notes.txt" not in patch


def test_patch_reproduces_the_working_tree(repo, tmp_path):
    make_dirty(repo)
    patch_file = tmp_path / PATCH_FILENAME
    patch_file.write_text(capture_code_patch(repo / "src"), encoding="utf-8")
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "--quiet", str(repo), str(clone))
    git(clone, "apply", str(patch_file))
    for name in ("src/tracked.py", "src/new_module.py"):
        assert (clone / name).read_text(encoding="utf-8") == (repo / name).read_text(
            encoding="utf-8"
        )


def test_outside_a_repository_raises(no_repo):
    with pytest.raises(CodePatchError):
        capture_code_patch(no_repo)


def test_write_code_patch_skips_clean_trees(repo, tmp_path, monkeypatch):
    monkeypatch.chdir(repo)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert write_code_patch(run_dir) is None
    assert not (run_dir / PATCH_FILENAME).exists()


def test_write_code_patch_outside_a_repository_returns_none(no_repo, monkeypatch):
    monkeypatch.chdir(no_repo)
    assert write_code_patch(no_repo) is None


def test_dirty_run_directory_records_the_patch(repo, tmp_path, monkeypatch):
    make_dirty(repo)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(storage, "RESULTS_BASE_PATH", str(tmp_path / "results"))
    run_path = Path(storage.create_run_directory("run-1"))
    config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
    assert config["git"]["dirty"] is True
    assert config["git"]["patch"] == PATCH_FILENAME
    assert "src/new_module.py" in (run_path / PATCH_FILENAME).read_text(encoding="utf-8")
```

**Step 2: Run to verify failure**

Run: `PYTHONPATH=src $PY -m pytest tests/test_code_patch.py -q`
Expected: collection error, `No module named 'fwbg.results.code_patch'`.

**Step 3: Implement**

`src/fwbg/results/code_patch.py`:
```python
"""Capture uncommitted source changes so a run stays reproducible."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

PATCH_FILENAME = "code.patch"
SOURCE_DIRS = ("src", "packages")


class CodePatchError(Exception):
    """The working tree's changes could not be captured."""


def capture_code_patch(cwd: Path) -> str:
    """Patch from HEAD to the working tree of the repository containing *cwd*.

    ``git diff HEAD`` misses new modules that were never added, so untracked
    files under ``src/`` and ``packages/`` are appended as new-file diffs.
    Returns an empty string for a clean tree.
    """
    root = Path(_git(cwd, "rev-parse", "--show-toplevel").strip())
    parts = [_git(root, "diff", "--binary", "HEAD")]
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "--", *SOURCE_DIRS)
    for name in untracked.splitlines():
        # --no-index exits 1 when the files differ, which they always do here.
        parts.append(
            _git(root, "diff", "--binary", "--no-index", "--", os.devnull, name, ok_codes=(0, 1))
        )
    return "".join(parts)


def write_code_patch(run_dir: str | os.PathLike[str]) -> str | None:
    """Write ``code.patch`` for the repository in the current directory into *run_dir*.

    Returns the file name, or ``None`` when the tree is clean or no patch could
    be captured (no git, not a repository) — a missing patch must never fail a run.
    """
    try:
        patch = capture_code_patch(Path.cwd())
    except CodePatchError as exc:
        log.warning("could not capture %s: %s", PATCH_FILENAME, exc)
        return None
    if not patch:
        return None
    (Path(run_dir) / PATCH_FILENAME).write_text(patch, encoding="utf-8")
    return PATCH_FILENAME


def _git(cwd: Path, *args: str, ok_codes: tuple[int, ...] = (0,)) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CodePatchError(f"git {args[0]} failed: {exc}") from exc
    if result.returncode not in ok_codes:
        raise CodePatchError(
            f"git {args[0]} exited {result.returncode}: {result.stderr.strip()}"
        )
    return result.stdout
```

In `src/fwbg/results/storage.py`, add the import next to the existing
`fwbg` imports:
```python
from fwbg.results.code_patch import write_code_patch
```
and in `create_run_directory`, directly after `git_info = get_git_info()`:
```python
    if git_info.get("dirty"):
        git_info["patch"] = write_code_patch(run_path)
```

This hook covers ordinary saved runs. The research integration must call the
same `write_code_patch` function for a `--no-save` run's staging directory
(`$FWBG_RESEARCH_DIR/.staging/<run_id>`); do not make registration of the
trial or its code state depend on `save_results=True`. Normal `test_results`
artefacts remain disabled for `--no-save`, while the research staging data is
kept until the ledger and repository commit succeed and is recoverable by
`reconcile` if either step fails.

**Step 4: Run to verify pass**

Run: `PYTHONPATH=src $PY -m pytest tests/test_code_patch.py tests/test_preview_run.py -q`
Expected: all passed (7 new tests; `test_preview_run.py` unchanged).

**Step 5: Commit**

```bash
git add src/fwbg/results/code_patch.py src/fwbg/results/storage.py tests/test_code_patch.py
git commit -m "feat(results): store code.patch for runs with uncommitted changes"
```

---

### Task 7: Verification and documentation

**Step 1: Lint, types, tests**

```bash
../../.venv/bin/ruff check src/fwbg/research src/fwbg/results/code_patch.py src/fwbg/results/storage.py src/fwbg/api/workspace.py tests/research tests/test_code_patch.py tests/conftest.py
PYTHONPATH=src ../../.venv/bin/mypy src/fwbg/research src/fwbg/results/code_patch.py
PYTHONPATH=src $PY -m pytest tests/research tests/test_code_patch.py -q
PYTHONPATH=src $PY -m pytest tests/test_preview_run.py tests/test_cli.py tests/test_api_result_paths.py tests/test_api_run_spawn.py -q
```
Expected: ruff clean, mypy clean for the new modules, research/patch tests (count as reported by the run)
passed, the regression subset (50 tests) unchanged. Then run the full suite once
(`PYTHONPATH=src $PY -m pytest -q`) and compare against `origin/develop` if
anything fails, to tell new failures from pre-existing ones.

Sanity check that the tests can fail: temporarily change `_next_id` to
`max(numbers, default=0)` (no `+ 1`), confirm four ledger tests fail,
revert.

**Step 2: Update the design doc status**

In `docs/plans/2026-10-19-research-register-design.md`, set the status line
to `## Status: Design approved (2026-10-19) — step 1 (core) implemented`.

**Step 3: Refresh the knowledge graph**

Run: `graphify update .`

**Step 4: Commit**

```bash
git add docs/plans/2026-10-19-research-register-design.md
git commit -m "docs: mark research register core as implemented"
```

---

## Notes for later steps

- Do not expose `--holdout` in the CLI/API before step 5 defines how a
  holdout trial trains up to the cutoff and evaluates only after it; until
  then a holdout run would be an ordinary walk-forward over all data and
  spend the free look.
- Treat any exception from `load_card` / `read_trials` as a denial (fail
  closed), not only `CardError` / `LedgerError`.
- Step 4 (API in Docker) needs decisions before the container can write the
  register: install `git` in the image (`python:3.14-slim` ships without
  it), choose a committer identity for register commits, and decide whether
  register commits follow the operator's `commit.gpgsign` setting.
- If init is ever called implicitly (CLI/API), hold `.register.lock` around
  it; concurrent first inits race today.
- Every git call must use `fwbg.core.git_env.isolated_git_env()`, as
  `fwbg.research.repo` and `fwbg.results.code_patch` do.
- `get_git_info` and `capture_code_patch` both follow the current working
  directory. Before trials record code state, anchor both to the fwbg source
  tree (e.g. the repository containing `fwbg.__file__`) and change them
  together, so commit and patch always come from the same repository.
- `write_code_patch` returns `None` both for "nothing under src/packages
  changed" and for "capture failed". Trials must distinguish the two (e.g.
  record a capture error) so a run without reproducible code state is
  visible.
- When the CLI exists, point the ledger's missing-file message at the CLI
  command instead of the Python function `init_research_repo`.

- Step 3 records the card's `holdout_cutoff` in every trial it appends;
  step 2 (`backfill --apply`) and `reconcile` must do the same for legacy
  and re-indexed runs, otherwise the frozen cutoff locks the card.
- The verdict step must reject unknown success-criteria names (a typo such
  as `min_sharp` would otherwise be ignored silently).

- Step 3 must pass the **same** preset-resolved dict to `fingerprint` and to
  `run_optimizer`, so a preset edited between gate and run cannot diverge,
  and should store that resolved dict with the run.
- Step 3 must turn a relative `--data-path` into an absolute path where the
  CLI merges it into `_cli_data_path`; otherwise the same string can point at
  different data.

- Canonicalise the timeframe before gating:
  `metadata["timeframe"] = Timeframe.from_str(metadata.get("timeframe") or data_config.TIMEFRAME).canonical`,
  so aliases and the environment default cannot bypass duplicate detection.
- The gate reads a ledger snapshot; the trial is appended minutes later.
  Re-check the duplicate and holdout rules under `.register.lock` when
  appending, so two concurrent identical runs (or two concurrent "free"
  holdout looks) cannot both pass. Required before `--holdout` ships.
- Pass metrics through `fwbg.data.config.convert_numpy` (and NaN → `None`)
  before building a `Trial`; numpy scalars make `append_trial` refuse it.
- The API preflight must fingerprint `resolve_presets(raw strategy file +
  overrides)`, never the dashboard loader's output: `api/strategies.py`
  resolves `exit_params` but not `regime_filter`, the opposite of
  `PRESET_SECTIONS`, so CLI and API fingerprints would diverge.
- Decide together with the signing question whether register commits run
  the operator's global git hooks (`core.hooksPath`) or use `--no-verify`.
- `code.patch` covers the whole repository (`git diff HEAD`), so a modified
  tracked data file is copied into every dirty run directory; revisit if
  that becomes a size problem (script changes outside `src/` matter, so do
  not simply limit it to source directories).

## Next steps

Step 2 (backfill and cleanup) gets its own plan once this one is merged and
the operator has confirmed the default success criteria.
