# Research Register Step 3 (CLI Gate and Card Commands) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Every backtest started with `python -m fwbg.cli` passes the research
gate and is recorded as a trial — also when it is aborted — and the operator
manages cards with `fwbg research new | list | show | amend | verdict | reopen`.

**Architecture:** `fwbg.research.runs` owns the wiring: it canonicalises the
run metadata, gates the run inside `run_optimizer` once the concrete symbols
are known (a `before_run` callback), and at that moment writes a *pending
marker* (`.staging/<run_id>.json`) that already holds the whole trial except
its outcome. Recording the finished run turns the marker into a ledger line.
A run that never reaches its own recording (Ctrl-C ends the process through
`os._exit`, SIGKILL, OOM) is recorded by the next register write that finds
its marker with a dead process. Card writes go through new helpers in
`fwbg.research.card`; write commands live in `fwbg.research.commands`,
read-only views in `fwbg.research.views`, shared statistics in
`fwbg.research.stats`. Design: `docs/plans/2026-10-19-research-register-design.md`;
steps 1 and 2 are merged (#180, #181) and the real register is backfilled.

**Tech Stack:** Python 3.14, argparse, PyYAML, pytest; git only through
`fwbg.research.repo` / `fwbg.core.git_env.isolated_git_env()`.

---

## Decisions

1. **Strict from this step on (operator decision 2026-10-20).** Every run of
   the backtest CLI needs `--research-id` and `--note`. API, dashboard and
   fwbg-agents runs spawn the same CLI without them and are refused (exit 2,
   recorded by the API as a failed run) until step 4. Other subcommands
   (`--list`, `--compare`, `--load`, `--reverse-worst`, `data`, `api`,
   `analyze`, `portfolio`, `research`) are not runs and not gated;
   `fwbg portfolio` and study scripts get `register_trial()` in step 4.
2. **`run_optimizer` is the enforcement point.** It refuses to run without a
   `before_run` gate while the gate is on, so no caller can bypass it by
   calling the function directly.
3. **The gate sees the symbols that really run.** `run_optimizer` derives
   them from the data files after all asset filters; `before_run(metadata,
   symbols)` is called right after that selection and before the run
   directory exists. It returns the effective metadata (end date clamped to
   the cutoff) from which `run_optimizer` rebuilds its `StrategyConfig`. A
   refused run writes nothing.
4. **One metadata dict for gate, run and record.** With the gate on,
   `main()` passes the loaded strategy plus CLI overrides through
   `prepare_metadata`: presets resolved, timeframe canonical, `_cli_data_path`
   absolute. That dict is fingerprinted, run, written to `strategy.json` and
   stored as `configs/<fp>.json`. A strategy file that fails to load is an
   error, not a default-config run.
5. **A gated run is on record from its start (pending marker).** The marker
   is written under `register_lock` in `before_run`; it holds the trial
   (card, kind, fingerprint, criteria hash, cutoff, window, note, reason),
   the canonical config, the run directory and the pid. `record()` appends
   the trial with its outcome and deletes the marker. Stale markers (dead
   pid) are recorded by the next `before_run` and by `verdict`. A live
   marker with the same fingerprint counts as a duplicate, and `verdict`
   waits for live runs on its card.
6. **Outcome from the run directory, one rule for all trials.** Status and
   per-symbol metrics come from `grid_details/` (`legacy.run_outcome`): `ok`
   iff at least one symbol was evaluated, whether the run finished, crashed
   or was killed. No directory → `failed`. For live runs an unknown symbol
   status counts as evaluated (conservative for N) instead of dropping the
   trial; the backfill keeps refusing unknown statuses. All statuses
   `process_symbol` produces are classified in `fwbg.statuses`.
7. **Recording never fails the run.** Any failure prints a warning; the
   marker stays and the next gated run records it. Every register commit
   stages all register paths (`strategies/`, `configs/`, `trials.jsonl`), so
   any commit picks up an earlier failed one.
8. **A run id is used once.** With the gate on, a run id whose directory
   already holds `config.json` or `grid_details/`, or that has a marker, is
   refused (a bare directory is allowed: the API pre-creates it for logs).
9. **`--no-save` is refused while the gate is on** (a run whose results were
   seen is a trial; the design's staging directory is not needed today).
10. **`check` is a flag of the run command: `python -m fwbg.cli --check …`**
    (it needs the run command's overrides and asset selection). It writes
    nothing; exit 0 if allowed, 2 if refused.
11. **Known criteria names only**: `min_sharpe`, `min_trades`,
    `max_drawdown`, `min_fold_stability`, `max_p_value`, enforced by
    `parse_card` and by the backfill mapping.
12. **Verdict statistics without DSR** (N per card and register, criteria
    versions, holdout state, best development result against the criteria);
    DSR comes with step 5.
13. **Code state from the fwbg source tree**, not the working directory; a
    failed patch capture is recorded as `patch_error`.

## Not in this step

`--holdout` and its under-lock re-check (step 5), `reconcile` beyond the
pending markers, API preflight and pass-through, `register_trial()` (step
4), DSR (step 5). Known limits: legacy fingerprints only match a new run with
the same window (new iterations always carry an end date); a reused pid can
keep a stale marker looking alive (`ponytail:` in `pending._alive`).

## Conventions for every task

- Worktree `/home/haex/Projekte/fwbg/.worktrees/<branch>` on a topic branch
  from `origin/develop`; the primary checkout is never touched.
- From the worktree: `PY=../../.venv/bin/python`;
  `PYTHONPATH=src $PY -m pytest -p no:cacheprovider <paths> -q`.
- Never write under `~/fwbg`; tests use `tmp_path` and the `git_env` fixture.
- Commit messages in English, short, no AI or model reference, no trailer.
- Before each commit: black `--check` and ruff on touched files; at the end
  `ruff check src/ packages/` and `mypy --follow-imports=silent src/fwbg/research`.
  Write conditions inline where mypy must narrow (`if x is not None and …`),
  not through an aliased boolean.
- `src/fwbg/cli/main.py` is already ~990 lines (pre-existing breach of the
  500-LoC rule); add only the few lines below there, all logic goes to
  `fwbg.research.runs`.

---

### Task 1: Code state from the fwbg source tree; patch errors recorded

**Files:** `src/fwbg/core/git_env.py`, `src/fwbg/results/code_patch.py`,
`src/fwbg/results/storage.py`, `src/fwbg/research/legacy.py` (`_GIT_KEYS`);
tests `tests/test_code_patch.py`, `tests/test_git_env.py`.

**Step 1: Tests first.**

```python
def test_write_code_patch_reports_a_failed_capture(tmp_path, monkeypatch):
    def fail(_cwd):
        raise CodePatchError("git diff exited 128: boom")

    monkeypatch.setattr("fwbg.results.code_patch.capture_code_patch", fail)
    assert write_code_patch(tmp_path, tmp_path) == (None, "git diff exited 128: boom")


def test_write_code_patch_clean_tree_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr("fwbg.results.code_patch.capture_code_patch", lambda _cwd: b"")
    assert write_code_patch(tmp_path, tmp_path) == (None, None)


def test_git_info_describes_the_given_tree_not_the_working_directory(git_env, tmp_path, monkeypatch):
    repo = <existing repo helper of this module>(tmp_path / "repo")
    monkeypatch.chdir(tmp_path)                      # outside the repo
    assert get_git_info(repo)["dirty"] is False and "commit" in get_git_info(repo)


def test_run_directory_records_a_patch_error(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RESULTS_BASE_PATH", str(tmp_path / "results"))
    monkeypatch.setattr(storage, "get_git_info", lambda _source: {"commit": "abc", "dirty": True})
    monkeypatch.setattr(storage, "write_code_patch", lambda _run, _src: (None, "boom"))
    run_path = storage.create_run_directory("run-1")
    git = json.loads((Path(run_path) / "config.json").read_text())["git"]
    assert git["patch"] is None and git["patch_error"] == "boom"
```

`tests/test_git_env.py`: `source_dir() == Path(fwbg.__file__).resolve().parent`.

**Existing tests to adapt:** every `write_code_patch(run_dir)` call in
`tests/test_code_patch.py` takes the new signature and tuple result;
`test_dirty_run_directory_records_the_patch` and
`test_git_info_isolated_and_detects_untracked_source` relied on
`chdir(repo)` — monkeypatch `storage.source_dir` (imported by name) to
return the repo, or call `get_git_info(repo)`.

**Step 2:** run `tests/test_code_patch.py tests/test_git_env.py` → FAIL.

**Step 3: Implement.**

`git_env.py`:

```python
def source_dir() -> Path:
    """The installed fwbg package; git run here describes the code that runs.

    Outside a git checkout (an installed wheel) git fails here and the run
    records no git information.
    """
    # ponytail: a non-editable install in a venv inside a checkout would report
    # that checkout's HEAD; check `git ls-files --error-unmatch` if that ever matters.
    return Path(__file__).resolve().parent.parent
```

`code_patch.py`:

```python
def write_code_patch(
    run_dir: str | os.PathLike[str], source: Path
) -> tuple[str | None, str | None]:
    """Write ``code.patch`` for the repository containing *source* into *run_dir*.

    Returns ``(file name, None)`` when a patch was written, ``(None, None)``
    for a clean tree and ``(None, error)`` when no patch could be captured or
    written. A missing patch never fails a run, but it must stay visible.
    """
    try:
        patch = capture_code_patch(source)
    except CodePatchError as exc:
        log.warning("could not capture %s: %s", PATCH_FILENAME, exc)
        return None, str(exc)
    if not patch:
        return None, None
    try:
        (Path(run_dir) / PATCH_FILENAME).write_bytes(patch)
    except OSError as exc:
        log.warning("could not write %s: %s", PATCH_FILENAME, exc)
        # config.json will record no patch, so a truncated file must not stay behind.
        with contextlib.suppress(OSError):
            (Path(run_dir) / PATCH_FILENAME).unlink(missing_ok=True)
        return None, f"could not write {PATCH_FILENAME}: {exc}"
    return PATCH_FILENAME, None
```

`storage.py`: `get_git_info(source: Path | None = None)` passes
`cwd=source or source_dir()` to every `subprocess.run`; in
`create_run_directory`:

```python
    source = source_dir()
    git_info = get_git_info(source)
    if git_info.get("dirty"):
        git_info["patch"], patch_error = write_code_patch(run_path, source)
        if patch_error:
            git_info["patch_error"] = patch_error
```

`legacy.py`: `_GIT_KEYS = ("commit", "branch", "dirty", "patch", "patch_error")`.

**Step 4:** `tests/test_code_patch.py tests/test_git_env.py tests/research` → PASS.
**Commit:** `fix(results): capture code state from the fwbg source tree`

---

### Task 2: Card editing helpers and known criteria

**Files:** `src/fwbg/research/card.py`, `src/fwbg/research/mapping.py`
(`parse_criteria`); tests `tests/research/test_card.py`,
`tests/research/test_mapping.py`, `tests/research/conftest.py`.

**Step 1: Tests.** Add a fixture `strategies_dir` to `tests/research/conftest.py`
(`tmp_path / "research"` with an empty `strategies/`; no git). Then:

```python
def test_unknown_criterion_is_refused():
    text = render_card(make_card()).replace("min_trades:", "min_trade:")
    with pytest.raises(CardError, match="unknown criterion 'min_trade'"):
        parse_card(text)


def test_replace_card_rewrites_the_existing_file(strategies_dir):
    path = write_card(strategies_dir, make_card(), "cusum")
    replace_card(strategies_dir, make_card(status=Status.PARKED))
    assert load_card(strategies_dir, "S-0007").status is Status.PARKED
    assert [p.name for p in (strategies_dir / "strategies").iterdir()] == [path.name]


def test_replace_card_needs_an_existing_card(strategies_dir):
    with pytest.raises(CardError, match="no research card S-0007"):
        replace_card(strategies_dir, make_card())


def test_next_card_id_follows_cards_and_ledger(strategies_dir):
    assert next_card_id(strategies_dir, []) == "S-0001"
    write_card(strategies_dir, make_card(id="S-0007"), "x")
    assert next_card_id(strategies_dir, []) == "S-0008"
    # an id whose card file is gone is never reused: its trials would follow it
    assert next_card_id(strategies_dir, [make_trial(strategy="S-0011")]) == "S-0012"


def test_add_section_entry_appends_inside_the_section():
    body = "## Hypothesis\n\nH.\n\n## Verdict\n\nOpen.\n\n## Reports\n\n- a\n"
    assert add_section_entry(body, "Verdict", "### 2026-10-20: parked") == (
        "## Hypothesis\n\nH.\n\n## Verdict\n\nOpen.\n\n### 2026-10-20: parked\n\n## Reports\n\n- a\n"
    )


def test_add_section_entry_creates_a_missing_section_before_an_anchor():
    body = "## Hypothesis\n\nH.\n\n## Verdict\n\nOpen.\n"
    assert add_section_entry(body, "History", "- reopened", before=("Verdict",)) == (
        "## Hypothesis\n\nH.\n\n## History\n\n- reopened\n\n## Verdict\n\nOpen.\n"
    )


def test_add_section_entry_appends_a_missing_section_at_the_end():
    assert add_section_entry("Intro.\n", "Verdict", "x") == "Intro.\n\n## Verdict\n\nx\n"
    assert add_section_entry("", "Verdict", "x") == "## Verdict\n\nx\n"
```

`test_mapping.py`: a card override `criteria: {min_sharp: 1}` →
`BackfillError` naming the unknown criterion at parse time.

**Existing test to adapt:** `test_card.py`
`test_render_names_the_card_when_it_would_not_read_back` uses
`success_criteria={"x": inf}` — change to `{"min_sharpe": inf}` (keep the
asserted `success_criteria\.min_sharpe` message).

**Step 2:** run → FAIL.

**Step 3: Implement** (`card.py`):

```python
CRITERIA_NAMES = frozenset(
    {"min_sharpe", "min_trades", "max_drawdown", "min_fold_stability", "max_p_value"}
)
CARD_FILE_RE = re.compile(r"(S-[0-9]{4})(?:-.*)?\.md")
_HEADING_RE = re.compile(r"^## (.+?)[ \t]*$", re.MULTILINE)
```

Rename `_criteria` to the public `validate_criteria` (same body) and add,
after the key type check:

```python
        if key not in CRITERIA_NAMES:
            raise CardError(
                f"success_criteria: unknown criterion {key!r} "
                f"(known: {', '.join(sorted(CRITERIA_NAMES))})"
            )
```

Path lookup split out of `load_card` (which then uses it):

```python
def card_path(research_dir: Path, card_id: str) -> Path | None:
    """The file of card *card_id*; ``None`` if there is none."""
    _card_id(card_id)
    folder = research_dir / "strategies"
    paths = sorted({*folder.glob(f"{card_id}-*.md"), *folder.glob(f"{card_id}.md")})
    if len(paths) > 1:
        raise CardError(f"ambiguous research id {card_id}: {[p.name for p in paths]}")
    return paths[0] if paths else None


def replace_card(research_dir: Path, card: ResearchCard) -> Path:
    """Rewrite the existing file of ``card.id`` atomically; call under ``register_lock``."""
    path = card_path(research_dir, card.id)
    if path is None:
        raise CardError(f"no research card {card.id}")
    text = render_card(card)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def next_card_id(research_dir: Path, trials: Iterable[Trial]) -> str:
    """The id after the highest card file or ledger card; call under ``register_lock``."""
    numbers = [
        int(match.group(1)[2:])
        for path in (research_dir / "strategies").glob("S-*.md")
        if (match := CARD_FILE_RE.fullmatch(path.name))
    ]
    numbers += [int(trial.strategy[2:]) for trial in trials]
    number = max(numbers, default=0) + 1
    if number > 9999:
        raise CardError("research ids are exhausted (S-9999)")
    return f"S-{number:04d}"


def add_section_entry(body: str, heading: str, entry: str, before: Sequence[str] = ()) -> str:
    """Append *entry* to the ``## <heading>`` section of a card body.

    A missing section is created before the first *before* section the body
    has, else at the end. Text outside the section stays as it was.
    """
    entry = entry.strip("\n")
    heads = list(_HEADING_RE.finditer(body))
    own = next((m for m in heads if m.group(1) == heading), None)
    if own is not None:
        end = next((m.start() for m in heads if m.start() > own.start()), len(body))
        section, tail = body[own.start() : end].rstrip("\n"), body[end:]
        return body[: own.start()] + section + "\n\n" + entry + "\n" + ("\n" + tail if tail else "")
    new = f"## {heading}\n\n{entry}\n"
    anchor = next((m.start() for m in heads if m.group(1) in before), None)
    if anchor is not None:
        return body[:anchor] + new + "\n" + body[anchor:]
    head = body.rstrip("\n")
    return (head + "\n\n" if head else "") + new
```

`card.py` imports `Trial` only under `TYPE_CHECKING` (ledger imports
`CARD_ID_RE` from card; avoid the cycle). Add `os`, `uuid`, `Iterable`,
`Sequence`. `mapping.parse_criteria` calls `validate_criteria` and wraps
`CardError` into `BackfillError(f"{where}: {exc}")`; drop its duplicated
checks.

**Step 4:** `tests/research` → PASS.
**Commit:** `feat(research): add card editing helpers and refuse unknown criteria`

---

### Task 3: Status classification, run outcome and pending markers

**Files:** `src/fwbg/statuses.py`, `src/fwbg/research/legacy.py`,
`src/fwbg/research/ledger.py` (`Trial.from_json(..., numbered=)`),
`src/fwbg/research/repo.py` (`REGISTER_PATHS`), create
`src/fwbg/research/pending.py`; tests `tests/test_statuses.py`,
`tests/research/test_legacy.py`, `tests/research/test_pending.py`.

**Step 1: Tests.**

```python
# tests/test_statuses.py
def test_every_status_process_symbol_returns_is_classified():
    source = Path("src/fwbg/optimization/process.py").read_text(encoding="utf-8")
    produced = set(re.findall(r'"status":\s*"([a-z_]+)"', source))
    assert produced, "the pattern no longer finds statuses; update this test"
    assert produced <= EXECUTED_STATUSES | NOT_EXECUTED_STATUSES
    assert not EXECUTED_STATUSES & NOT_EXECUTED_STATUSES
```

`test_legacy.py`:
- `run_outcome` of a run with `EURUSD` `ok` and `DXY` `macro_asset` →
  `(OK, {"EURUSD": …})`;
- an unknown status raises `LegacyRunError` by default and counts as
  evaluated with `count_unknown=True`;
- `scan_runs` skips a run whose `config.json` has a `research` block
  (gated runs are already in the ledger).

`test_pending.py` (register from `init_research_repo` + `git_env`):
- `write_pending` then `read_pending` returns the same `PendingRun`
  (trial without id, config, run dir, pid); a second `write_pending` for the
  same run id raises `PendingError`.
- `finish` on a run directory made with `make_run` appends the trial with
  the outcome (`ok`, metrics, `code` from `config.json`'s git block), stores
  the config, deletes the marker, and returns the stored trial (`T-000001`).
- `finish` without a run directory appends `failed` with `{}` metrics and
  `{}` code.
- `finish` for a run already in the ledger (crash between append and
  unlink) only removes the marker and returns the recorded trial.
- `recover_stale` records a marker whose pid is dead (use a pid from a
  finished `subprocess.Popen(["true"])`) and leaves one with `os.getpid()`;
  `live_runs(register, "S-0007")` lists the live run id.
- A finished trial survives a failing `append_trial` (monkeypatched to
  raise): the marker stays for a later `recover_stale`.
- `Trial.from_json(line, numbered=False)` requires `"trial": null`;
  `numbered=True` (default) still refuses it.

**Step 2:** run → FAIL.

**Step 3: Implement.**

`src/fwbg/statuses.py`:

```python
# Per-symbol statuses that end before a backtest was evaluated.
NOT_EXECUTED_STATUSES = frozenset(
    {
        "error",
        "insufficient_data",
        "insufficient_data_for_folds",
        "macro_asset",
        "missing_model_dependencies",
        "no_data",
        "no_signal_source",
        "no_successful_folds",
    }
)
```

`legacy.py`: `NOT_EVALUATED_STATUSES = NOT_EXECUTED_STATUSES`; `_evaluated`
and `_metrics` get `count_unknown: bool = False` (unknown → evaluated when
true, `LegacyRunError` otherwise); public helpers:

```python
def run_outcome(run_dir: Path, *, count_unknown: bool = False) -> tuple[TrialStatus, Metrics]:
    """Status and per-symbol metrics of a run directory, by the ledger's rule.

    ``ok`` iff at least one symbol was evaluated. An unknown symbol status
    raises :class:`LegacyRunError`, or counts as evaluated with
    *count_unknown* (live runs: never drop a trial over a classification gap).
    """
    metrics = _metrics(run_dir, _symbol_files(run_dir), count_unknown)
    return (TrialStatus.OK if metrics else TrialStatus.FAILED), metrics
```

Rename `_code` → `run_code`, `_window` → `run_window`; `load_run` uses
`run_outcome`. `scan_runs` also skips run directories whose `config.json`
contains a `research` key.

`ledger.py`: `Trial.from_json(cls, line: str, *, numbered: bool = True)`;
with `numbered=False` the `"trial"` value must be `None` and `trial_id`
stays `None`.

`repo.py`: `REGISTER_PATHS = ("strategies", "configs", LEDGER_FILENAME)` —
every register commit in this step passes it.

`src/fwbg/research/pending.py`:

```python
"""Pending trials: a gated run is on record from the moment it starts.

The gate writes ``.staging/<run_id>.json`` (ignored by the register's git)
before the run starts. :func:`finish` turns it into a ledger line. A run
that never records itself — Ctrl-C ends the process through ``os._exit``,
SIGKILL, the OOM killer — leaves its marker behind, and :func:`recover_stale`
records it once its process is gone.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fwbg.research.ledger import LEDGER_FILENAME, Trial, TrialStatus, append_trial, read_trials
from fwbg.research.legacy import run_code, run_outcome
from fwbg.research.repo import store_config

STAGING = ".staging"


class PendingError(Exception):
    """A pending marker cannot be written or read."""


@dataclass(frozen=True)
class PendingRun:
    run_id: str
    run_dir: Path
    pid: int
    trial: Trial
    config: Mapping[str, Any]


def write_pending(research_dir: Path, pending: PendingRun) -> None:
    """Write the marker atomically; a run id that already has one is refused."""
    path = _path(research_dir, pending.run_id)
    if path.exists():
        raise PendingError(f"run {pending.run_id} is already pending")
    path.parent.mkdir(exist_ok=True)
    record = {
        "run_id": pending.run_id,
        "run_dir": str(pending.run_dir),
        "pid": pending.pid,
        "trial": json.loads(pending.trial.to_json()),
        "config": pending.config,
    }
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_pending(research_dir: Path) -> list[PendingRun]:
    """All markers, oldest run id first."""
    runs = []
    for path in sorted((research_dir / STAGING).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            trial = Trial.from_json(json.dumps(record["trial"]), numbered=False)
            runs.append(
                PendingRun(record["run_id"], Path(record["run_dir"]), int(record["pid"]),
                           trial, record["config"])
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PendingError(f"{path.name}: {exc}") from exc
    return runs


def finish(research_dir: Path, pending: PendingRun) -> Trial:
    """Append the run's trial with its outcome and drop the marker; call under the lock."""
    marker = _path(research_dir, pending.run_id)
    # A crash between append and unlink must not record the run twice.
    recorded = next(
        (
            trial
            for trial in read_trials(research_dir / LEDGER_FILENAME)
            if trial.run_id == pending.run_id and trial.strategy == pending.trial.strategy
        ),
        None,
    )
    if recorded is not None:
        marker.unlink()
        return recorded
    status, metrics, code = TrialStatus.FAILED, {}, {}
    if pending.run_dir.is_dir():
        status, metrics = run_outcome(pending.run_dir, count_unknown=True)
        code = _recorded_code(pending.run_dir)
    trial = replace(pending.trial, status=status, metrics=metrics, code=code)
    # Config first: the ledger must never name a fingerprint without its config.
    store_config(research_dir, trial.fingerprint, pending.config)
    stored = append_trial(research_dir / LEDGER_FILENAME, trial)
    marker.unlink()
    return stored


def recover_stale(research_dir: Path) -> list[Trial]:
    """Record every pending run whose process is gone; call under the lock."""
    return [finish(research_dir, run) for run in read_pending(research_dir) if not _alive(run.pid)]


def live_runs(research_dir: Path, card_id: str | None = None) -> list[PendingRun]:
    """Pending runs whose process still runs, optionally of one card."""
    return [
        run
        for run in read_pending(research_dir)
        if _alive(run.pid) and card_id in (None, run.trial.strategy)
    ]


def _alive(pid: int) -> bool:
    # ponytail: a reused pid keeps a stale marker looking alive until that
    # process ends; store the process start time if that ever bites.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recorded_code(run_dir: Path) -> dict[str, Any]:
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return run_code(config) if isinstance(config, dict) else {}


def _path(research_dir: Path, run_id: str) -> Path:
    if not run_id or "/" in run_id or run_id in (".", ".."):
        raise PendingError(f"invalid run id {run_id!r}")
    return research_dir / STAGING / f"{run_id}.json"
```

(`run_dir` is stored so recovery finds the directory even if
`RESULTS_BASE_PATH` differs in a later process.)

**Step 4:** `tests/test_statuses.py tests/research` → PASS.
**Commit:** `feat(research): record gated runs from their start with pending markers`

---

### Task 4: The gate wiring module

**Files:** `src/fwbg/research/gate.py` (`DenialCode.REGISTER_UNREADABLE`,
`DenialCode.RUN_ID_IN_USE`), create `src/fwbg/research/runs.py`; tests
`tests/research/conftest.py` (fixture `gate_register`),
`tests/research/test_runs.py`.

**Step 1: Fixture** `gate_register` in `tests/research/conftest.py`
(uses `git_env`): `init_research_repo(tmp_path / "research")`, writes the
active card `make_card()` (cutoff 2025-04-30) with `write_card` and commits
it with `commit_register(..., REGISTER_PATHS, "test card")`. Shared by
`test_runs.py`, `test_run_cli.py`, `test_commands.py`, `test_views.py`.

**Step 2: Tests** (`metadata = {"name": "x", "timeframe": "HOUR_1",
"pipeline": {"indicators": []}, "model": {}}`; `results = tmp_path / "results"`;
`ResearchRun(register, run_id, results / run_id, research_id, note, reason=None, check_only=False)`):

```python
def test_before_run_returns_the_clamped_metadata_and_marks_the_run(gate_register, results):
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "baseline")
    assert run.before_run(metadata, ["eurusd"])["end_date"] == "2025-04-30"
    assert [p.run_id for p in read_pending(gate_register)] == ["run-1"]


def test_refusal_writes_nothing(gate_register, results):
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", None)
    with pytest.raises(GateDenied) as denied:
        run.before_run(metadata, ["EURUSD"])
    assert [d.code for d in denied.value.denials] == [DenialCode.MISSING_NOTE]
    assert read_pending(gate_register) == [] and not results.exists()


def test_unreadable_register_fails_closed(tmp_path):
    run = ResearchRun(tmp_path / "missing", "r", tmp_path / "r", "S-0007", "x")
    with pytest.raises(GateDenied) as denied:
        run.before_run(metadata, ["EURUSD"])
    assert denied.value.denials[0].code is DenialCode.REGISTER_UNREADABLE


def test_malformed_research_id_is_card_not_found(gate_register, results): ...
def test_check_only_stops_without_a_marker(gate_register, results): ...   # GateChecked, no marker


def test_used_run_id_is_refused(gate_register, results):
    make_run(results, "run-1")
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "x")
    with pytest.raises(GateDenied) as denied:
        run.before_run(metadata, ["EURUSD"])
    assert denied.value.denials[0].code is DenialCode.RUN_ID_IN_USE


def test_bare_run_directory_is_allowed(gate_register, results):
    (results / "run-1").mkdir(parents=True)           # the API pre-creates it for logs
    ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "x").before_run(metadata, ["EURUSD"])


def test_record_appends_tags_and_commits(gate_register, results):
    run = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "baseline")
    effective = run.before_run(metadata, ["EURUSD"])
    make_run(results, "run-1", symbols={"EURUSD": {"unified": {"sharpe": 0.4, "trades": 120}}})
    trial = run.record()
    assert trial.trial_id == "T-000001" and trial.status is TrialStatus.OK
    assert trial.metrics["EURUSD"]["sharpe"] == 0.4 and trial.code["commit"] == "abc123"
    assert trial.window == {"start": None, "end": "2025-04-30"}
    assert json.loads((results / "run-1" / "config.json").read_text())["research"] == {
        "id": "S-0007", "trial": "T-000001", "fingerprint": trial.fingerprint,
        "kind": "iteration", "note": "baseline", "reason": None}
    assert read_pending(gate_register) == [] and git_status(gate_register) == ""
    again = ResearchRun(gate_register, "run-2", results / "run-2", "S-0007", "again")
    with pytest.raises(GateDenied) as denied:
        again.before_run(effective, ["EURUSD"])
    assert denied.value.denials[0].code is DenialCode.DUPLICATE_FINGERPRINT


def test_identical_run_in_progress_is_a_duplicate(gate_register, results):
    ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "a").before_run(metadata, ["EURUSD"])
    with pytest.raises(GateDenied) as denied:
        ResearchRun(gate_register, "run-2", results / "run-2", "S-0007", "b").before_run(metadata, ["EURUSD"])
    assert "running now" in denied.value.denials[0].message


def test_killed_run_is_recorded_by_the_next_gated_run(gate_register, results):
    first = ResearchRun(gate_register, "run-1", results / "run-1", "S-0007", "a")
    first.before_run(metadata, ["EURUSD"])
    make_run(results, "run-1", symbols={"EURUSD": {"unified": {"sharpe": 0.4}}})
    dead = subprocess.Popen(["true"]); dead.wait()
    rewrite_marker_pid(gate_register, "run-1", dead.pid)        # simulates os._exit
    ResearchRun(gate_register, "run-2", results / "run-2", "S-0007", "b").before_run(
        {**metadata, "model": {"x": 1}}, ["EURUSD"])
    trials = read_trials(gate_register / "trials.jsonl")
    assert [(t.run_id, t.status) for t in trials] == [("run-1", TrialStatus.OK)]
    assert git_status(gate_register) == ""


def test_crash_after_evaluation_keeps_the_outcome(gate_register, results):
    # record() after an exception: the run directory decides, not the crash
    ...  # make_run with an ok symbol, record() → status OK


def test_record_failure_only_warns_and_keeps_the_marker(gate_register, results, monkeypatch, capsys):
    run = ResearchRun(gate_register, "run-4", results / "run-4", "S-0007", "baseline")
    run.before_run(metadata, ["EURUSD"])
    monkeypatch.setattr("fwbg.research.pending.append_trial",
                        lambda *_: (_ for _ in ()).throw(OSError("disk full")))
    assert run.record() is None
    assert "run-4 is not in the ledger yet" in capsys.readouterr().err
    assert [p.run_id for p in read_pending(gate_register)] == ["run-4"]


def test_prepare_metadata_resolves_and_canonicalises(tmp_path): ...   # as before: preset dict,
    # "DAY" → "DAY_1", absolute _cli_data_path, default timeframe used when missing


def test_gate_is_on_unless_switched_off(monkeypatch): ...
```

(`git_status` and `rewrite_marker_pid` are small helpers in the test module.)

**Step 3:** run → FAIL.

**Step 4: Implement.** `gate.py`: add
`REGISTER_UNREADABLE = "register_unreadable"` and `RUN_ID_IN_USE = "run_id_in_use"`
(both 409). `src/fwbg/research/runs.py`:

```python
"""The research gate on the backtest run path (``python -m fwbg.cli``).

``main()`` prepares the run metadata, ``run_optimizer`` calls
:meth:`ResearchRun.before_run` once the run's symbols are known (which puts
the run on record as pending), and :func:`run_gated` records the outcome.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fwbg_sdk.enums import Timeframe

from fwbg.core.presets import resolve_presets
from fwbg.research.card import CARD_ID_RE, ResearchCard, load_card
from fwbg.research.fingerprint import FINGERPRINT_VERSION, canonical_config
from fwbg.research.gate import Decision, Denial, DenialCode, RunRequest, check
from fwbg.research.ledger import LEDGER_FILENAME, Trial, TrialStatus, read_trials
from fwbg.research.legacy import run_window
from fwbg.research.pending import PendingRun, finish, live_runs, recover_stale, write_pending
from fwbg.research.repo import REGISTER_PATHS, ResearchRepoError, commit_register, register_lock

GATE_ENV = "FWBG_RESEARCH_GATE"

BeforeRun = Callable[[Mapping[str, Any], Sequence[str]], dict[str, Any]]


def gate_enabled() -> bool:
    """On unless ``FWBG_RESEARCH_GATE=off``, which only the test suite sets."""
    return os.environ.get(GATE_ENV, "").strip().lower() != "off"


class GateDenied(Exception):
    """The register refused the run; nothing was started or written."""

    def __init__(self, denials: Sequence[Denial]) -> None:
        super().__init__("; ".join(denial.message for denial in denials))
        self.denials = tuple(denials)


class GateChecked(Exception):
    """``--check``: the run would be allowed; it stops before it starts."""

    def __init__(self, decision: Decision, symbols: tuple[str, ...]) -> None:
        super().__init__("allowed")
        self.decision = decision
        self.symbols = symbols


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """The research flags of the backtest command."""
    group = parser.add_argument_group("research register")
    group.add_argument("--research-id", metavar="S-NNNN", help="research card this run iterates")
    group.add_argument("--note", help="one line: what changed since the last iteration and why")
    group.add_argument("--reason", help="why an identical configuration runs again")
    group.add_argument(
        "--check", action="store_true", help="check the run against the register; do not run it"
    )


def prepare_metadata(metadata: Mapping[str, Any], default_timeframe: str) -> dict[str, Any]:
    """The one metadata dict the gate fingerprints, the run uses and the register stores.

    Raises:
        OSError, ValueError: a preset cannot be loaded or the timeframe is unknown.
    """
    prepared = resolve_presets(metadata)
    timeframe = prepared.get("timeframe") or default_timeframe
    prepared["timeframe"] = Timeframe.from_str(timeframe).canonical
    data_path = prepared.get("_cli_data_path")
    if isinstance(data_path, str) and data_path:
        prepared["_cli_data_path"] = str(Path(data_path).resolve())
    return prepared


@dataclass
class ResearchRun:
    """Gate and record one backtest run."""

    research_dir: Path
    run_id: str
    run_dir: Path
    research_id: str | None
    note: str | None
    reason: str | None = None
    check_only: bool = False
    _pending: PendingRun | None = field(default=None, init=False, repr=False)

    def before_run(self, metadata: Mapping[str, Any], symbols: Sequence[str]) -> dict[str, Any]:
        """Gate the run once its symbols are known; put it on record; return its metadata.

        Raises:
            GateDenied: the run must not start (also when the register is unreadable).
            GateChecked: ``check_only`` and the run would be allowed.
        """
        canonical = tuple(sorted({symbol.upper() for symbol in symbols}))
        request = RunRequest(self.research_id, metadata, canonical, self.note, self.reason)
        try:
            with register_lock(self.research_dir):
                return self._gate(request, canonical)
        except (GateDenied, GateChecked):
            raise
        except Exception as exc:  # fail closed: an unreadable register never lets a run through
            raise GateDenied(
                [Denial(DenialCode.REGISTER_UNREADABLE, f"research register: {exc}")]
            ) from exc

    def _gate(self, request: RunRequest, symbols: tuple[str, ...]) -> dict[str, Any]:
        if recover_stale(self.research_dir):
            commit_register(self.research_dir, REGISTER_PATHS, "trial: record interrupted runs")
        card: ResearchCard | None = None
        if self.research_id is not None and CARD_ID_RE.fullmatch(self.research_id):
            card = load_card(self.research_dir, self.research_id)
        decision = check(card, request, read_trials(self.research_dir / LEDGER_FILENAME))
        denials = list(decision.denials) + self._run_denials(decision)
        if denials:
            raise GateDenied(denials)
        if self.check_only:
            raise GateChecked(decision, symbols)
        assert card is not None and decision.fingerprint is not None  # allowed decisions have both
        metadata = copy.deepcopy(dict(decision.effective_metadata))
        self._pending = PendingRun(
            run_id=self.run_id,
            run_dir=self.run_dir,
            pid=os.getpid(),
            trial=Trial(
                strategy=card.id,
                run_id=self.run_id,
                kind=decision.kind,
                status=TrialStatus.FAILED,  # replaced by the outcome when recorded
                ts=datetime.now().isoformat(timespec="seconds"),
                fingerprint=decision.fingerprint,
                fingerprint_version=FINGERPRINT_VERSION,
                criteria_hash=card.criteria_hash,
                holdout_cutoff=card.holdout_cutoff,
                code={},
                window=run_window(metadata),
                note=self.note,
                reason=self.reason or None,
            ),
            config=canonical_config(metadata, symbols),
        )
        write_pending(self.research_dir, self._pending)
        return metadata

    def _run_denials(self, decision: Decision) -> list[Denial]:
        denials = []
        if (self.run_dir / "config.json").exists() or (self.run_dir / "grid_details").exists():
            denials.append(
                Denial(DenialCode.RUN_ID_IN_USE, f"run id {self.run_id} already holds a run")
            )
        running = live_runs(self.research_dir)
        if any(run.run_id == self.run_id for run in running):
            denials.append(Denial(DenialCode.RUN_ID_IN_USE, f"run {self.run_id} is running"))
        if decision.fingerprint is not None and not (self.reason and self.reason.strip()):
            same = [run.run_id for run in running if run.trial.fingerprint == decision.fingerprint]
            if same:
                denials.append(
                    Denial(
                        DenialCode.DUPLICATE_FINGERPRINT,
                        f"identical configuration is running now (run {', '.join(same)}); "
                        "pass a reason to run it again",
                    )
                )
        return denials

    def record(self) -> Trial | None:
        """Record the gated run's outcome; warn instead of failing the run.

        ``None`` if the gate never let the run start or recording failed (the
        marker then stays and the next gated run records it).
        """
        if self._pending is None:
            return None
        try:
            with register_lock(self.research_dir):
                trial = finish(self.research_dir, self._pending)
                try:
                    commit_register(
                        self.research_dir,
                        REGISTER_PATHS,
                        f"trial {trial.trial_id}: {trial.strategy} run {self.run_id}",
                    )
                except ResearchRepoError as exc:
                    print(f"Warning: {trial.trial_id} is not committed yet ({exc}); "
                          "the next register commit includes it", file=sys.stderr)
        except Exception as exc:
            print(f"Warning: run {self.run_id} is not in the ledger yet ({exc}); "
                  "the next gated run records it", file=sys.stderr)
            return None
        if self.run_dir.is_dir():
            try:
                _tag_run(self.run_dir, trial)
            except (OSError, ValueError) as exc:
                print(f"Warning: could not tag run {self.run_id} with {trial.trial_id}: {exc}",
                      file=sys.stderr)
        return trial


def run_gated(research: ResearchRun, run: Callable[[BeforeRun], Any]) -> Any:
    """Run *run* behind the gate and record it.

    Exits with 2 on a refusal and 0 after ``--check``. Any other exception is
    recorded (outcome from the run directory) and re-raised. A process ended
    by ``os._exit`` (the pool's Ctrl-C handler) or SIGKILL keeps its pending
    marker and is recorded by the next gated run.
    """
    try:
        result = run(research.before_run)
    except GateDenied as exc:
        print("Research gate refused the run:", file=sys.stderr)
        for denial in exc.denials:
            print(f"  - [{denial.code.value}] {denial.message}", file=sys.stderr)
        raise SystemExit(2) from None
    except GateChecked as exc:
        _print_check(exc.decision, exc.symbols)
        raise SystemExit(0) from None
    except BaseException:
        research.record()
        raise
    research.record()
    return result


def _tag_run(run_dir: Path, trial: Trial) -> None:
    """Add the ``research`` block to the run's config.json, so the run names its trial."""
    path = run_dir / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["research"] = {
        "id": trial.strategy,
        "trial": trial.trial_id,
        "fingerprint": trial.fingerprint,
        "kind": trial.kind.value,
        "note": trial.note,
        "reason": trial.reason,
    }
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _print_check(decision: Decision, symbols: tuple[str, ...]) -> None:
    window = run_window(decision.effective_metadata)
    print(f"Research gate: allowed ({decision.kind.value})")
    print(f"  fingerprint: {decision.fingerprint}")
    print(f"  symbols:     {', '.join(symbols)}")
    print(f"  window:      {window['start'] or 'open'} .. {window['end'] or 'open'} (end exclusive)")
```

**Step 5:** `tests/research` → PASS.
**Commit:** `feat(research): gate and record backtest runs`

---

### Task 5: Wire the gate into the backtest CLI

**Files:** `src/fwbg/cli/main.py`, `tests/conftest.py`; test
`tests/research/test_run_cli.py`.

**Step 1:** `tests/conftest.py` `pytest_configure`, next to the other
`setdefault` calls:

```python
    # The research gate needs a register; only tests that test it switch it on.
    os.environ.setdefault("FWBG_RESEARCH_GATE", "off")
```

**Step 2: Tests.** Fixture `gated`: `GATE_ENV=on`,
`FWBG_RESEARCH_DIR=<gate_register>`,
`monkeypatch.setattr(storage, "RESULTS_BASE_PATH", str(results))`, a strategy
file in `tmp_path` referencing a pipeline preset, and a fake
`fwbg.cli.main.run_optimizer` that records its kwargs, calls
`before_run(strategy_metadata, ["EURUSD"])` and creates the run directory
with `make_run`. Drive `main()` through `monkeypatch.setattr(sys, "argv", …)`.

- no `--research-id` → `SystemExit(2)`, `[missing_research_id]` on stderr,
  no trial, no marker;
- a gated run → one trial with the run id, the fake saw the run id and a
  resolved pipeline dict;
- `--check` → `SystemExit(0)`, "Research gate: allowed" on stdout, no trial,
  no marker, fake not past `before_run`;
- `--no-save` → `SystemExit(2)` (argparse error);
- a strategy file that fails to load → exit 1, nothing gated;
- the fake raises `RuntimeError` after `before_run` and after writing an ok
  symbol → re-raised, trial recorded with status `ok`;
- gate off → warning "neither gated nor recorded", fake got `before_run=None`;
- **real `run_optimizer`** with a data dir holding `EURUSD_HOUR_1.csv`:
  (a) `before_run=None` with the gate on → `RuntimeError` naming the gate;
  (b) a `before_run` returning `{**metadata, "end_date": "2024-01-01"}` while
  `SimplePoolManager.map_adaptive` is patched to capture the `func`
  partial: the partial's `strategy.end_date == "2024-01-01"` and the run's
  `strategy.json` has the same end date — this is what keeps holdout data
  out of iterations; (c) a `before_run` raising `GateDenied` → nothing under
  `RESULTS_BASE_PATH`.

**Step 3:** run → FAIL.

**Step 4: Implement** (`src/fwbg/cli/main.py`, imports in the module's
existing `# noqa: E402` block):

```python
from fwbg.api.workspace import get_research_dir  # noqa: E402
from fwbg.research.runs import (  # noqa: E402
    GATE_ENV,
    ResearchRun,
    add_run_arguments,
    gate_enabled,
    prepare_metadata,
    run_gated,
)
```

`run_optimizer(..., run_id=None, before_run=None)`; docstring line for
`before_run`. At its top:

```python
    if before_run is None and gate_enabled():
        raise RuntimeError(
            f"run_optimizer needs the research gate (before_run) while {GATE_ENV} is on"
        )
```

After the `if not files: … return None` block, before the run id block:

```python
    # Research gate: needs the concrete symbols and runs before anything is written.
    if before_run is not None:
        symbols = [os.path.basename(f).split("_")[0] for f in files]
        strategy_metadata = before_run(strategy_metadata or {}, symbols)
        strategy = StrategyConfig.from_dict(strategy_metadata)
```

In `main()`: `add_run_arguments(parser)` after the last
`parser.add_argument`. In the run branch, after the strategy is loaded and
the overrides and asset filter are built:

```python
        research = None
        run_id = args.run_id or generate_run_id(args.description)
        if gate_enabled():
            if args.no_save:
                parser.error("--no-save runs cannot be recorded in the research register")
            if args.strategy_file and not strategy_metadata_loaded:
                sys.exit(1)        # load_strategy_from_file already printed why
            try:
                strategy_metadata = prepare_metadata(strategy_metadata, data_config.TIMEFRAME)
            except (OSError, ValueError) as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            research = ResearchRun(
                get_research_dir(),
                run_id,
                Path(results_storage.RESULTS_BASE_PATH) / run_id,
                args.research_id,
                args.note,
                args.reason,
                args.check,
            )
        else:
            if args.check:
                parser.error(f"--check needs the research gate ({GATE_ENV} is off)")
            print(f"Warning: {GATE_ENV}=off - this run is neither gated nor recorded",
                  file=sys.stderr)

        def start(before_run=None):
            return run_optimizer(
                description=args.description,
                save_results=not args.no_save,
                strategy_metadata=strategy_metadata if strategy_metadata else None,
                asset_filter=asset_filter,
                run_id=run_id,
                before_run=before_run,
            )

        result = start() if research is None else run_gated(research, start)
        if result is None:
            sys.exit(1)
```

- `strategy_metadata_loaded`: keep the result of `load_strategy_from_file`
  before the existing `if strategy_metadata is None: strategy_metadata = {}`
  replaces it.
- `results_storage`: add `from fwbg.results import storage as results_storage  # noqa: E402`
  so `RESULTS_BASE_PATH` is read at call time (tests patch it), as
  `create_run_directory` does; `Path` comes from `pathlib` (main.py imports
  it locally in `run_optimizer` today — add a module-level import).

**Step 5:** `tests/research tests/test_cli.py`, then the full suite
(explicit paths if fwbg-mcp collection fails). Expected: only the known
environment failures (broker_ig, premium plugin contracts, premium
multi_timeframe).
**Commit:** `feat(cli): gate every backtest run through the research register`

---

### Task 6: `fwbg research new | amend | verdict | reopen`

**Files:** create `src/fwbg/research/stats.py`,
`src/fwbg/research/commands.py`; modify `src/fwbg/research/cli.py`; tests
`tests/research/test_stats.py`, `test_commands.py`, `test_cli.py`.

**Step 1: `stats.py` tests**

```python
def test_best_result_ignores_holdout_failed_and_missing_sharpe():
    trials = [
        make_trial(trial_id="T-000001", metrics={"EURUSD": {"sharpe": 0.4}}),
        make_trial(trial_id="T-000002", metrics={"GBPUSD": {"sharpe": 0.9}, "X": {"sharpe": None}}),
        make_trial(trial_id="T-000003", kind=TrialKind.HOLDOUT, metrics={"EURUSD": {"sharpe": 2.0}}),
        make_trial(trial_id="T-000004", status=TrialStatus.FAILED, metrics={"EURUSD": {"sharpe": 3.0}}),
    ]
    best = best_result(trials)
    assert (best.trial_id, best.symbol, best.sharpe) == ("T-000002", "GBPUSD", 0.9)


def test_criteria_check():
    assert criteria_check(
        {"min_sharpe": 1.0, "max_p_value": 0.05, "min_trades": 100.0},
        {"sharpe": 0.8, "p_value": 0.01},
    ) == ["min_sharpe fail (0.8 < 1)", "max_p_value pass (0.01 <= 0.05)", "min_trades n/a"]
```

```python
"""Statistics shared by the research views and the verdict."""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from fwbg.research.ledger import Trial, TrialKind

# criterion -> (metric, comparison that passes, sign shown on pass, sign shown on fail)
_CHECKS: dict[str, tuple[str, Callable[[float, float], bool], str, str]] = {
    "min_sharpe": ("sharpe", operator.ge, ">=", "<"),
    "min_trades": ("trades", operator.ge, ">=", "<"),
    "max_drawdown": ("max_drawdown", operator.le, "<=", ">"),
    "min_fold_stability": ("fold_stability", operator.ge, ">=", "<"),
    "max_p_value": ("p_value", operator.le, "<=", ">"),
}


@dataclass(frozen=True)
class BestResult:
    trial_id: str
    symbol: str
    sharpe: float
    metrics: Mapping[str, float | int | None]


def development_trials(trials: Iterable[Trial]) -> list[Trial]:
    """Counted trials that saw only development data (no holdout looks)."""
    return [trial for trial in trials if trial.counted and trial.kind is not TrialKind.HOLDOUT]


def best_result(trials: Iterable[Trial]) -> BestResult | None:
    """The symbol evaluation with the highest Sharpe ratio among development trials."""
    candidates = [
        BestResult(str(trial.trial_id), symbol, float(sharpe), values)
        for trial in development_trials(trials)
        for symbol, values in trial.metrics.items()
        if isinstance(sharpe := values.get("sharpe"), (int, float))
    ]
    return max(candidates, key=lambda candidate: candidate.sharpe, default=None)


def criteria_check(
    criteria: Mapping[str, float], metrics: Mapping[str, float | int | None]
) -> list[str]:
    """One line per criterion: pass, fail or n/a (metric missing)."""
    lines = []
    for name, threshold in criteria.items():
        metric, passes, ok_sign, fail_sign = _CHECKS[name]
        value = metrics.get(metric)
        if not isinstance(value, (int, float)):
            lines.append(f"{name} n/a")
        elif passes(value, threshold):
            lines.append(f"{name} pass ({value:g} {ok_sign} {threshold:g})")
        else:
            lines.append(f"{name} fail ({value:g} {fail_sign} {threshold:g})")
    return lines


def evaluations(trials: Iterable[Trial]) -> int:
    """Counted symbol evaluations: the N a multiple-testing correction uses."""
    return sum(len(trial.metrics) for trial in trials if trial.counted)
```

**Step 2: `commands.py` tests** (`gate_register`, `today = date(2026, 10, 20)`):

- `new_card` writes `strategies/S-0008-<slug>.md` next to `S-0007`, status
  active, criteria = template merged with overrides, matching
  `registered_criteria_hash`, cutoff default 2025-04-20 (18 months), body
  sections Hypothesis/Mechanism/History/Verdict; `git status` clean.
- `new_card` refuses a future cutoff, `min_sharp`, a title with a line
  break, and a title without ASCII words when no slug is given.
- `months_before(date(2026, 8, 31), 18) == date(2025, 2, 28)`.
- `amend_criteria` appends (today, reason, new hash); afterwards
  `gate.check` allows a run although the card's last trial has the old hash;
  refuses unchanged criteria, an unknown name, a multi-line reason, and a
  card whose criteria were edited by hand.
- `write_verdict` on the active card sets the status and appends
  `### 2026-10-20: rejected` with reason, reopen condition, N lines and the
  best result's criteria check under `## Verdict`; refuses a non-active
  card, the status `active`, and a card with a live pending run (marker
  with `os.getpid()`); records a stale marker of the card first (its trial
  appears in N).
- `reopen` turns parked/rejected into active with
  `- 2026-10-20: reopened from parked: <reason>` under `## History`
  (created before `## Verdict` when missing); refuses active and candidate.
- Every command leaves `git status --porcelain` empty.

**Step 3: Implement `commands.py`**

```python
"""Write commands of the research register: register, amend, verdict, reopen.

Each command re-reads the card under ``register_lock``, rewrites it and
commits all register paths. If the commit fails, the change stays on disk
and the next register commit includes it.
"""

from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

import yaml

from fwbg.research.card import (
    CriteriaAmendment,
    ResearchCard,
    Status,
    UniqueKeyLoader,
    add_section_entry,
    criteria_hash,
    load_card,
    next_card_id,
    replace_card,
    slugify,
    validate_criteria,
    write_card,
)
from fwbg.research.ledger import LEDGER_FILENAME, Trial, TrialKind, read_trials
from fwbg.research.pending import live_runs, recover_stale
from fwbg.research.repo import CRITERIA_TEMPLATE, REGISTER_PATHS, commit_register, register_lock
from fwbg.research.stats import best_result, criteria_check, evaluations

DEFAULT_HOLDOUT_MONTHS = 18
VERDICT_STATUSES = (Status.PARKED, Status.REJECTED, Status.CANDIDATE)
_PLACEHOLDER = "To be written before the first iteration."
_BODY = (
    "## Hypothesis\n\n{hypothesis}\n\n## Mechanism\n\n{mechanism}\n\n"
    "## History\n\n## Verdict\n\nOpen.\n"
)


class CommandError(Exception):
    """A research command was refused; nothing was changed."""


def months_before(day: date, months: int) -> date:
    """The same day *months* earlier, clamped to the end of a shorter month."""
    year, month = divmod(day.year * 12 + day.month - 1 - months, 12)
    return date(year, month + 1, min(day.day, calendar.monthrange(year, month + 1)[1]))


def new_card(
    research_dir: Path,
    title: str,
    tags: Sequence[str],
    today: date,
    *,
    cutoff: date | None = None,
    slug: str | None = None,
    criteria: Mapping[str, float] | None = None,
    hypothesis: str | None = None,
    mechanism: str | None = None,
) -> Path:
    """Register a new active card with criteria frozen from the template plus *criteria*."""
    _one_line(title, "title")
    holdout_cutoff = cutoff or months_before(today, DEFAULT_HOLDOUT_MONTHS)
    if holdout_cutoff > today:
        raise CommandError(f"holdout cutoff {holdout_cutoff} lies in the future")
    frozen = validate_criteria({**_template(research_dir), **(criteria or {})})
    body = _BODY.format(hypothesis=hypothesis or _PLACEHOLDER, mechanism=mechanism or _PLACEHOLDER)
    with register_lock(research_dir):
        card = ResearchCard(
            id=next_card_id(research_dir, read_trials(research_dir / LEDGER_FILENAME)),
            title=title,
            status=Status.ACTIVE,
            tags=_tags(tags),
            created=today,
            holdout_cutoff=holdout_cutoff,
            holdout_contaminated=False,
            success_criteria=frozen,
            registered_criteria_hash=criteria_hash(frozen),
            criteria_amendments=(),
            body=body,
        )
        path = write_card(research_dir, card, slug or slugify(title))
        commit_register(research_dir, REGISTER_PATHS, f"research: register {card.id} {title}")
    return path


def amend_criteria(
    research_dir: Path, card_id: str, changes: Mapping[str, float], reason: str, today: date
) -> ResearchCard:
    """Change success criteria and record the change as an amendment."""
    _one_line(reason, "reason")
    if not changes:
        raise CommandError("nothing to amend; pass --set name=value")
    with register_lock(research_dir):
        card = _require(research_dir, card_id)
        recorded = (
            card.criteria_amendments[-1].criteria_hash
            if card.criteria_amendments
            else card.registered_criteria_hash
        )
        if card.criteria_hash != recorded:
            raise CommandError(
                f"the criteria of {card_id} were edited by hand; restore them before amending"
            )
        criteria = validate_criteria({**card.success_criteria, **changes})
        new_hash = criteria_hash(criteria)
        if new_hash == card.criteria_hash:
            raise CommandError(f"the criteria of {card_id} are unchanged")
        amendment = CriteriaAmendment(date=today, reason=reason, criteria_hash=new_hash)
        updated = replace(
            card,
            success_criteria=criteria,
            criteria_amendments=(*card.criteria_amendments, amendment),
        )
        replace_card(research_dir, updated)
        commit_register(research_dir, REGISTER_PATHS, f"research: amend {card_id} criteria")
    return updated


def write_verdict(
    research_dir: Path, card_id: str, status: Status, reason: str, reopen_if: str, today: date
) -> ResearchCard:
    """Close an active card with a verdict and the statistics it rests on."""
    _one_line(reason, "reason")
    _one_line(reopen_if, "reopen condition")
    if status not in VERDICT_STATUSES:
        raise CommandError(f"a verdict sets parked, rejected or candidate, not {status.value}")
    with register_lock(research_dir):
        recover_stale(research_dir)
        card = _require(research_dir, card_id)
        if card.status is not Status.ACTIVE:
            raise CommandError(f"{card_id} is {card.status.value}; reopen it before a new verdict")
        running = live_runs(research_dir, card_id)
        if running:
            names = ", ".join(run.run_id for run in running)
            raise CommandError(f"{card_id} has runs in progress ({names}); wait for them")
        trials = read_trials(research_dir / LEDGER_FILENAME)
        entry = verdict_entry(card, trials, status, reason, reopen_if, today)
        updated = replace(card, status=status, body=add_section_entry(card.body, "Verdict", entry))
        replace_card(research_dir, updated)
        commit_register(research_dir, REGISTER_PATHS, f"research: verdict {card_id} {status.value}")
    return updated


def reopen(research_dir: Path, card_id: str, reason: str, today: date) -> ResearchCard:
    """Turn a parked or rejected card active again, recording why."""
    _one_line(reason, "reason")
    with register_lock(research_dir):
        card = _require(research_dir, card_id)
        if card.status not in (Status.PARKED, Status.REJECTED):
            raise CommandError(f"{card_id} is {card.status.value}; only parked or rejected reopen")
        entry = f"- {today.isoformat()}: reopened from {card.status.value}: {reason}"
        body = add_section_entry(card.body, "History", entry, before=("Verdict",))
        updated = replace(card, status=Status.ACTIVE, body=body)
        replace_card(research_dir, updated)
        commit_register(research_dir, REGISTER_PATHS, f"research: reopen {card_id}")
    return updated


def verdict_entry(
    card: ResearchCard,
    trials: Sequence[Trial],
    status: Status,
    reason: str,
    reopen_if: str,
    today: date,
) -> str:
    """The dated verdict text: reasoning plus the statistics it rests on."""
    own = [trial for trial in trials if trial.strategy == card.id]
    looks = [str(trial.trial_id) for trial in own if trial.kind is TrialKind.HOLDOUT]
    lines = [
        f"### {today.isoformat()}: {status.value}",
        "",
        f"Reason: {reason}",
        f"Reopen if: {reopen_if}",
        f"Trials: {sum(t.counted for t in own)} counted on {card.id} "
        f"({evaluations(own)} symbol evaluations); {sum(t.counted for t in trials)} counted "
        f"in the register ({evaluations(trials)} symbol evaluations).",
        f"Criteria versions used: {len({trial.criteria_hash for trial in own})}.",
        f"Holdout: {'used by ' + ', '.join(looks) if looks else 'unused'}.",
    ]
    best = best_result(own)
    if best is None:
        lines.append("Best development result: none with a Sharpe ratio.")
    else:
        check = "; ".join(criteria_check(card.success_criteria, best.metrics))
        lines.append(f"Best development result: {best.trial_id} {best.symbol}: {check}.")
    lines.append("Deflated Sharpe ratio: not computed yet (register step 5).")
    return "\n".join(lines)


def _require(research_dir: Path, card_id: str) -> ResearchCard:
    card = load_card(research_dir, card_id)
    if card is None:
        raise CommandError(f"no research card {card_id}")
    return card


def _template(research_dir: Path) -> dict[str, object]:
    path = research_dir / CRITERIA_TEMPLATE
    if not path.exists():
        raise CommandError(
            f"research register not initialised at {research_dir}; run 'fwbg research init'"
        )
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except (yaml.YAMLError, ValueError) as exc:
        raise CommandError(f"{CRITERIA_TEMPLATE}: {exc}") from exc
    if not isinstance(data, dict):
        raise CommandError(f"{CRITERIA_TEMPLATE} must be a mapping of criteria")
    return data


def _tags(tags: Sequence[str]) -> tuple[str, ...]:
    cleaned = tuple(tag.strip() for tag in tags if tag.strip())
    for tag in cleaned:
        _one_line(tag, "tag")
    return cleaned


def _one_line(value: str | None, what: str) -> None:
    # The gate's note rule: non-blank, no line break of any kind.
    if value is None or not value.strip() or value.splitlines() != [value]:
        raise CommandError(f"{what} must be one non-empty line, got {value!r}")
```

**Step 4: CLI** (`src/fwbg/research/cli.py`; description "Manage the
research register: cards, trials and the legacy backfill."):

```text
new TITLE --tags a,b [--cutoff YYYY-MM-DD] [--slug SLUG] [--set name=value ...]
          [--hypothesis TEXT] [--mechanism TEXT]
amend S-NNNN --set name=value [--set ...] --reason TEXT
verdict S-NNNN {parked,rejected,candidate} --reason TEXT --reopen-if TEXT
reopen S-NNNN --reason TEXT
```

`--cutoff` accepts only `YYYY-MM-DD` (regex, then `date.fromisoformat`);
`--set` (`action="append"`) parses `name=value` with `float`; malformed →
argparse error. Output lines: `Registered S-0025: <path>`,
`Amended S-0025 (amendment 2)`, `S-0025 is now rejected`,
`S-0025 is active again`. Add `CommandError` and `PendingError` to
`_OPERATOR_ERRORS`; pass `date.today()`. CLI tests: one happy path per
command through `main([...])`, plus `new --set min_sharp=1` → exit 1 with
`Error:` and no traceback, `--cutoff 2026-13-01` → exit 2.

**Step 5:** `tests/research` → PASS.
**Commit:** `feat(research): add new, amend, verdict and reopen commands`

---

### Task 7: `fwbg research list | show`

**Files:** create `src/fwbg/research/views.py`; modify `cli.py`; test
`tests/research/test_views.py`.

**Tests:**
- `list_lines(register)`: one line per card, sorted by id —
  `S-0007  active    3 trials  best Sharpe 0.90 (T-000002 GBPUSD)  holdout unused  <title>`;
  `status=` and `tag=` filter; a card file that does not parse yields
  `S-0009  unreadable: <error>` instead of aborting.
- `show_text(register, "S-0007")`: header (id, title, status, tags, created,
  cutoff, contamination); criteria with their state (`registered`,
  `amended 1x (latest 2026-11-02)`, or `edited by hand - runs are refused`);
  the body; runs in progress (live markers of the card); one timeline row
  per trial (id, date, kind, status, symbols, best Sharpe, note); coverage.
- `coverage(configs)` over three stored configs that differ in
  `strategy.pipeline.indicators[0].params.range_bars` and `symbols`: lists
  exactly the varied paths with their sorted distinct values, always lists
  `symbols`, `strategy.timeframe`, `strategy.start_date`,
  `strategy.end_date`; a path missing from one config shows `-` for it; more
  than 10 values → `… (+N more)`.
- Trials whose `configs/<hex>.json` is missing (study trials) are left out of
  the coverage, which says how many configs it covers.
- `show` of an unknown id → `CommandError` (exit 1 through the CLI).

**Implement** `views.py`:

```python
def list_lines(research_dir: Path, status: Status | None = None, tag: str | None = None) -> list[str]
def show_text(research_dir: Path, card_id: str) -> str
def coverage(configs: Sequence[Mapping[str, Any]]) -> list[str]
```

- Cards from `strategies/S-*.md` names (`CARD_FILE_RE`), loaded with
  `load_card`; the ledger is read once.
- Flattening: mappings by sorted key (`a.b`), lists whose items are all
  mappings or lists by index (`a[0]`), every other value is a leaf rendered
  with `canonical_json`.
- Criteria state: the gate's rule (latest amendment hash, else
  `registered_criteria_hash`, compared with `card.criteria_hash`).
- Best Sharpe and N from `fwbg.research.stats`.

CLI: `list [--status S] [--tag T]`, `show S-NNNN` → stdout.
**Commit:** `feat(research): add list and show views with coverage`

---

### Task 8: Rules for AI sessions, docs, verification

**Files:** `CLAUDE.md` (append), create `AGENTS.md` (same section),
`docs/plans/2026-10-19-research-register-design.md` (status line; CLI
table: `check` → `python -m fwbg.cli --check …`; `--no-save` and pending
marker decisions; rollout step 3 done).

**The rule** (public repository: no results, no real card ids):

```markdown
## Research register

Every backtest run (`python -m fwbg.cli --strategy-file …`) is gated by the
private research register in `$FWBG_WORKSPACE/research`.

- Before proposing a test, look at what was tried:
  `python -m fwbg.cli research list` and `python -m fwbg.cli research show S-NNNN`.
- A new idea gets a card first: `python -m fwbg.cli research new "<title>" --tags …`,
  with its hypothesis and mechanism written before the first run.
- Every run passes `--research-id S-NNNN --note "<what changed and why>"`;
  re-running an identical configuration needs `--reason`. Dry-run with `--check`.
- Never set `FWBG_RESEARCH_GATE`; it exists for the test suite only.
- A strategy is closed only by a deliberate `research verdict`, never after
  one failed configuration.
- Never copy research results, hypotheses or card contents into this
  repository; it is public.
```

**Verification:**
1. `ruff check src/ packages/`, `mypy --follow-imports=silent src/fwbg/research`,
   black on touched files.
2. Full suite: only the known environment failures.
3. Manual smoke on a **copy** of the real register (never the real one):
   `cp -a ~/fwbg/research <scratchpad>/register`, then with
   `FWBG_RESEARCH_DIR=<scratchpad>/register` and
   `FWBG_TEST_RESULTS_DIR=<scratchpad>/results`: `research list`,
   `research show S-0004`, `research new "Smoke card" --tags smoke`, a
   `--check` run of a real strategy file on one asset against the new card,
   one short real run on that card (then `show` lists its trial and its
   run's `config.json` has the `research` block), a second identical run →
   refused as duplicate, `research reopen S-0019 --reason …`,
   `research verdict S-0019 parked --reason … --reopen-if …`;
   `git -C <scratchpad>/register log` shows one commit per write and
   `--check` wrote nothing.
4. `git log --format=%B origin/develop..HEAD | grep -ciE 'claude|co-authored|generated'` → 0.

**Commit:** `docs: document the research gate for AI sessions`

## Implementation notes (step 3)

- `legacy.run_outcome(run_dir, *, lenient=False)`: the plan's
  `count_unknown` is named `lenient`; live runs (`pending.finish`) count an
  unknown status or unreadable symbol file as evaluated.
- `pending.recover_stale` finishes each stale marker on its own; one that
  fails stays and is logged. Markers carry the writing `host`; markers from
  another host count as alive.
- Run ids are used once: a run id in the ledger, with `config.json` or
  `grid_details/`, or with a marker is refused (`RUN_ID_IN_USE`); a
  malformed one is `INVALID_RUN_ID`; an unreadable card or ledger is
  `REGISTER_UNREADABLE`.
- `--check` and refused runs write nothing about the checked run but may
  record other runs' stale markers (recovery runs first).
- The gate also denies `CRITERIA_CHANGED` when any of the card's trials ran
  under a criteria hash that is neither the registration nor an amendment.
- `amend` works only on `active` cards and refuses when the card's
  criteria do not match the recorded registration/amendments; `verdict`
  refuses while the card has pending runs.
- `add_section_entry` ignores `## ` lines in fenced code blocks and raises
  `CardError` on a duplicated target section.
- Coverage keys list items by their `name`, `[*]` otherwise.

## Notes for later steps

- Step 4 passes `research_id`, `note`, `reason` from `RunStartRequest` to the
  CLI and preflights with the same gate; until then API, dashboard and
  fwbg-agents runs are refused (operator decision 2026-10-20). The API
  container mounts the named volume `fwbg-workspace`, not the host
  workspace: decide its register (own or bind mount) together with git in
  the image, committer identity, signing and hooks. The API pre-creates the
  run directory for logs; the run-id rule already allows a bare directory.
- Step 4: `register_trial()` for scripts and `fwbg portfolio`; `reconcile`
  for anything the pending markers do not cover.
- Step 5: re-check duplicate and holdout rules under the lock when
  appending, before `--holdout` ships (live duplicates are already refused
  through the markers).
- If `--no-save` is needed again, add the design's staging run directory
  instead of lifting the refusal.
