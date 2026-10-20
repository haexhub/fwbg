# Research Register Step 4: API, `register_trial`, `reconcile` — Implementation Plan

Execution: task by task, each task reviewed for spec compliance and code quality.

**Goal:** API-started runs, study scripts and the Docker API container record into the same
private research register as the CLI; the API exposes cards for fwbg-agents; `reconcile`
repairs an uncommitted register.

**Architecture:** The CLI gate stays authoritative. `POST /api/runs/start` runs the CLI once
with `--check-json` (a machine-readable `--check`) before it spawns the run, so the API refuses
with the gate's own denials and identical fingerprints without duplicating asset resolution.
A new router `fwbg.api.research` wraps the existing card commands and views. The API container
runs as root on the host's register (bind mount); register writes hand created files back to
the register's owner and git trusts the directory explicitly.

**Tech Stack:** Python 3.14, FastAPI, pytest, git; Docker Compose.

Design: `docs/plans/2026-10-19-research-register-design.md` (step 4). Step 3 plan:
`docs/plans/2026-10-20-research-register-gate.md`.

---

## Operator decisions (2026-10-20)

- The Docker stack runs on the operator's machine. The API container gets the host register
  `~/fwbg/research` as a bind mount at `/root/fwbg/research`; there is one register.
- `git` is installed in the image. The container commits as `fwbg-api <fwbg-api@localhost>`
  (env in `docker-compose.yml`); it has no signing key and no hooks, so its commits are
  unsigned. The operator's own commits stay signed (their global git config applies locally).
- The container runs as root. Files it creates in the register are chowned back to the owner
  of the register directory after every register write; otherwise the host user could no
  longer write `.git/objects`. Git refuses a repository owned by another user ("dubious
  ownership"), so every register git call passes `-c safe.directory=<register>`.
- `hostname: fwbg-api` in compose: pending markers carry the host name, and a marker from
  another host always looks alive. A stable name lets the restarted container recover its own
  killed runs.
- `/api/runs/preview` is **not** a trial: it shows one symbol's signals with fixed TP/SL and
  reports no aggregated metrics; fwbg-agents does not call it. (Design open question closed.)
- `reconcile` validates, records stale markers, commits register paths and **reports** runs
  missing from the ledger. It does not index them itself: assigning a run to a card needs the
  operator (`research backfill scan/apply`). (Design wording adjusted in Task 7.)
- `register_trial` lives in `fwbg.research.study` (not re-exported from `fwbg.research`, to
  avoid import cycles). It requires the evaluated `end_date` in the config: a study has
  already run, so the gate cannot clamp its window.
- The holdout flag stays out of the API until step 5.

## Rules for every task

- Work only in the step-4 worktree (branch `feat/research-api`). Never touch the main
  checkout, never write under `~/fwbg`. No bare `git stash`.
- TDD: failing test first, then the code. Tests use `tmp_path` registers
  (`init_research_repo(tmp_path / "research")`) and set `FWBG_RESEARCH_DIR`,
  `FWBG_TEST_RESULTS_DIR`, `FWBG_STRATEGIES_DIR` via `monkeypatch.setenv`.
- `tests/conftest.py` sets `FWBG_RESEARCH_GATE=off`; gate tests switch it on with
  `monkeypatch.setenv("FWBG_RESEARCH_GATE", "on")` or patch `gate_enabled`.
- 500-LoC limit per file (spaex). `ponytail:` comments for deliberate shortcuts.
- English commit messages, short, no AI/model references or trailers. Stage only touched files.
- Before reporting: `black`/`ruff` on touched files, `ruff check src/ packages/`,
  `mypy --follow-imports=silent src/fwbg/research`, and
  `PYTHONPATH=src ../../.venv/bin/python -m pytest -p no:cacheprovider tests/research tests/test_cli.py tests/test_api_run_spawn.py tests/test_api_run_start_validation.py -q`
  plus the task's own test files.

---

### Task 1: Register writes work from a root container

**Files:**
- Modify: `src/fwbg/research/repo.py` (`register_lock`, `_git`, new `restore_owner`)
- Test: `tests/research/test_repo_owner.py` (new)

**Step 1: Failing tests**

```python
"""Register writes from a root process (the API container) on a host-owned register."""

import subprocess
from pathlib import Path

from fwbg.research import repo
from fwbg.research.repo import init_research_repo, register_lock, restore_owner


def test_restore_owner_is_a_no_op_for_a_normal_user(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(repo.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(repo.os, "chown", lambda *a, **k: calls.append(a))
    init_research_repo(tmp_path)
    restore_owner(tmp_path)
    assert calls == []


def test_restore_owner_chowns_everything_root_created(tmp_path, monkeypatch):
    init_research_repo(tmp_path)
    calls = []
    monkeypatch.setattr(repo.os, "geteuid", lambda: 0)
    monkeypatch.setattr(repo, "_owner", lambda path: (4242, 4343))
    monkeypatch.setattr(
        repo.os, "chown", lambda path, uid, gid, **k: calls.append((Path(path), uid, gid))
    )
    restore_owner(tmp_path)
    changed = {path.relative_to(tmp_path).as_posix() for path, _, _ in calls}
    assert "trials.jsonl" in changed
    assert any(name.startswith(".git/objects/") for name in changed)
    assert {(uid, gid) for _, uid, gid in calls} == {(4242, 4343)}


def test_restore_owner_skips_a_root_owned_register(tmp_path, monkeypatch):
    init_research_repo(tmp_path)
    calls = []
    monkeypatch.setattr(repo.os, "geteuid", lambda: 0)
    monkeypatch.setattr(repo, "_owner", lambda path: (0, 0))
    monkeypatch.setattr(repo.os, "chown", lambda *a, **k: calls.append(a))
    restore_owner(tmp_path)
    assert calls == []


def test_register_lock_restores_the_owner_on_exit(tmp_path, monkeypatch):
    init_research_repo(tmp_path)
    seen = []
    monkeypatch.setattr(repo, "restore_owner", lambda path: seen.append(path))
    with register_lock(tmp_path):
        assert seen == []
    assert seen == [tmp_path]


def test_register_lock_restores_the_owner_after_an_error(tmp_path, monkeypatch):
    init_research_repo(tmp_path)
    seen = []
    monkeypatch.setattr(repo, "restore_owner", lambda path: seen.append(path))
    try:
        with register_lock(tmp_path):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert seen == [tmp_path]


def test_git_trusts_the_register_directory(tmp_path, monkeypatch):
    init_research_repo(tmp_path)
    seen = []
    real = subprocess.Popen

    def spy(args, **kwargs):
        seen.append(args)
        return real(args, **kwargs)

    monkeypatch.setattr(repo.subprocess, "Popen", spy)
    repo._git(tmp_path, "status")
    assert f"safe.directory={tmp_path.resolve()}" in seen[0]
```

Run: `PYTHONPATH=src ../../.venv/bin/python -m pytest -p no:cacheprovider tests/research/test_repo_owner.py -q`
Expected: FAIL (`restore_owner` missing).

**Step 2: Implementation** (in `repo.py`)

```python
@contextmanager
def register_lock(research_dir: Path) -> Iterator[None]:
    """...(keep the docstring)...

    On exit, files a root process created are handed to the register's owner
    (:func:`restore_owner`).
    """
    with (research_dir / REGISTER_LOCK).open("a", encoding="utf-8") as handle:
        # ponytail: (keep)
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            restore_owner(research_dir)


def restore_owner(research_dir: Path) -> None:
    """Give what a root process created in the register to the register directory's owner.

    The API container runs as root on the register bind-mounted from the host;
    root-owned files (``.git/objects`` above all) would lock the host user out of
    their own register. Never raises: a failure is logged and the next write retries.
    """
    if os.geteuid() != 0:
        return
    try:
        uid, gid = _owner(research_dir)
        if uid == 0:
            return
        for root, dirs, files in os.walk(research_dir):
            for name in (*dirs, *files):
                path = os.path.join(root, name)
                status = os.lstat(path)
                if (status.st_uid, status.st_gid) != (uid, gid):
                    os.chown(path, uid, gid, follow_symlinks=False)
    except OSError as exc:
        log.warning("could not hand register files back to their owner: %s", exc)


def _owner(path: Path) -> tuple[int, int]:
    status = path.stat()
    return status.st_uid, status.st_gid
```

Add `import logging` / `log = logging.getLogger(__name__)` at the top. In `_git`, always pass
`safe.directory`:

```python
    # The API container (root) works on a register its host user owns; git refuses that
    # without an explicit safe.directory, which only protected config (here: -c) may set.
    settings = (f"safe.directory={cwd.resolve()}", *config)
    overrides = [option for setting in settings for option in ("-c", setting)]
```

Note: `os.walk` lists the directory once; `restore_owner` runs after `commit_register`
inside the lock, so it sees the new git objects. In the test, `_owner` is patched; the files
created by the test belong to the test user, which differs from `(4242, 4343)`.

**Step 3:** Run the new tests and `tests/research` → PASS.

**Step 4: Commit** `fix(research): let a root container write a host-owned register`

---

### Task 2: `--check-json` — machine-readable gate check

**Files:**
- Modify: `src/fwbg/research/runs.py` (`add_run_arguments`, `run_gated`, new `CHECK_PREFIX`,
  `check_result`, `parse_check_output`)
- Modify: `src/fwbg/cli/main.py:1006-1043` (`check_only=args.check or args.check_json`,
  gate-off error for both flags, pass `as_json=args.check_json` to `run_gated`)
- Test: `tests/research/test_check_json.py` (new); extend `tests/research/test_run_cli.py`
  only if a CLI-level test is cheap there (one allowed and one refused case through `main()`).

**Protocol** (one module owns both sides):

```python
CHECK_PREFIX = "RESEARCH_CHECK "


def check_result(
    denials: Sequence[Denial] = (),
    decision: Decision | None = None,
    symbols: Sequence[str] = (),
) -> dict[str, Any]:
    """The ``--check-json`` payload: allowed with kind, fingerprint, symbols and window,
    or refused with every denial and its HTTP status."""
    if denials:
        return {
            "allowed": False,
            "denials": [
                {"code": d.code.value, "status": d.code.http_status, "message": d.message}
                for d in denials
            ],
        }
    assert decision is not None
    return {
        "allowed": True,
        "kind": decision.kind.value,
        "fingerprint": decision.fingerprint,
        "symbols": list(symbols),
        "window": run_window(decision.effective_metadata),
        "denials": [],
    }


def parse_check_output(stdout: str) -> dict[str, Any] | None:
    """The last ``--check-json`` payload in *stdout*; ``None`` if the CLI printed none."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(CHECK_PREFIX):
            try:
                payload = json.loads(line.removeprefix(CHECK_PREFIX))
            except ValueError:
                return None
            return payload if isinstance(payload, dict) else None
    return None
```

`add_run_arguments` gains
`--check-json` (`action="store_true"`, help "like --check; also print the result as one JSON line").
`run_gated(research, run, *, as_json=False)`: on `GateDenied` keep the stderr lines and, with
`as_json`, also `print(CHECK_PREFIX + json.dumps(check_result(exc.denials)), flush=True)`
before `SystemExit(2)`; on `GateChecked` keep `_print_check` and, with `as_json`, print
`check_result(decision=exc.decision, symbols=exc.symbols)` before `SystemExit(0)`.

**Tests** (`test_check_json.py`): `check_result` for an allowed decision and for two denials
(codes, statuses 409/422, messages); `parse_check_output` finds the last prefixed line among
noise, returns `None` without one and for broken JSON; `run_gated(..., as_json=True)` with a
fake `run` that raises `GateDenied` / `GateChecked` prints a parseable payload to stdout and
exits 2 / 0 (use `capsys` and `pytest.raises(SystemExit)`).

**Commit** `feat(research): add a machine-readable --check-json`

---

### Task 3: `POST /api/runs/start` passes research fields and preflights

**Files:**
- Modify: `src/fwbg/api/runs.py` (`RunStartRequest`, `start_run`, new `_cli_command`,
  `_preflight`, `PREFLIGHT_TIMEOUT`)
- Test: `tests/test_api_run_research.py` (new)

`RunStartRequest` gains `research_id: Optional[str] = None`, `note: Optional[str] = None`,
`reason: Optional[str] = None`. Move the command building out of `start_run` unchanged into
`_cli_command(body, strategy_file, job_id) -> list[str]`, then append
`--research-id`, `--note`, `--reason` when set (as separate argv items; no shell).

`start_run` order: validate → reserve → strategy exists → build `cmd` and `env` (as today) →
**if `gate_enabled()`: `_preflight(cmd, env)`** → spawn. Every refusal releases the slot with
`_run_service.release(...)` before raising.

```python
# A --check run imports fwbg and selects the data files; it never loads market data.
PREFLIGHT_TIMEOUT = 120.0


def _preflight(cmd: list[str], env: dict) -> None:
    """Run the CLI's research gate (``--check-json``) before the real run starts.

    The CLI stays authoritative (it gates again when the run starts); this gives
    callers such as fwbg-agents the gate's refusal as an HTTP error instead of a
    run that dies.

    Raises:
        HTTPException: 409/422 with every denial, 422 if the CLI failed before
            its gate (e.g. an unloadable strategy), 504 on timeout.
    """
    try:
        result = subprocess.run(
            [*cmd, "--check-json"],
            capture_output=True,
            text=True,
            env=env,
            timeout=PREFLIGHT_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "research preflight timed out") from exc
    payload = parse_check_output(result.stdout)
    if payload is None:
        tail = (result.stderr.strip() or result.stdout.strip())[-500:]
        raise HTTPException(422, f"research preflight failed: {tail}")
    if not payload.get("allowed"):
        denials = payload.get("denials", [])
        status = 409 if any(d.get("status") == 409 for d in denials) else 422
        raise HTTPException(
            status, {"message": "research gate refused the run", "denials": denials}
        )
```

Import `gate_enabled` and `parse_check_output` from `fwbg.research.runs` at module top
(the tests patch `fwbg.api.runs.gate_enabled`). The `--run-id job_id` in `cmd` makes the
preflight check the real run id; the reserved run directory holds only `run_state.json`,
which the gate allows.

**Tests** (`test_api_run_research.py`; `TestClient(create_app())`; a strategy JSON in a
tmp `FWBG_STRATEGIES_DIR`; tmp `FWBG_TEST_RESULTS_DIR` and `FWBG_RESEARCH_DIR`; patch
`fwbg.api.runs.subprocess.run` to return a `CompletedProcess` whose stdout carries a
`CHECK_PREFIX` payload; patch `fwbg.api.runs._spawn_cli_process` to record the command and
return a fake process with a `pid`; patch `gate_enabled` to `True`):

1. a refusal with a 409 and a 422 denial → 409, `detail.denials` has both, no spawn, and the
   next start is not 429 (slot released);
2. only 422 denials → 422;
3. allowed → spawn called once; its command contains `--research-id S-0001`, `--note`, its
   text, `--reason` when given, and not `--check-json`; the preflight command is the spawn
   command plus `--check-json`;
4. CLI exits 1 without a payload → 422 with the stderr tail, slot released;
5. `TimeoutExpired` → 504, slot released;
6. gate off (`gate_enabled` → `False`) → no preflight, spawn as before.

**Commit** `feat(api): pass research fields to runs and preflight the gate`

---

### Task 4: Research endpoints (`/api/research`) and `actor` on status changes

**Files:**
- Modify: `src/fwbg/research/commands.py` (`write_verdict(..., actor=None)`,
  `reopen(..., actor=None)`, `verdict_entry(..., actor=None)`)
- Modify: `src/fwbg/research/views.py` (new `card_summary`, `card_data`)
- Create: `src/fwbg/api/research.py` (router)
- Modify: `src/fwbg/api/__init__.py` (include the router with `prefix="/api"`)
- Test: `tests/research/test_commands.py` (actor cases), `tests/test_api_research.py` (new)

**Actor:** `verdict_entry` adds `By: {actor}` after the `Reopen if:` line when `actor` is set;
`reopen` writes `- {date}: reopened from {status} by {actor}: {reason}` when set (unchanged
text without an actor). `_one_line(actor, "actor")` validates it. The CLI does not pass one.

**Views** (pure data, no formatting):

```python
def card_summary(card: ResearchCard, own: Sequence[Trial], running: int) -> dict[str, Any]:
    best = best_result(own)
    return {
        "id": card.id,
        "title": card.title,
        "status": card.status.value,
        "tags": list(card.tags),
        "created": card.created.isoformat(),
        "holdout_cutoff": card.holdout_cutoff.isoformat(),
        "trials": sum(trial.counted for trial in own),
        "best": None if best is None else {
            "trial": best.trial_id, "symbol": best.symbol, "metrics": dict(best.metrics)
        },
        "holdout_used": any(trial.kind is TrialKind.HOLDOUT for trial in own),
        "running": running,
    }


def card_data(card: ResearchCard) -> dict[str, Any]:
    """The whole card as JSON-ready data: front matter plus the Markdown body."""
```

(`BestResult` field names: check `stats.py:22` and adapt.) `card_data` serialises every
`ResearchCard` field (dates as ISO strings, amendments as dicts, status as its value).

**Router** `src/fwbg/api/research.py` (`APIRouter(prefix="/research", tags=["research"])`):

| Endpoint | Behaviour |
|---|---|
| `GET /research?status=&tag=` | `card_summary` per readable card in id order; unreadable cards as `{"id", "error"}` entries; `running` from `live_runs(research_dir, card.id)`. |
| `GET /research/{card_id}` | `{"card": card_data, "trials": [json.loads(t.to_json()) …own], "pending": [run ids of the card's markers], "coverage": coverage lines as in `show`}`; 404 if no card. Reuse what `views._coverage_section` computes (factor out a helper returning the lines if needed; keep `show_text` output unchanged). |
| `POST /research` | body `{title, tags: list[str], hypothesis, mechanism, cutoff?: date, criteria?: dict[str, float]}` → `new_card(...)` → 201 `{"card": card_data}`. Hypothesis and mechanism are required (min length 1) for API callers. |
| `POST /research/{card_id}/status` | body `{status, reason, actor, reopen_if?}`; `status == "active"` → `reopen`; a verdict status → `write_verdict` (422 if `reopen_if` missing) → `{"card": card_data}`. `actor` required, non-empty. |

`card_id` must match `CARD_ID_RE` (else 422). Errors: missing card → 404; `CommandError` →
409 with its message; `ValueError` from `validate_criteria` → 422; `CardError`,
`LedgerError` → 409 `"research register unreadable: …"`. Use `get_research_dir(create=False)`
and `date.today()`. Sync endpoints (`def`), like the rest of the API.

**Tests** (`tests/test_api_research.py`, tmp register via `FWBG_RESEARCH_DIR`): create a card
via POST → 201, file exists, committed; list returns it with `trials == 0`; status filter and
tag filter; GET detail has card, empty trials, coverage list; unknown id → 404; malformed id →
422; verdict without `reopen_if` → 422; verdict → parked with `By: <actor>` in the card body;
reopen → active with the actor in the History entry; verdict on a parked card → 409; missing
`actor` → 422.

**Commit(s)** `feat(research): record the actor of verdicts and reopens`,
`feat(api): add research register endpoints`

---

### Task 5: `register_trial()` for study scripts

**Files:**
- Create: `src/fwbg/research/study.py`
- Modify: `CLAUDE.md`, `AGENTS.md` (rule line)
- Test: `tests/research/test_study.py` (new)

```python
def register_trial(
    research_id: str,
    config: Mapping[str, Any],
    symbols: Sequence[str],
    metrics: Mapping[str, Mapping[str, float | int | None]],
    note: str,
    *,
    reason: str | None = None,
    failed: bool = False,
    run_id: str | None = None,
    research_dir: Path | None = None,
) -> Trial:
    """Record one evaluated variant of a study script as an iteration trial.

    Call it for every variant the script evaluated, discarded and failed ones too.
    *config* is what the variant evaluated, including ``end_date`` (and
    ``start_date``) of its data; it is fingerprinted like a run's metadata.
    The register's gate applies (active card, frozen criteria and cutoff, end on
    or before the cutoff, duplicate configuration needs *reason*).

    Raises:
        GateDenied: the register refused the trial; nothing was written.
    """
```

Implementation outline: `research_dir = research_dir or get_research_dir(create=False)`;
refuse with `GateDenied([Denial(DenialCode.INVALID_WINDOW, "a study trial must state the end_date of the data it evaluated")])`
when `config.get("end_date")` is not a non-empty string; canonical symbols
`tuple(sorted({s.upper() for s in symbols}))`; under `register_lock`: `load_card` (only when
`CARD_ID_RE` matches), `read_trials`, `check(card, RunRequest(research_id, config, symbols, note, reason), trials)`;
denials → `GateDenied`; also deny `DUPLICATE_FINGERPRINT` against live pending markers with
the same fingerprint (as `ResearchRun._run_denials` does) unless a reason is given. Then build
the `Trial` (kind `ITERATION`, status `FAILED` if `failed` else `OK`, `ts` now, fingerprint,
`FINGERPRINT_VERSION`, card's `criteria_hash` and `holdout_cutoff`,
`code=get_git_info(source_dir())` from `fwbg.results.storage` / `fwbg.core.git_env`,
`window=run_window(decision.effective_metadata)`, note, reason, metrics, run_id), `append_trial`,
`store_config(research_dir, fingerprint, canonical_config(effective_metadata, symbols))`,
`commit_register(..., f"trial {id}: {card} study")` — a commit failure only warns (stderr), as
`ResearchRun.record` does. Unreadable register (`OSError`, `CardError`, `LedgerError`,
`ResearchRepoError` before the append) → `GateDenied` with `REGISTER_UNREADABLE`.

**Tests:** recorded trial (kind iteration, metrics, window, config file stored, committed);
missing `end_date` → refused, ledger unchanged; end after the cutoff → `AFTER_CUTOFF`;
parked card → `NOT_ACTIVE`; same config twice → `DUPLICATE_FINGERPRINT`, with `reason`
allowed; `failed=True` → status failed (not counted); missing note → `MISSING_NOTE`.

**Rule line** (both `CLAUDE.md` and `AGENTS.md`, after the `--research-id` bullet):

```markdown
- Scripts that evaluate strategies outside `python -m fwbg.cli` record every
  evaluated variant (failed and discarded ones too) with
  `fwbg.research.study.register_trial(...)`, stating the config they evaluated
  including its `end_date`.
```

**Commit** `feat(research): add register_trial for study scripts`

---

### Task 6: `fwbg research reconcile`

**Files:**
- Create: `src/fwbg/research/reconcile.py`
- Modify: `src/fwbg/research/cli.py` (subcommand)
- Test: `tests/research/test_reconcile.py` (new)

```python
@dataclass(frozen=True)
class ReconcileResult:
    recovered: tuple[str, ...]   # trial ids recorded from stale markers
    committed: bool              # a recovery commit was made
    unindexed: tuple[str, ...]   # run ids in the workspace that are neither in the ledger nor tagged


def reconcile(research_dir: Path, results_dir: Path) -> ReconcileResult:
    """Validate the register, record stale runs, commit what is uncommitted, report gaps.

    Idempotent. Raises (and changes nothing) when the ledger or a card does not
    parse; a failed commit raises ResearchRepoError with the data left in place.
    """
```

Steps: `read_trials` (raises `LedgerError`); load every card file (`load_card` per id from
`strategies/`; a `CardError` raises). Under `register_lock`: `recover_stale`, then
`commit_register(research_dir, (*REGISTER_PATHS, "reports"), "chore: reconcile register")`.
Outside the lock: `ungated_runs(scan_runs(results_dir), research_dir)` (reuse what
`cli._scan_runs` does, including skipping runs in progress) → `unindexed` run ids.

CLI `research reconcile [--results DIR]`: prints recovered trials, `committed` or
`nothing to commit`, and the unindexed runs with the hint
`map them with: python -m fwbg.cli research backfill scan --out mapping.yaml`. Exit 0; errors
→ 1 through `_OPERATOR_ERRORS`.

**Tests:** clean register → nothing recovered, no commit, second call identical; an
uncommitted appended trial (append via `append_trial` without commit) → committed, then
nothing; a stale marker (dead pid) → recovered and committed; an untagged run directory not
in the ledger → reported; a run already in the ledger → not reported; a broken ledger line →
raises, no commit.

**Commit** `feat(research): add reconcile`

---

### Task 7: Docker, compose and docs

**Files:**
- Modify: `Dockerfile` (`apt-get install -y libgomp1 git`)
- Modify: `docker-compose.yml` (api service)
- Modify: `docs/plans/2026-10-19-research-register-design.md` (status line, step 4 notes,
  preview decision, reconcile wording, `fwbg.research.study.register_trial`)

`docker-compose.yml`, api service:

```yaml
    hostname: fwbg-api
    volumes:
      # ...existing...
      # Research register: the host's private git repo, so CLI and API record
      # into one register. Must exist (python -m fwbg.cli research init); never created here.
      - type: bind
        source: ${FWBG_RESEARCH_HOST_DIR:-${HOME}/fwbg/research}
        target: /root/fwbg/research
        bind:
          create_host_path: false
    environment:
      # ...existing...
      # Committer of register commits made by API-started runs (no signing key in the image).
      - GIT_AUTHOR_NAME=fwbg-api
      - GIT_AUTHOR_EMAIL=fwbg-api@localhost
      - GIT_COMMITTER_NAME=fwbg-api
      - GIT_COMMITTER_EMAIL=fwbg-api@localhost
```

Add a comment at `hostname`: pending markers carry the host name; a stable one lets a restarted
container record its own interrupted runs.

Verify: `docker compose -f docker-compose.yml config --quiet` exits 0 and
`docker compose config | grep -A4 research` shows the resolved host path. (Do not start the
stack.)

Design doc: status → steps 1-4 implemented; step 4 notes: preflight through `--check-json`,
container ownership/safe.directory/identity, preview is no trial, `reconcile` reports instead of
indexing, `register_trial` path, dashboard runs stay refused until the dashboard sends
`research_id` (step 7).

**Commit** `chore(docker): mount the research register into the API container` and
`docs: record step 4 of the research register`

---

## After all tasks

- Final whole-branch review, full suite
  (`PYTHONPATH=src ../../.venv/bin/python -m pytest -p no:cacheprovider -q`; 6 known env
  failures: `test_broker_ig_contract` import path, two premium `plugin_contracts`, three premium
  `multi_timeframe`).
- Smoke on a scratch copy of the register (remote removed): API `POST /api/research` creates a
  card; `POST /api/runs/start` without research id → 422 before spawning; with id and note →
  runs and is recorded; `research reconcile` reports nothing left.
