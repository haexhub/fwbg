# Research Register — Step 2: Backfill Implementation Plan

> Execute task by task. Every task is test-first and ends with a commit.
> Design: [2026-10-19-research-register-design.md](2026-10-19-research-register-design.md).
> Step 1: [2026-10-19-research-register-core.md](2026-10-19-research-register-core.md)
> (its "Notes for later steps" apply here too).

**Goal:** Import every existing backtest run and every documented study into
the research register, so the trial count N, duplicate detection and the
per-strategy holdout start from the full history instead of from zero.

**Architecture:** Four new modules in `fwbg.research` plus small additions to
the step-1 modules. `legacy.py` reads run directories into `LegacyRun`
records (pure apart from reading files). `mapping.py` drafts a mapping of
runs to cards and loads/validates the operator-reviewed mapping.
`backfill.py` applies it: it validates everything first, then writes cards
once, appends legacy trials, stores configurations content-addressed and
commits everything in one research-repo commit under the register lock.
`cli.py` exposes `fwbg research init` and `fwbg research backfill
scan|apply`. Re-applying a mapping only adds what is new. (The split of
mapping parsing out of `backfill.py` happened after review; the code blocks
below show the first iteration.)

**Tech stack:** Python ≥ 3.11, stdlib, PyYAML, `fwbg_sdk.enums.Timeframe`,
pytest. No new dependencies.

**Out of scope:** the run-path gate and `new/list/show/amend/verdict` commands
(step 3), the API (step 4), holdout evaluation and DSR (step 5).

---

## Decisions settled while planning

1. **Forward holdout for legacy cards** (operator decision, 2026-10-19). A
   legacy card's cutoff is the day after its last run (or study) date and
   `holdout_contaminated` is `false`: no run has seen data after that day,
   so everything after it is a clean holdout that grows with new data. This
   replaces the design's earlier rule that strategies whose runs reach the
   present are contaminated. New iterations on such a card stay before the
   cutoff.
2. **What counts as a legacy trial** (corrected after review). Each symbol
   has its own result file — `grid_details/<SYMBOL>.json` (runs before
   2026-03-02) or `grid_details/<SYMBOL>/config.json` — whose `status`
   decides: `ok`, `no_edge`, `not_significant` and `no_unified_trades`
   count as evaluated (the last one conservatively, for N); `error` and
   `no_successful_folds` do not; any other status stops the scan. A run is
   `ok` when at least one symbol was evaluated, otherwise `failed` (not
   counted). Metrics come from `unified_metrics.json` where it exists, else
   from the result file's `metrics` / `holdout_metrics` /
   `selected_metrics`; `p_value` and `fold_stability` from the per-symbol
   files, with the `results.json` elite list only as a fallback. Runs named
   `m<N>_smoke…` (fwbg-agents smoke tests) are `smoke` (not counted). The
   first version keyed on `unified_metrics.json` alone and would have
   recorded 20 completed early runs as failed.
3. **Legacy fingerprints use the step-1 definition.** Metadata from
   `strategy.json`, timeframe canonicalised (falling back to the run's
   `config.json` timeframe), presets resolved, symbols from `grid_details/`.
   Presets that no longer exist at the recorded `_strategy_dir` resolve
   against the workspace strategies directory — an approximation (today's
   preset content). A configuration that cannot be rebuilt gets a unique
   stand-in digest: it counts as a trial and never matches a real config.
4. **Open-ended legacy runs are not duplicates of clamped iterations.** A
   legacy run without `end_date` saw data up to its run date; a new
   iteration is clamped to the cutoff, so its window — and fingerprint —
   differ. Duplicate protection applies to the same configuration over the
   same window.
5. **Legacy cards are `parked` or `rejected` and frozen once written.**
   Re-applying keeps existing cards and refuses a mapping whose cutoff,
   criteria, title, slug, status, tags or body for an existing card differ —
   after the first apply the card files, not the mapping, hold these. A
   mapping that regroups runs or studies already in the ledger is refused;
   changing the grouping after an apply means resetting the (unpushed)
   register. `rejected` only where a conclusion is documented; everything
   else is `parked`. A per-card `criteria` override is merged into the
   template (`{**template, **override}`), it does not replace it.
6. **Studies outside the run system** become study trials:
   `run_id = "study:<card>:<variant>"`, fingerprint
   `digest({"study", "variant", "symbols"})`, counted like legacy runs.
7. **The mapping file is the reviewed artefact.** `scan` drafts one card per
   strategy name; the operator regroups it. It is kept in the register's
   `reports/` for provenance.
8. **Real-data dry run (2026-10-20, merged code, scratch register):**
   102 runs → 89 counted, 10 failed, 3 smoke, 0 unresolvable (20 with presets
   resolved from the workspace), 93 distinct configurations, 164 counted
   symbol evaluations (111 with a p-value); the draft has 41 cards; the first
   apply writes 41 cards and 102 trials, a second apply adds nothing. The
   code blocks below are the first iteration (a prototype); each task was
   amended after code review — see the notes at each task. The source of
   truth is the code on the branch.

## Before you start

- Work in the worktree `.worktrees/research-backfill` on branch
  `feat/research-backfill` (based on `origin/develop`); run every command
  from there.
- `PY=../../.venv/bin/python`; prefix every Python command with
  `PYTHONPATH=src` and pass `-p no:cacheprovider` to pytest.
- Tests use `tmp_path` and the `git_env` fixture; nothing touches `~/fwbg`.
- Diffs below are against `origin/develop`; apply them by hand or with
  `git apply` from a file.

---

### Task 1: Card writer

> **Amended after code review (done).** `write_card` creates the file
> exclusively (`open("x")`) and removes a partial file after a failed write;
> `SLUG_RE` caps slugs at 60 characters; `render_card` refuses `\r` in the
> body and does not wrap long lines; tests pin the round-trip guard, the
> slug boundaries and the exact card layout.

**Files:** Modify `src/fwbg/research/card.py`, `tests/research/test_card.py`.

`render_card` serialises a card and refuses output that `parse_card` would
not read back unchanged; `slugify` turns a title into a file-name slug;
`write_card` writes `strategies/<id>-<slug>.md` and never overwrites.

**Step 1: Write the failing tests** (append to `tests/research/test_card.py`):

```diff
diff --git a/tests/research/test_card.py b/tests/research/test_card.py
index 31dc429..7fe6341 100644
--- a/tests/research/test_card.py
+++ b/tests/research/test_card.py
@@ -267,3 +267,60 @@ def test_load_card_refuses_non_utf8_file(tmp_path):
 def test_load_card_refuses_malformed_id(tmp_path):
     with pytest.raises(CardError, match="invalid research id"):
         load_card(tmp_path, "../S-0007")
+
+
+def test_render_round_trips_through_parse():
+    from fwbg.research.card import render_card
+
+    card = parse_card(CARD)
+    assert parse_card(render_card(card)) == card
+
+
+def test_render_round_trips_awkward_titles_and_no_amendments():
+    from fwbg.research.card import render_card
+    from tests.research.factories import make_card
+
+    card = make_card(title='ORB: 15m ä "quoted" #1', tags=(), body="")
+    assert parse_card(render_card(card)) == card
+
+
+def test_write_card_writes_once_and_never_overwrites(tmp_path):
+    from fwbg.research.card import write_card
+    from tests.research.factories import make_card
+
+    card = make_card()
+    (tmp_path / "research" / "strategies").mkdir(parents=True)
+    path = write_card(tmp_path / "research", card, "cusum-continuation")
+    assert path.name == "S-0007-cusum-continuation.md"
+    assert load_card(tmp_path / "research", "S-0007") == card
+    with pytest.raises(CardError, match="never overwritten"):
+        write_card(tmp_path / "research", card, "other-slug")
+
+
+def test_write_card_rejects_invalid_slug(tmp_path):
+    from fwbg.research.card import write_card
+    from tests.research.factories import make_card
+
+    with pytest.raises(CardError, match="invalid slug"):
+        write_card(tmp_path, make_card(), "Not A Slug")
+
+
+@pytest.mark.parametrize(
+    ("title", "slug"),
+    [
+        ("FWBG creative CUSUM event continuation", "fwbg-creative-cusum-event-continuation"),
+        ("ORB + PDH/PDL Scalping", "orb-pdh-pdl-scalping"),
+        ("m3_smoke_20260623", "m3-smoke-20260623"),
+    ],
+)
+def test_slugify(title, slug):
+    from fwbg.research.card import slugify
+
+    assert slugify(title) == slug
+
+
+def test_slugify_refuses_titles_without_ascii_words():
+    from fwbg.research.card import slugify
+
+    with pytest.raises(CardError, match="empty slug"):
+        slugify("äöü")
```

**Step 2:** `PYTHONPATH=src $PY -m pytest tests/research/test_card.py -q -p no:cacheprovider`
→ the new tests fail (`ImportError: cannot import name 'render_card'`).

**Step 3: Implement** (`src/fwbg/research/card.py`):

```diff
diff --git a/src/fwbg/research/card.py b/src/fwbg/research/card.py
index 51cd596..5ff8d99 100644
--- a/src/fwbg/research/card.py
+++ b/src/fwbg/research/card.py
@@ -17,6 +17,7 @@ import yaml
 from fwbg.research.fingerprint import DIGEST_RE, digest
 
 CARD_ID_RE = re.compile(r"S-[0-9]{4}")
+SLUG_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
 _FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*\n?(.*)\Z", re.DOTALL)
 _FIELDS = frozenset(
     {
@@ -244,3 +245,55 @@ def _amendment(value: object) -> CriteriaAmendment:
         )
     except KeyError as exc:
         raise CardError(f"criteria_amendments: missing {exc.args[0]}") from exc
+
+
+def render_card(card: ResearchCard) -> str:
+    """Serialise *card* so that :func:`parse_card` reads it back unchanged."""
+    meta: dict[str, Any] = {
+        "id": card.id,
+        "title": card.title,
+        "status": card.status.value,
+        "tags": list(card.tags),
+        "created": card.created,
+        "holdout_cutoff": card.holdout_cutoff,
+        "holdout_contaminated": card.holdout_contaminated,
+        "success_criteria": dict(card.success_criteria),
+        "registered_criteria_hash": card.registered_criteria_hash,
+    }
+    if card.criteria_amendments:
+        meta["criteria_amendments"] = [
+            {"date": item.date, "reason": item.reason, "criteria_hash": item.criteria_hash}
+            for item in card.criteria_amendments
+        ]
+    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, default_flow_style=False)
+    text = f"---\n{front}---\n{card.body}"
+    if parse_card(text) != card:
+        raise CardError(f"{card.id} would not read back unchanged")
+    return text
+
+
+def slugify(title: str) -> str:
+    """File-name slug for a card title: lowercase ASCII words joined by hyphens."""
+    words = re.findall(r"[a-z0-9]+", title.lower())
+    slug = "-".join(words)[:60].strip("-")
+    if not slug:
+        raise CardError(f"title {title!r} gives an empty slug")
+    return slug
+
+
+def write_card(research_dir: Path, card: ResearchCard, slug: str) -> Path:
+    """Write *card* to ``strategies/<id>-<slug>.md``; an existing card is never overwritten.
+
+    The register must be initialised (``strategies/`` exists); filesystem errors
+    (``OSError``) pass through.
+
+    Raises:
+        CardError: the slug is invalid, or a card with this id already exists.
+    """
+    if not SLUG_RE.fullmatch(slug):
+        raise CardError(f"invalid slug {slug!r}")
+    if load_card(research_dir, card.id) is not None:
+        raise CardError(f"{card.id} already exists; cards are never overwritten")
+    path = research_dir / "strategies" / f"{card.id}-{slug}.md"
+    path.write_text(render_card(card), encoding="utf-8")
+    return path
```

**Step 4:** same command → all pass.

**Step 5:** `git commit -m "feat(research): render and write research cards"`

---

### Task 2: Register lock, commits and config store

> **Amended after code review (done).** `commit_register` stages with only
> the user's global excludes disabled (the register's `.gitignore` still
> applies), commits exactly the changed names (works with an empty
> `configs/`/`reports/`, commits renames and deletions, never other staged
> work) and refuses empty `paths`; a git timeout sends SIGTERM first so no
> `index.lock` is left behind; `store_config` writes atomically; tests pin
> that the lock excludes other holders.

**Files:** Modify `src/fwbg/research/repo.py`, `tests/research/test_repo.py`.

`register_lock` serialises register writes with `.register.lock` (separate
from the ledger's own `flock`); `commit_register` commits the changed files
under the named paths (isolated git env) and reports whether anything changed;
`store_config` writes `configs/<hex>.json` after checking that the content
is the fingerprinted configuration.

**Step 1: Write the failing tests** (append to `tests/research/test_repo.py`):

```diff
diff --git a/tests/research/test_repo.py b/tests/research/test_repo.py
index 61a8288..8b15a24 100644
--- a/tests/research/test_repo.py
+++ b/tests/research/test_repo.py
@@ -174,3 +174,53 @@ def test_research_dir_defaults_to_workspace(tmp_path, monkeypatch):
 def test_research_dir_env_override(tmp_path, monkeypatch):
     monkeypatch.setenv("FWBG_RESEARCH_DIR", str(tmp_path / "elsewhere"))
     assert get_research_dir() == tmp_path / "elsewhere"
+
+
+def test_commit_register_commits_only_named_paths(tmp_path, git_env):
+    from fwbg.research.repo import commit_register
+
+    research = tmp_path / "research"
+    init_research_repo(research)
+    (research / "strategies" / "S-0001-x.md").write_text("card\n", encoding="utf-8")
+    (research / "notes.txt").write_text("private scratch\n", encoding="utf-8")
+    assert commit_register(research, ["strategies"], "chore: add card") is True
+    files = subprocess.run(
+        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
+        cwd=research,
+        capture_output=True,
+        text=True,
+        check=True,
+    ).stdout.split()
+    assert "strategies/S-0001-x.md" in files
+    assert "notes.txt" not in files
+    assert commit_register(research, ["strategies"], "chore: nothing") is False
+
+
+def test_register_lock_creates_an_ignored_lock_file(tmp_path, git_env):
+    from fwbg.research.repo import REGISTER_LOCK, register_lock
+
+    research = tmp_path / "research"
+    init_research_repo(research)
+    with register_lock(research):
+        assert (research / REGISTER_LOCK).exists()
+    status = subprocess.run(
+        ["git", "status", "--porcelain"], cwd=research, capture_output=True, text=True, check=True
+    ).stdout
+    assert REGISTER_LOCK not in status
+
+
+def test_store_config_is_content_addressed_and_verified(tmp_path):
+    from fwbg.research.fingerprint import canonical_config, digest
+    from fwbg.research.repo import store_config
+
+    (tmp_path / "configs").mkdir()
+    config = canonical_config({"timeframe": "DAY_1"}, ("GOLD",))
+    fingerprint = digest(config)
+    path = store_config(tmp_path, fingerprint, config)
+    assert path.name == fingerprint.removeprefix("sha256:") + ".json"
+    assert store_config(tmp_path, fingerprint, config) == path
+    with pytest.raises(ValueError, match="not the fingerprint"):
+        store_config(tmp_path, fingerprint, canonical_config({"timeframe": "HOUR_1"}, ("GOLD",)))
+    path.write_text("tampered\n", encoding="utf-8")
+    with pytest.raises(ResearchRepoError, match="different content"):
+        store_config(tmp_path, fingerprint, config)
```

**Step 2:** `PYTHONPATH=src $PY -m pytest tests/research/test_repo.py -q -p no:cacheprovider` → new tests fail.

**Step 3: Implement** (`src/fwbg/research/repo.py`):

```diff
diff --git a/src/fwbg/research/repo.py b/src/fwbg/research/repo.py
index ccabf7d..3223b0b 100644
--- a/src/fwbg/research/repo.py
+++ b/src/fwbg/research/repo.py
@@ -2,16 +2,22 @@
 
 from __future__ import annotations
 
+import fcntl
 import subprocess
+from collections.abc import Iterator, Mapping, Sequence
+from contextlib import contextmanager
 from pathlib import Path
+from typing import Any
 
 from fwbg.core.git_env import isolated_git_env
+from fwbg.research.fingerprint import DIGEST_RE, canonical_json, digest
 from fwbg.research.ledger import LEDGER_FILENAME
 
 CRITERIA_TEMPLATE = Path("templates") / "criteria-default.yaml"
 _DIRECTORIES = ("strategies", "configs", "templates", "reports")
 _GITIGNORE = ".gitignore"
-_GITIGNORE_TEXT = ".staging/\n.register.lock\n"
+REGISTER_LOCK = ".register.lock"
+_GITIGNORE_TEXT = f".staging/\n{REGISTER_LOCK}\n"
 
 DEFAULT_CRITERIA_YAML = """\
 # Starting values for the success criteria of new research cards.
@@ -74,6 +80,48 @@ def init_research_repo(research_dir: Path) -> bool:
     return True
 
 
+@contextmanager
+def register_lock(research_dir: Path) -> Iterator[None]:
+    """Serialise register writes (ledger appends plus their commit) across processes.
+
+    Separate from the ``flock`` that ``append_trial`` takes on the ledger itself:
+    locking ``trials.jsonl`` again from the same process would deadlock.
+    """
+    with (research_dir / REGISTER_LOCK).open("a", encoding="utf-8") as handle:
+        fcntl.flock(handle, fcntl.LOCK_EX)
+        yield
+
+
+def commit_register(research_dir: Path, paths: Sequence[str], message: str) -> bool:
+    """Commit exactly *paths* (relative to the register); ``False`` if nothing changed."""
+    # --force: global excludes must not hide register files; only the named paths are added.
+    _git(research_dir, "add", "--force", "--", *paths)
+    staged = _git(research_dir, "diff", "--cached", "--quiet", "--", *paths, ok_codes=(0, 1))
+    if staged.returncode == 0:
+        return False
+    _git(research_dir, "commit", "--quiet", "--message", message, "--", *paths)
+    return True
+
+
+def store_config(research_dir: Path, fingerprint: str, config: Mapping[str, Any]) -> Path:
+    """Store the fingerprinted *config* as ``configs/<hex>.json`` (content-addressed).
+
+    Raises:
+        ValueError: *fingerprint* is malformed or is not the digest of *config*.
+        ResearchRepoError: a different file already exists under that name.
+    """
+    if not DIGEST_RE.fullmatch(fingerprint) or digest(config) != fingerprint:
+        raise ValueError(f"{fingerprint!r} is not the fingerprint of the given config")
+    path = research_dir / "configs" / f"{fingerprint.removeprefix('sha256:')}.json"
+    text = canonical_json(config) + "\n"
+    if path.exists():
+        if path.read_text(encoding="utf-8") != text:
+            raise ResearchRepoError(f"{path.name} exists with different content")
+        return path
+    path.write_text(text, encoding="utf-8")
+    return path
+
+
 def _write_if_missing(path: Path, text: str) -> None:
     if not path.exists():
         path.write_text(text, encoding="utf-8")
```

**Step 4:** same command → all pass.

**Step 5:** `git commit -m "feat(research): add register lock, commits and config store"`

---

### Task 3: Legacy run scanner

> **Amended after code review (done).** The scanner reads all three run
> formats and decides per symbol from its status field (decision 2); every
> error names the run; symlinks are skipped; `LegacyRun` records
> `presets_from_workspace`.

**Files:** Create `src/fwbg/research/legacy.py`, `tests/research/test_legacy.py`;
modify `tests/research/factories.py`.

**Step 1: Test factory** (`tests/research/factories.py`):

```diff
diff --git a/tests/research/factories.py b/tests/research/factories.py
index f37f835..3b7bc49 100644
--- a/tests/research/factories.py
+++ b/tests/research/factories.py
@@ -2,6 +2,8 @@
 
 from __future__ import annotations
 
+import json
+from collections.abc import Mapping, Sequence
 from datetime import date
 from pathlib import Path
 from typing import Any
@@ -51,3 +53,41 @@ def append_many(ledger: str, worker: int, count: int) -> None:
     """Process target for concurrency tests: append *count* trials."""
     for index in range(count):
         append_trial(Path(ledger), make_trial(run_id=f"w{worker}-{index}"))
+
+
+def make_run(
+    results_dir: Path,
+    run_id: str,
+    *,
+    name: str = "orb-test",
+    timestamp: str = "2026-03-05T10:00:00",
+    timeframe: str | None = "MINUTE_15",
+    symbols: Mapping[str, Mapping[str, Any] | None] | None = None,
+    strategy_extra: Mapping[str, Any] | None = None,
+    elite: Sequence[Mapping[str, Any]] = (),
+) -> Path:
+    """Write a minimal legacy run directory; ``symbols`` maps symbol -> unified metrics."""
+    run_dir = results_dir / run_id
+    run_dir.mkdir(parents=True)
+    strategy: dict[str, Any] = {"name": name, "pipeline": {"indicators": []}, "model": {}}
+    if timeframe is not None:
+        strategy["timeframe"] = timeframe
+    strategy.update(strategy_extra or {})
+    (run_dir / "strategy.json").write_text(json.dumps(strategy), encoding="utf-8")
+    config = {
+        "run_id": run_id,
+        "timestamp": timestamp,
+        "timeframe": timeframe or "HOUR_1",
+        "git": {"commit": "abc123", "branch": "main", "dirty": True, "last_commit_message": "x"},
+    }
+    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
+    for symbol, metrics in (symbols if symbols is not None else {"GOLD": {"sharpe": 0.5}}).items():
+        folder = run_dir / "grid_details" / symbol
+        folder.mkdir(parents=True)
+        if metrics is not None:
+            (folder / "unified_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
+    if elite:
+        (run_dir / "results.json").write_text(
+            json.dumps({"elite_results": list(elite)}), encoding="utf-8"
+        )
+    return run_dir
```

**Step 2: Write the failing tests** (`tests/research/test_legacy.py`):

```python
import json

import pytest

from fwbg.research.ledger import TrialKind, TrialStatus
from fwbg.research.legacy import LegacyRunError, load_run, scan_runs
from tests.research.factories import make_run


@pytest.fixture
def preset_base(tmp_path):
    base = tmp_path / "workspace-strategies"
    (base / "configs").mkdir(parents=True)
    (base / "pipelines").mkdir()
    (base / "pipelines" / "orb_simple_v1.json").write_text(
        json.dumps({"indicators": [{"name": "opening_range"}]}), encoding="utf-8"
    )
    return base


def test_reads_metrics_status_code_and_window(tmp_path, preset_base):
    run_dir = make_run(
        tmp_path,
        "run-1",
        symbols={"GOLD": {"sharpe": 0.14, "trades": 37, "max_drawdown": 0.19, "noise": 1}},
        elite=[{"symbol": "GOLD", "monte_carlo": {"p_value": 0.032}, "fold_stability": 0.67}],
        strategy_extra={"start_date": "2020-01-01", "end_date": "2025-01-01"},
    )
    run = load_run(run_dir, preset_base)
    assert run.status is TrialStatus.OK
    assert run.kind is TrialKind.LEGACY
    assert run.symbols == ("GOLD",)
    assert run.metrics["GOLD"]["sharpe"] == 0.14
    assert run.metrics["GOLD"]["p_value"] == 0.032
    assert run.metrics["GOLD"]["fold_stability"] == 0.67
    assert "noise" not in run.metrics["GOLD"]
    assert run.window == {"start": "2020-01-01", "end": "2025-01-01"}
    assert run.code == {"commit": "abc123", "branch": "main", "dirty": True}
    assert run.unresolved is None and run.config is not None


def test_every_evaluated_symbol_counts_not_only_the_elite(tmp_path, preset_base):
    run = load_run(
        make_run(tmp_path, "run-1", symbols={"GOLD": {"sharpe": 0.5}, "EURUSD": {"sharpe": -0.2}}),
        preset_base,
    )
    assert set(run.metrics) == {"GOLD", "EURUSD"}
    assert run.metrics["EURUSD"]["p_value"] is None


def test_run_without_symbol_metrics_is_failed(tmp_path, preset_base):
    run = load_run(make_run(tmp_path, "run-1", symbols={"GOLD": None}), preset_base)
    assert run.status is TrialStatus.FAILED
    assert run.metrics == {}
    assert run.symbols == ("GOLD",)


def test_agent_smoke_runs_are_smoke(tmp_path, preset_base):
    run = load_run(make_run(tmp_path, "run-1", name="m3_smoke_20260623_231634"), preset_base)
    assert run.kind is TrialKind.SMOKE


def test_non_finite_metrics_become_null(tmp_path, preset_base):
    run_dir = make_run(tmp_path, "run-1")
    (run_dir / "grid_details" / "GOLD" / "unified_metrics.json").write_text(
        '{"sharpe": NaN, "calmar": Infinity, "trades": true}', encoding="utf-8"
    )
    assert load_run(run_dir, preset_base).metrics["GOLD"] == {
        "sharpe": None,
        "calmar": None,
        "trades": None,
        "p_value": None,
        "fold_stability": None,
    }


def test_timeframe_aliases_share_a_fingerprint(tmp_path, preset_base):
    a = load_run(make_run(tmp_path, "run-a", timeframe="DAY"), preset_base)
    b = load_run(make_run(tmp_path, "run-b", timeframe="DAY_1"), preset_base)
    assert a.fingerprint == b.fingerprint


def test_missing_timeframe_falls_back_to_the_runs_effective_timeframe(tmp_path, preset_base):
    run_dir = make_run(tmp_path, "run-1", timeframe=None)
    assert load_run(run_dir, preset_base).config["strategy"]["timeframe"] == "HOUR_1"


def test_presets_resolve_from_the_workspace_when_the_old_directory_is_gone(tmp_path, preset_base):
    run_dir = make_run(
        tmp_path / "results",
        "run-1",
        strategy_extra={"pipeline": "orb_simple_v1", "_strategy_dir": "/nonexistent/configs"},
    )
    run = load_run(run_dir, preset_base)
    assert run.unresolved is None
    assert run.config["strategy"]["pipeline"] == {"indicators": [{"name": "opening_range"}]}


def test_unresolvable_presets_get_a_unique_stand_in(tmp_path, preset_base):
    extra = {"pipeline": "missing_preset", "_strategy_dir": "/nonexistent/configs"}
    a = load_run(make_run(tmp_path, "run-a", strategy_extra=extra), preset_base)
    b = load_run(make_run(tmp_path, "run-b", strategy_extra=extra), preset_base)
    assert a.config is None
    assert "missing_preset" in a.unresolved
    assert a.fingerprint != b.fingerprint
    assert a.status is TrialStatus.OK


def test_scan_skips_directories_without_strategy(tmp_path, preset_base):
    make_run(tmp_path, "run-1")
    (tmp_path / "portfolio").mkdir()
    assert [run.run_id for run in scan_runs(tmp_path, preset_base)] == ["run-1"]


def test_unreadable_config_is_a_legacy_run_error(tmp_path, preset_base):
    run_dir = make_run(tmp_path, "run-1")
    (run_dir / "config.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(LegacyRunError, match="run-1/config.json"):
        load_run(run_dir, preset_base)
```

**Step 3:** `PYTHONPATH=src $PY -m pytest tests/research/test_legacy.py -q -p no:cacheprovider`
→ collection error (`No module named 'fwbg.research.legacy'`).

**Step 4: Implement** (`src/fwbg/research/legacy.py`):

```python
"""Read legacy run directories (``test_results/<run_id>``) for the backfill."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from fwbg_sdk.enums import Timeframe

from fwbg.core.presets import resolve_presets
from fwbg.research.fingerprint import canonical_config, digest
from fwbg.research.ledger import Metrics, TrialKind, TrialStatus

METRIC_KEYS = (
    "sharpe",
    "trades",
    "max_drawdown",
    "win_rate",
    "annual_return",
    "calmar",
    "profit_factor",
    "rrr",
)
_SMOKE_NAME_RE = re.compile(r"m[0-9]+_smoke")
_GIT_KEYS = ("commit", "branch", "dirty", "patch")


class LegacyRunError(Exception):
    """A run directory cannot be read as a legacy run."""


@dataclass(frozen=True)
class LegacyRun:
    """One stored run, reduced to what a legacy trial records."""

    run_id: str
    ts: str
    run_date: date
    name: str
    kind: TrialKind
    status: TrialStatus
    symbols: tuple[str, ...]
    metrics: Metrics
    config: Mapping[str, Any] | None
    fingerprint: str
    unresolved: str | None
    code: Mapping[str, Any]
    window: Mapping[str, str | None]


def scan_runs(results_dir: Path, preset_base: Path) -> list[LegacyRun]:
    """Read every run directory under *results_dir* that has a ``strategy.json``.

    *preset_base* is the workspace strategies directory; it stands in for a
    run's recorded ``_strategy_dir`` when that location no longer holds the
    presets the run referenced.
    """
    return [
        load_run(run_dir, preset_base)
        for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir())
        if (run_dir / "strategy.json").exists()
    ]


def load_run(run_dir: Path, preset_base: Path) -> LegacyRun:
    """Read one run directory; raise :class:`LegacyRunError` if its files are unusable."""
    strategy = _read_object(run_dir / "strategy.json")
    config = _read_object(run_dir / "config.json")
    ts = config.get("timestamp")
    if not isinstance(ts, str) or len(ts) < 10:
        raise LegacyRunError(f"{run_dir.name}: config.json has no timestamp")
    try:
        run_date = date.fromisoformat(ts[:10])
    except ValueError as exc:
        raise LegacyRunError(f"{run_dir.name}: bad timestamp {ts!r}") from exc
    name = strategy.get("name")
    name = name if isinstance(name, str) and name.strip() else run_dir.name
    symbols = _symbols(run_dir)
    metrics = _metrics(run_dir, symbols)
    canonical, unresolved = _canonical(strategy, config, symbols, preset_base)
    # An unresolvable run still counts as a trial, under a stand-in that never
    # matches a real configuration.
    fingerprint = (
        digest(canonical)
        if canonical is not None
        else digest({"unresolvable_legacy_run": run_dir.name})
    )
    return LegacyRun(
        run_id=run_dir.name,
        ts=ts,
        run_date=run_date,
        name=name,
        kind=TrialKind.SMOKE if _SMOKE_NAME_RE.match(name) else TrialKind.LEGACY,
        status=TrialStatus.OK if metrics else TrialStatus.FAILED,
        symbols=symbols,
        metrics=metrics,
        config=canonical,
        fingerprint=fingerprint,
        unresolved=unresolved,
        code=_code(config),
        window=_window(strategy),
    )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LegacyRunError(f"{path.parent.name}/{path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise LegacyRunError(f"{path.parent.name}/{path.name}: expected a JSON object")
    return data


def _symbols(run_dir: Path) -> tuple[str, ...]:
    details = run_dir / "grid_details"
    if not details.is_dir():
        return ()
    return tuple(sorted(p.name.upper() for p in details.iterdir() if p.is_dir()))


def _metrics(run_dir: Path, symbols: tuple[str, ...]) -> dict[str, dict[str, float | int | None]]:
    """Per-symbol metrics for every evaluated symbol (not only the reported elite)."""
    elite = _elite_by_symbol(run_dir / "results.json")
    metrics: dict[str, dict[str, float | int | None]] = {}
    for symbol in symbols:
        path = run_dir / "grid_details" / symbol / "unified_metrics.json"
        if not path.exists():
            continue
        values = {
            key: _number(raw) for key, raw in _read_object(path).items() if key in METRIC_KEYS
        }
        extra = elite.get(symbol, {})
        values["p_value"] = _number(extra.get("monte_carlo", {}).get("p_value"))
        values["fold_stability"] = _number(extra.get("fold_stability"))
        metrics[symbol] = values
    return metrics


def _elite_by_symbol(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    entries = _read_object(path).get("elite_results") or []
    return {
        str(entry["symbol"]).upper(): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("symbol"), str)
    }


def _number(value: object) -> float | int | None:
    """A finite number for the ledger; anything else (NaN, inf, text, bool) is None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical(
    strategy: Mapping[str, Any],
    config: Mapping[str, Any],
    symbols: tuple[str, ...],
    preset_base: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """The fingerprinted configuration (presets resolved, canonical timeframe).

    Returns ``(None, reason)`` when the run's configuration cannot be rebuilt.
    """
    metadata = dict(strategy)
    try:
        timeframe = metadata.get("timeframe") or config.get("timeframe")
        if timeframe:
            metadata["timeframe"] = Timeframe.from_str(str(timeframe)).canonical
        try:
            resolved = resolve_presets(metadata)
        except FileNotFoundError:
            # Presets moved since the run: resolve against the workspace instead.
            moved = {**metadata, "_strategy_dir": str(preset_base / "configs")}
            resolved = resolve_presets(moved)
        canonical = canonical_config(resolved, symbols)
        digest(canonical)
        return canonical, None
    except (OSError, ValueError, TypeError) as exc:
        return None, f"configuration unresolvable: {exc}".replace("\n", " ")


def _code(config: Mapping[str, Any]) -> dict[str, Any]:
    git = config.get("git")
    if not isinstance(git, dict):
        return {}
    return {key: git[key] for key in _GIT_KEYS if key in git}


def _window(strategy: Mapping[str, Any]) -> dict[str, str | None]:
    return {
        "start": (
            strategy.get("start_date") if isinstance(strategy.get("start_date"), str) else None
        ),
        "end": strategy.get("end_date") if isinstance(strategy.get("end_date"), str) else None,
    }
```

**Step 5:** same command → 11 passed.

**Step 6:** `git commit -m "feat(research): read legacy run directories"`

---

### Task 4: Backfill

> **Amended after code review (done).** `apply_mapping` validates the whole
> mapping (cards, trials, ledger consistency, no future dates) before the
> first write and again under the lock; regrouping after an apply, duplicate
> study variants and edits to existing cards are refused; criteria
> overrides merge with the template. The mapping side moved to
> `fwbg.research.mapping` (`load_mapping` refuses duplicate YAML keys);
> `backfill.py` keeps `apply_mapping`.

**Files:** Create `src/fwbg/research/backfill.py`, `tests/research/test_backfill.py`
(after review also `src/fwbg/research/mapping.py`, `tests/research/test_mapping.py`,
`tests/research/conftest.py`).

**Step 1: Write the failing tests** (`tests/research/test_backfill.py`):

```python
import subprocess
from dataclasses import replace
from datetime import date

import pytest

from fwbg.research.backfill import BackfillError, apply_mapping, draft_mapping, parse_mapping
from fwbg.research.card import Status, load_card
from fwbg.research.gate import DenialCode, RunRequest, check
from fwbg.research.ledger import LEDGER_FILENAME, TrialKind, read_trials
from fwbg.research.legacy import scan_runs
from fwbg.research.repo import init_research_repo
from tests.research.factories import make_run

CRITERIA = {"min_sharpe": 1.0, "min_trades": 100.0}
TODAY = date(2026, 10, 20)


@pytest.fixture
def register(tmp_path, git_env):
    research = tmp_path / "research"
    init_research_repo(research)
    return research


@pytest.fixture
def runs(tmp_path):
    results = tmp_path / "results"
    make_run(results, "r1", name="orb-test", timestamp="2026-03-05T10:00:00")
    make_run(results, "r2", name="orb-test", timestamp="2026-03-06T09:00:00")
    make_run(results, "r3", name="pdh retest", timestamp="2026-02-22T08:00:00")
    return scan_runs(results, tmp_path / "presets")


def mapping_for(runs, **card_overrides):
    data = draft_mapping(runs)
    for card in data["cards"]:
        card.update(card_overrides.get(card["id"], {}))
    return parse_mapping(data)


def git_log(path):
    return subprocess.run(
        ["git", "log", "--format=%s"], cwd=path, capture_output=True, text=True, check=True
    ).stdout.splitlines()


def test_draft_proposes_one_parked_card_per_name_in_order_of_first_run(runs):
    cards = draft_mapping(runs)["cards"]
    assert [(c["id"], c["title"], c["runs"]) for c in cards] == [
        ("S-0001", "pdh retest", ["r3"]),
        ("S-0002", "orb-test", ["r1", "r2"]),
    ]
    assert {c["status"] for c in cards} == {"parked"}
    assert cards[0]["slug"] == "pdh-retest"


def test_apply_writes_cards_trials_configs_and_one_commit(runs, register):
    result = apply_mapping(mapping_for(runs), runs, register, CRITERIA, TODAY)
    assert (result.cards_written, result.trials_appended, result.committed) == (2, 3, True)
    card = load_card(register, "S-0002")
    assert card.status is Status.PARKED
    assert card.holdout_cutoff == date(2026, 3, 7)
    assert card.holdout_contaminated is False
    assert card.created == TODAY
    trials = read_trials(register / LEDGER_FILENAME)
    assert [t.run_id for t in trials] == ["r3", "r1", "r2"]
    assert {t.kind for t in trials} == {TrialKind.LEGACY}
    assert all(t.holdout_cutoff == load_card(register, t.strategy).holdout_cutoff for t in trials)
    assert all(t.criteria_hash == card.criteria_hash for t in trials)
    # Names are not part of the fingerprint: all three runs share one stored config.
    assert len(list((register / "configs").glob("*.json"))) == 1
    assert git_log(register)[0].startswith("chore: backfill legacy research (2 cards, 3 trials)")


def test_applying_again_adds_nothing(runs, register):
    plans = mapping_for(runs)
    apply_mapping(plans, runs, register, CRITERIA, TODAY)
    result = apply_mapping(plans, runs, register, CRITERIA, date(2026, 10, 21))
    assert (result.cards_written, result.trials_appended, result.committed) == (0, 0, False)
    assert len(read_trials(register / LEDGER_FILENAME)) == 3


def test_new_runs_are_added_to_existing_cards_only_if_the_cutoff_holds(tmp_path, runs, register):
    apply_mapping(mapping_for(runs), runs, register, CRITERIA, TODAY)
    make_run(tmp_path / "results", "r4", name="orb-test", timestamp="2026-04-01T10:00:00")
    more = scan_runs(tmp_path / "results", tmp_path / "presets")
    with pytest.raises(BackfillError, match="S-0002 exists with a different cutoff"):
        apply_mapping(mapping_for(more), more, register, CRITERIA, TODAY)


def test_every_scanned_run_must_be_mapped_exactly_once(runs, register):
    data = draft_mapping(runs)
    data["cards"][1]["runs"] = ["r1", "r1", "ghost"]
    with pytest.raises(BackfillError) as excinfo:
        apply_mapping(parse_mapping(data), runs, register, CRITERIA, TODAY)
    message = str(excinfo.value)
    assert "unknown runs: ghost" in message
    assert "mapped more than once: r1" in message
    assert "unmapped runs: r2" in message
    assert read_trials(register / LEDGER_FILENAME) == []


def test_criteria_override_and_rejected_status(runs, register):
    plans = mapping_for(
        runs, **{"S-0001": {"status": "rejected", "criteria": {"max_drawdown": 0.6}}}
    )
    apply_mapping(plans, runs, register, CRITERIA, TODAY)
    card = load_card(register, "S-0001")
    assert card.status is Status.REJECTED
    assert dict(card.success_criteria) == {"max_drawdown": 0.6}


def test_study_trials_are_recorded_once_with_a_synthetic_fingerprint(runs, register):
    study = {
        "variant": "config-4",
        "symbols": ["eurusd"],
        "date": date(2026, 9, 25),
        "note": "narrative state, alternated key order",
        "metrics": {"eurusd": {"accuracy": 0.29}},
    }
    plans = mapping_for(runs, **{"S-0001": {"studies": [study]}})
    apply_mapping(plans, runs, register, CRITERIA, TODAY)
    apply_mapping(plans, runs, register, CRITERIA, TODAY)
    trials = [t for t in read_trials(register / LEDGER_FILENAME) if t.run_id.startswith("study:")]
    assert [t.run_id for t in trials] == ["study:S-0001:config-4"]
    assert trials[0].metrics == {"EURUSD": {"accuracy": 0.29}}
    assert load_card(register, "S-0001").holdout_cutoff == date(2026, 9, 26)


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"id": "X-1"}, "invalid id"),
        ({"status": "active"}, "parked or rejected"),
        ({"slug": "Bad Slug"}, "invalid slug"),
        ({"runs": [], "studies": []}, "at least one run"),
        ({"criteria": {"min_sharpe": "high"}}, "must be a number"),
        ({"extra": 1}, "unknown keys: extra"),
        ({"studies": [{"variant": "v", "symbols": ["A"], "date": "soon", "note": "n"}]}, "date"),
        (
            {
                "studies": [
                    {"variant": "v", "symbols": ["A"], "date": date(2026, 1, 1), "note": "a\nb"}
                ]
            },
            "one line",
        ),
        (
            {
                "studies": [
                    {
                        "variant": "v",
                        "symbols": ["A"],
                        "date": date(2026, 1, 1),
                        "note": "n",
                        "metrics": {"A": {"x": float("nan")}},
                    }
                ]
            },
            "finite",
        ),
    ],
    ids=["id", "status", "slug", "empty", "criteria", "unknown", "date", "note", "nan"],
)
def test_parse_mapping_rejects_invalid_cards(runs, change, fragment):
    data = draft_mapping(runs)
    data["cards"][0].update(change)
    with pytest.raises(BackfillError, match=fragment):
        parse_mapping(data)


def test_duplicate_card_ids_are_rejected(runs):
    data = draft_mapping(runs)
    data["cards"][1]["id"] = "S-0001"
    with pytest.raises(BackfillError, match="duplicate card ids: S-0001"):
        parse_mapping(data)


def test_gate_recognises_a_backfilled_config_with_the_same_window(tmp_path, git_env):
    results = tmp_path / "results"
    make_run(
        results, "r1", timestamp="2026-03-05T10:00:00", strategy_extra={"end_date": "2026-03-01"}
    )
    runs = scan_runs(results, tmp_path / "presets")
    research = tmp_path / "research"
    init_research_repo(research)
    apply_mapping(parse_mapping(draft_mapping(runs)), runs, research, CRITERIA, TODAY)
    card = replace(load_card(research, "S-0001"), status=Status.ACTIVE)
    request = RunRequest(
        research_id="S-0001",
        strategy_metadata=dict(runs[0].config["strategy"]),
        symbols=runs[0].symbols,
        note="same config, same window",
    )
    decision = check(card, request, read_trials(research / LEDGER_FILENAME))
    assert {d.code for d in decision.denials} == {DenialCode.DUPLICATE_FINGERPRINT}


def test_open_ended_legacy_runs_differ_from_clamped_iterations(runs, register):
    # A legacy run without end_date saw data up to its run date; a new iteration is
    # clamped to the cutoff, so it evaluates a different window and is not a duplicate.
    apply_mapping(mapping_for(runs), runs, register, CRITERIA, TODAY)
    run = next(r for r in runs if r.run_id == "r1")
    card = replace(load_card(register, "S-0002"), status=Status.ACTIVE)
    request = RunRequest(
        research_id="S-0002",
        strategy_metadata=dict(run.config["strategy"]),
        symbols=run.symbols,
        note="clamped re-run",
    )
    assert check(card, request, read_trials(register / LEDGER_FILENAME)).allowed
```

**Step 2:** `PYTHONPATH=src $PY -m pytest tests/research/test_backfill.py -q -p no:cacheprovider`
→ collection error (`No module named 'fwbg.research.backfill'`).

**Step 3: Implement** (`src/fwbg/research/backfill.py`):

```python
"""Backfill legacy runs and documented studies into the research register.

``draft_mapping`` proposes one card per strategy name; the operator regroups
the cards, adds verdict prose and study trials, and ``apply_mapping`` writes
cards, legacy trials and stored configs. Applying again adds only what is new.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fwbg.research.card import (
    CARD_ID_RE,
    SLUG_RE,
    CardError,
    ResearchCard,
    Status,
    criteria_hash,
    load_card,
    slugify,
    write_card,
)
from fwbg.research.fingerprint import FINGERPRINT_VERSION, digest
from fwbg.research.legacy import LegacyRun
from fwbg.research.ledger import (
    LEDGER_FILENAME,
    Metrics,
    Trial,
    TrialKind,
    TrialStatus,
    append_trial,
    read_trials,
)
from fwbg.research.repo import commit_register, register_lock, store_config

_LEGACY_STATUSES = (Status.PARKED, Status.REJECTED)
_CARD_KEYS = frozenset(
    {"id", "title", "slug", "status", "tags", "criteria", "body", "runs", "studies"}
)
_STUDY_KEYS = frozenset({"variant", "symbols", "date", "note", "metrics"})


class BackfillError(Exception):
    """The mapping is invalid or does not match the scanned runs or the register."""


@dataclass(frozen=True)
class StudyTrial:
    """One documented evaluation of a study that ran outside the run system."""

    variant: str
    symbols: tuple[str, ...]
    date: date
    note: str
    metrics: Metrics


@dataclass(frozen=True)
class CardPlan:
    """One card of the mapping, validated."""

    id: str
    title: str
    slug: str
    status: Status
    tags: tuple[str, ...]
    criteria: Mapping[str, float] | None
    body: str
    runs: tuple[str, ...]
    studies: tuple[StudyTrial, ...]


@dataclass(frozen=True)
class BackfillResult:
    cards_written: int
    trials_appended: int
    committed: bool


def draft_mapping(runs: Sequence[LegacyRun]) -> dict[str, Any]:
    """Propose one parked card per strategy name, in order of first run."""
    by_name: dict[str, list[LegacyRun]] = {}
    for run in sorted(runs, key=lambda r: (r.run_date, r.run_id)):
        by_name.setdefault(run.name, []).append(run)
    cards = []
    for number, (name, group) in enumerate(by_name.items(), start=1):
        cards.append(
            {
                "id": f"S-{number:04d}",
                "title": name,
                "slug": slugify(name),
                "status": Status.PARKED.value,
                "tags": [],
                "body": "## Hypothesis\n\n## Verdict\n\nNo conclusion documented.\n",
                "runs": [run.run_id for run in group],
            }
        )
    return {"cards": cards}


def parse_mapping(data: object) -> list[CardPlan]:
    """Validate a mapping document; raise :class:`BackfillError` on any defect."""
    if not isinstance(data, Mapping) or not isinstance(data.get("cards"), list):
        raise BackfillError("mapping must be a mapping with a 'cards' list")
    plans = [_card_plan(item, index) for index, item in enumerate(data["cards"], start=1)]
    ids = [plan.id for plan in plans]
    duplicates = sorted({card_id for card_id in ids if ids.count(card_id) > 1})
    if duplicates:
        raise BackfillError(f"duplicate card ids: {', '.join(duplicates)}")
    return plans


def apply_mapping(
    plans: Sequence[CardPlan],
    runs: Sequence[LegacyRun],
    research_dir: Path,
    default_criteria: Mapping[str, float],
    today: date,
) -> BackfillResult:
    """Write the mapped cards and their legacy trials into the register.

    Every scanned run must be mapped exactly once. Cards that already exist
    are kept and must agree with the mapping; runs and study variants already
    in the ledger are skipped, so applying again only adds what is new.
    """
    template = _criteria(default_criteria, "criteria template")
    if template is None:
        raise BackfillError("criteria template is empty")
    runs_by_id = {run.run_id: run for run in runs}
    _check_coverage(plans, runs_by_id)
    ledger = research_dir / LEDGER_FILENAME
    cards_written = 0
    trials_appended = 0
    with register_lock(research_dir):
        recorded = {trial.run_id for trial in read_trials(ledger)}
        for plan in plans:
            card, written = _ensure_card(plan, runs_by_id, research_dir, template, today)
            cards_written += written
            for run_id in plan.runs:
                if run_id in recorded:
                    continue
                run = runs_by_id[run_id]
                if run.config is not None:
                    store_config(research_dir, run.fingerprint, run.config)
                append_trial(ledger, _legacy_trial(card, run))
                trials_appended += 1
            for study in plan.studies:
                study_run_id = f"study:{card.id}:{study.variant}"
                if study_run_id in recorded:
                    continue
                append_trial(ledger, _study_trial(card, study, study_run_id))
                trials_appended += 1
        committed = commit_register(
            research_dir,
            ["strategies", "configs", LEDGER_FILENAME],
            f"chore: backfill legacy research ({cards_written} cards, {trials_appended} trials)",
        )
    return BackfillResult(cards_written, trials_appended, committed)


def _card_plan(item: object, index: int) -> CardPlan:
    where = f"cards[{index}]"
    if not isinstance(item, Mapping):
        raise BackfillError(f"{where}: expected a mapping")
    unknown = sorted(str(key) for key in item.keys() - _CARD_KEYS)
    if unknown:
        raise BackfillError(f"{where}: unknown keys: {', '.join(unknown)}")
    card_id = item.get("id")
    if not isinstance(card_id, str) or not CARD_ID_RE.fullmatch(card_id):
        raise BackfillError(f"{where}: invalid id {card_id!r}")
    where = card_id
    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        raise BackfillError(f"{where}: title must be non-empty text")
    slug = item.get("slug") or slugify(title)
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug):
        raise BackfillError(f"{where}: invalid slug {slug!r}")
    try:
        status = Status(item.get("status", Status.PARKED.value))
    except ValueError as exc:
        raise BackfillError(f"{where}: {exc}") from exc
    if status not in _LEGACY_STATUSES:
        raise BackfillError(f"{where}: legacy cards are parked or rejected, not {status.value}")
    runs = _text_list(item.get("runs"), f"{where}.runs")
    studies = tuple(
        _study(study, f"{where}.studies[{number}]")
        for number, study in enumerate(item.get("studies") or [], start=1)
    )
    if not runs and not studies:
        raise BackfillError(f"{where}: a card needs at least one run or study trial")
    body = item.get("body", "")
    if not isinstance(body, str):
        raise BackfillError(f"{where}: body must be text")
    return CardPlan(
        id=card_id,
        title=title,
        slug=slug,
        status=status,
        tags=_text_list(item.get("tags"), f"{where}.tags"),
        criteria=_criteria(item.get("criteria"), where),
        body=body,
        runs=runs,
        studies=studies,
    )


def _study(item: object, where: str) -> StudyTrial:
    if not isinstance(item, Mapping):
        raise BackfillError(f"{where}: expected a mapping")
    unknown = sorted(str(key) for key in item.keys() - _STUDY_KEYS)
    if unknown:
        raise BackfillError(f"{where}: unknown keys: {', '.join(unknown)}")
    variant = item.get("variant")
    note = item.get("note")
    when = item.get("date")
    if not isinstance(variant, str) or not variant.strip():
        raise BackfillError(f"{where}: variant must be non-empty text")
    if not isinstance(note, str) or len(note.splitlines()) != 1:
        raise BackfillError(f"{where}: note must be one line of text")
    if not isinstance(when, date):
        raise BackfillError(f"{where}: date must be a YYYY-MM-DD date")
    symbols = tuple(symbol.upper() for symbol in _text_list(item.get("symbols"), where))
    if not symbols:
        raise BackfillError(f"{where}: a study trial names at least one symbol")
    return StudyTrial(variant, symbols, when, note, _study_metrics(item.get("metrics"), where))


def _study_metrics(value: object, where: str) -> dict[str, dict[str, float | int | None]]:
    """Symbol -> metric -> finite number or null, as the ledger stores it."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BackfillError(f"{where}: metrics must be a mapping")
    metrics: dict[str, dict[str, float | int | None]] = {}
    for symbol, values in value.items():
        if not isinstance(symbol, str) or not isinstance(values, Mapping):
            raise BackfillError(f"{where}: metrics.{symbol} must map names to numbers")
        checked: dict[str, float | int | None] = {}
        for name, number in values.items():
            valid = number is None or (
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                and math.isfinite(number)
            )
            if not isinstance(name, str) or not valid:
                raise BackfillError(f"{where}: metrics.{symbol}.{name} must be a finite number")
            checked[name] = number
        metrics[symbol.upper()] = checked
    return metrics


def _text_list(value: object, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise BackfillError(f"{where}: expected a list of text")
    return tuple(value)


def _criteria(value: object, where: str) -> Mapping[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise BackfillError(f"{where}: criteria must be a non-empty mapping")
    criteria: dict[str, float] = {}
    for key, threshold in value.items():
        if not isinstance(key, str) or isinstance(threshold, bool):
            raise BackfillError(f"{where}: invalid criterion {key!r}")
        if not isinstance(threshold, (int, float)):
            raise BackfillError(f"{where}: criterion {key} must be a number")
        criteria[key] = float(threshold)
    return criteria


def _check_coverage(plans: Sequence[CardPlan], runs_by_id: Mapping[str, LegacyRun]) -> None:
    mapped = [run_id for plan in plans for run_id in plan.runs]
    unknown = sorted(set(mapped) - runs_by_id.keys())
    twice = sorted({run_id for run_id in mapped if mapped.count(run_id) > 1})
    missing = sorted(runs_by_id.keys() - set(mapped))
    problems = []
    if unknown:
        problems.append(f"unknown runs: {', '.join(unknown)}")
    if twice:
        problems.append(f"runs mapped more than once: {', '.join(twice)}")
    if missing:
        problems.append(f"unmapped runs: {', '.join(missing)}")
    if problems:
        raise BackfillError("; ".join(problems))


def _ensure_card(
    plan: CardPlan,
    runs_by_id: Mapping[str, LegacyRun],
    research_dir: Path,
    default_criteria: Mapping[str, float],
    today: date,
) -> tuple[ResearchCard, int]:
    """The plan's card: the existing one if it agrees, else a newly written one."""
    criteria = dict(plan.criteria or default_criteria)
    dates = [runs_by_id[run_id].run_date for run_id in plan.runs]
    dates += [study.date for study in plan.studies]
    # end_date is exclusive, so the last day a legacy run saw stays out of the holdout.
    cutoff = max(dates) + timedelta(days=1)
    existing = load_card(research_dir, plan.id)
    if existing is not None:
        if existing.holdout_cutoff != cutoff or existing.criteria_hash != criteria_hash(criteria):
            raise BackfillError(
                f"{plan.id} exists with a different cutoff or criteria; "
                "legacy cards are frozen once written"
            )
        return existing, 0
    card = ResearchCard(
        id=plan.id,
        title=plan.title,
        status=plan.status,
        tags=plan.tags,
        created=today,
        holdout_cutoff=cutoff,
        holdout_contaminated=False,
        success_criteria=criteria,
        registered_criteria_hash=criteria_hash(criteria),
        criteria_amendments=(),
        body=plan.body,
    )
    try:
        write_card(research_dir, card, plan.slug)
    except CardError as exc:
        raise BackfillError(str(exc)) from exc
    return card, 1


def _legacy_trial(card: ResearchCard, run: LegacyRun) -> Trial:
    note = f"legacy run: {run.name}"
    if run.unresolved:
        note += f" ({run.unresolved})"
    return Trial(
        strategy=card.id,
        run_id=run.run_id,
        kind=run.kind,
        status=run.status,
        ts=run.ts,
        fingerprint=run.fingerprint,
        fingerprint_version=FINGERPRINT_VERSION,
        criteria_hash=card.criteria_hash,
        holdout_cutoff=card.holdout_cutoff,
        code=run.code,
        window=run.window,
        note=note.replace("\n", " "),
        metrics=run.metrics,
    )


def _study_trial(card: ResearchCard, study: StudyTrial, run_id: str) -> Trial:
    return Trial(
        strategy=card.id,
        run_id=run_id,
        kind=TrialKind.LEGACY,
        status=TrialStatus.OK,
        ts=study.date.isoformat(),
        fingerprint=digest({"study": card.id, "variant": study.variant, "symbols": study.symbols}),
        fingerprint_version=FINGERPRINT_VERSION,
        criteria_hash=card.criteria_hash,
        holdout_cutoff=card.holdout_cutoff,
        code={},
        window={"start": None, "end": None},
        note=study.note,
        metrics={symbol: dict(values) for symbol, values in study.metrics.items()},
    )
```

A failure in the middle of `apply_mapping` (for example a full disk) leaves
uncommitted cards and trials in the register; applying the same mapping
again skips what was recorded and commits the rest.

**Step 4:** same command → 19 passed.

**Step 5:** `git commit -m "feat(research): backfill legacy runs and studies"`

---

### Task 5: `fwbg research` CLI

> **Amended after code review (done).** Operator mistakes (malformed or
> missing mapping, apply before init, missing `--results`) end in `Error: …`
> on stderr with exit 1, never a traceback; `scan` does not create the
> register directory and refuses to overwrite before scanning; the draft
> keeps multi-line prose as YAML literal blocks; `scan` and `apply` warn
> about runs whose `progress.json` still says `starting`/`running`.

**Files:** Create `src/fwbg/research/cli.py`, `tests/research/test_cli.py`;
modify `src/fwbg/cli/main.py` (route `fwbg research …` like `fwbg portfolio …`).

**Step 1: Write the failing tests** (`tests/research/test_cli.py`):

```python
import yaml

from fwbg.research.cli import main
from fwbg.research.ledger import LEDGER_FILENAME, read_trials
from tests.research.factories import make_run


def test_init_scan_and_apply_end_to_end(tmp_path, monkeypatch, git_env, capsys):
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setenv("FWBG_TEST_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("FWBG_STRATEGIES_DIR", str(tmp_path / "strategies" / "configs"))
    make_run(tmp_path / "results", "r1", name="orb-test")
    make_run(tmp_path / "results", "r2", name="m3_smoke_x")
    mapping = tmp_path / "mapping.yaml"

    assert main(["init"]) == 0
    assert main(["backfill", "scan", "--out", str(mapping)]) == 0
    assert "2 runs (1 counted, 0 unresolvable)" in capsys.readouterr().out
    cards = yaml.safe_load(mapping.read_text(encoding="utf-8"))["cards"]
    assert [card["title"] for card in cards] == ["orb-test", "m3_smoke_x"]

    assert main(["backfill", "apply", str(mapping)]) == 0
    assert "2 cards written, 2 trials appended, committed" in capsys.readouterr().out
    assert len(read_trials(tmp_path / "research" / LEDGER_FILENAME)) == 2


def test_scan_refuses_to_overwrite_a_mapping(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setenv("FWBG_TEST_RESULTS_DIR", str(tmp_path / "results"))
    make_run(tmp_path / "results", "r1")
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("keep me\n", encoding="utf-8")
    assert main(["backfill", "scan", "--out", str(mapping)]) == 1
    assert mapping.read_text(encoding="utf-8") == "keep me\n"


def test_apply_reports_mapping_errors_without_traceback(tmp_path, monkeypatch, git_env, capsys):
    monkeypatch.setenv("FWBG_RESEARCH_DIR", str(tmp_path / "research"))
    monkeypatch.setenv("FWBG_TEST_RESULTS_DIR", str(tmp_path / "results"))
    make_run(tmp_path / "results", "r1")
    main(["init"])
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("cards: []\n", encoding="utf-8")
    assert main(["backfill", "apply", str(mapping)]) == 1
    assert "Error: unmapped runs: r1" in capsys.readouterr().out
```

**Step 2:** `PYTHONPATH=src $PY -m pytest tests/research/test_cli.py -q -p no:cacheprovider`
→ collection error (`No module named 'fwbg.research.cli'`).

**Step 3: Implement** (`src/fwbg/research/cli.py`):

```python
"""``fwbg research``: set up the research register and backfill legacy runs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import yaml

from fwbg.api.workspace import get_research_dir, get_strategies_dir, get_test_results_dir
from fwbg.research.backfill import BackfillError, apply_mapping, draft_mapping, parse_mapping
from fwbg.research.card import CardError
from fwbg.research.ledger import LedgerError
from fwbg.research.legacy import LegacyRun, LegacyRunError, scan_runs
from fwbg.research.repo import CRITERIA_TEMPLATE, ResearchRepoError, init_research_repo


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fwbg research", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="create the research register (idempotent)")
    backfill = commands.add_parser("backfill", help="import legacy runs and studies")
    actions = backfill.add_subparsers(dest="action", required=True)
    scan = actions.add_parser("scan", help="write a draft mapping of legacy runs to cards")
    scan.add_argument("--out", type=Path, required=True, help="mapping file to create")
    apply = actions.add_parser("apply", help="write the cards and trials of a mapping")
    apply.add_argument("mapping", type=Path, help="reviewed mapping file")
    for sub in (scan, apply):
        sub.add_argument("--results", type=Path, help="run directory root (default: workspace)")
    args = parser.parse_args(argv)

    research_dir = get_research_dir()
    try:
        if args.command == "init":
            created = init_research_repo(research_dir)
            print(f"{'Created' if created else 'Already initialised'}: {research_dir}")
            return 0
        runs = scan_runs(args.results or get_test_results_dir(), get_strategies_dir().parent)
        if args.action == "scan":
            return _scan(runs, args.out)
        return _apply(runs, args.mapping, research_dir)
    except (BackfillError, CardError, LedgerError, LegacyRunError, ResearchRepoError) as exc:
        print(f"Error: {exc}")
        return 1


def _scan(runs: list[LegacyRun], out: Path) -> int:
    if out.exists():
        print(f"Error: {out} exists; refusing to overwrite a mapping")
        return 1
    out.write_text(
        yaml.safe_dump(draft_mapping(runs), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    unresolved = sum(1 for run in runs if run.unresolved)
    counted = sum(1 for run in runs if run.status.value == "ok" and run.kind.value == "legacy")
    print(f"{len(runs)} runs ({counted} counted, {unresolved} unresolvable) -> {out}")
    return 0


def _apply(runs: list[LegacyRun], mapping_path: Path, research_dir: Path) -> int:
    plans = parse_mapping(yaml.safe_load(mapping_path.read_text(encoding="utf-8")))
    template = yaml.safe_load((research_dir / CRITERIA_TEMPLATE).read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        print(f"Error: {CRITERIA_TEMPLATE} must be a mapping of criteria")
        return 1
    result = apply_mapping(plans, runs, research_dir, template, date.today())
    print(
        f"{result.cards_written} cards written, {result.trials_appended} trials appended, "
        f"{'committed' if result.committed else 'nothing to commit'}"
    )
    return 0
```

and route the sub-command (`src/fwbg/cli/main.py`):

```diff
diff --git a/src/fwbg/cli/main.py b/src/fwbg/cli/main.py
index 88bf8d8..067d0b3 100644
--- a/src/fwbg/cli/main.py
+++ b/src/fwbg/cli/main.py
@@ -838,6 +838,11 @@ def main():
         portfolio_main(sys.argv[2:])
         return
 
+    if len(sys.argv) > 1 and sys.argv[1] == "research":
+        from fwbg.research.cli import main as research_main
+
+        raise SystemExit(research_main(sys.argv[2:]))
+
     # Handle 'data' subcommand (ETL: raw → datasource)
     if len(sys.argv) > 1 and sys.argv[1] == "data":
         _run_data_command(sys.argv[2:])
```

**Step 4:** same command → 3 passed.

**Step 5:** `git commit -m "feat(research): add fwbg research init and backfill commands"`

---

### Task 6: Verification and documentation

**Step 1: Checks**

```bash
../../.venv/bin/ruff check src/ packages/
../../.venv/bin/black --check src/fwbg/research tests/research
PYTHONPATH=src $PY -m mypy --follow-imports=silent src/fwbg/research
PYTHONPATH=src $PY -m pytest tests/research tests/test_code_patch.py tests/test_git_env.py tests/test_cli.py -q -p no:cacheprovider
```

Then the full suite (`tests src/fwbg/plugins src/fwbg/adapters/broker/ig
packages/fwbg-premium/src/fwbg_premium/plugins packages/fwbg-sdk/tests`) and
compare failures with `origin/develop` (six environment failures are known:
`fwbg_broker_ig` not installed, fwbg-premium imported from another checkout).

**Step 2: Real-data dry run, without touching the real register.** Point
`FWBG_RESEARCH_DIR` and `GIT_CONFIG_GLOBAL` (a temp file with a user name and
email) at scratch locations, then run (from the worktree, so the branch
code is used — the installed `fwbg` script points at the primary checkout)
`PYTHONPATH=src $PY -m fwbg.cli research init`,
`… research backfill scan --out <scratch>/mapping.yaml` and
`… research backfill apply <scratch>/mapping.yaml` twice. Expect the
numbers from decision 8 (or more, if new runs exist) and a second apply that
adds nothing. Confirm `~/fwbg/research` was not created.

**Step 3:** In the design doc, set the status to
`Design approved (2026-10-19) — steps 1 (core) and 2 (backfill) implemented`
and commit `docs: mark research register backfill as implemented`.

---

### Task 7: Run the backfill (operator session, no code)

Done together with the operator after the branch is merged; nothing here is
committed to this repository.

**Preconditions:** stop the API and any active runs between `scan` and
`apply` (`scan`/`apply` warn about runs whose `progress.json` still says
`running`; on 2026-10-20 seven such files were stale leftovers). Agree the
grouping completely before the first `apply`: afterwards card text lives in
the card files, and regrouping requires resetting the register (possible
only while it has not been pushed).

1. **Consolidate run directories.** Copy runs that exist only in the
   repo-local `test_results/` (the `alt-*` alternative-data runs and the
   `portfolio/` study artefacts) into the workspace `test_results/`;
   verify each copy with `diff -r`. The repo-local `creative-*` directories
   are byte-identical copies of workspace runs — delete them only after a
   `diff -r` check and the operator's explicit confirmation.
2. `fwbg research init`.
3. `fwbg research backfill scan --out <register>/reports/backfill-mapping.yaml`.
4. **Regroup with the operator.** Merge name variants into idea cards
   (e.g. the `orb-test-rrr-v*` series), set tags and hypothesis prose,
   `rejected` with a verdict only where a conclusion is documented, per-card
   criteria overrides (e.g. `max_drawdown`). Add study cards with study
   trials, counted as documented: the Jev signal provider feasibility
   check, the calendar-proxy test, the swing study, the monthly portfolio
   research, and the alternative-data study (its `alt-*` runs are regular
   runs on that card). Copy the study write-ups into `<register>/reports/`
   and link them from the cards.
5. `fwbg research backfill apply <register>/reports/backfill-mapping.yaml`,
   then commit `reports/` in the register repository by hand.
6. **Verify:** every card loads; the ledger reads back; the trial count per
   card matches the mapping; a second apply adds nothing.
7. **Remote:** decide whether the register gets a private remote (it holds
   the research results; never push it to a public repository).
