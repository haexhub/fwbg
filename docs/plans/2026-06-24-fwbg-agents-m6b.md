# fwbg-agents M6b — Paper-Analyst + Promote-Live Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or superpowers:subagent-driven-development if running inline) to implement this plan task-by-task.

**Status:** ✓ done 2026-06-24 — 8 tasks + 5 polish refactors = 13 agents commits, agents-only repo, alembic 0005→0006. Final HEAD `eccb07e`. Tests 282→323 (+41). Manual `scripts/m6b_smoke.py` against dev DB ends `[m6b_smoke] PASSED` twice in a row (idempotent). Commits: `e558b17` (T1 migration), `33301e4` (T2 YAMLs), `ffa7dfb`+`ec2b07a` (T3 criteria_paper +polish), `61b997c`+`54ac7dd` (T4 PaperAnalyst +polish), `51fdf9c`+`0f20213` (T5 paper_flow +polish), `16d8003`+`ca00697` (T6 paper-analyze endpoint +polish), `f3daf61`+`d17807f` (T7 promote-live endpoint +polish), `eccb07e` (T8 smoke).

**Goal:** Close the decision loop after M6a's telemetry pipeline. Once a Strategy has been running in `PAPER_TRADING` long enough to have meaningful summary metrics on disk, an LLM **Paper-Analyst** reads the summary + positions, compares against hand-curated paper-criteria YAMLs, and emits a typed recommendation: **PromotePaperToLive**, **AbandonPaper**, or **ContinueObservation**. The Analyst **never** transitions state — it only writes a sidecar and sets a metadata flag. A separate **human-gated** `POST /promote-live` endpoint is the only path into `LIVE_TRADING`, requiring `human_approval=True` AND a prior promote-recommended flag.

**Architecture:**
- **Hand-curated paper-criteria YAMLs at `data/criteria/paper/<class>.yaml`.** Schema mirrors M2's backtest-criteria (`required_all`, `hard_blockers` lists of `{metric: "<op> <value>"}` entries). Looser thresholds than backtest (sharpe ≥0.8 vs backtest's ≥1.5). 3 classes shipped: equity, forex, crypto.
- **New `orchestrator/criteria_paper.py`** — concrete parallel to M2's `check_backtest_criteria`. `load_paper_criteria(asset_class) -> dict` + `evaluate_paper_criteria(summary: PaperTradeSummary, criteria: dict) -> CriteriaEvalResult(passed, failures)`. Copies the ~15-line comparator parser from `lifecycle._eval_comparator` (concrete-before-generic — extract to shared module only when a 3rd evaluator appears).
- **New `agents/paper_analyst.py`** — pydantic-ai agent producing a discriminated union: `PromotePaperToLive | AbandonPaper | ContinueObservation`. System prompt instructs: bias toward `ContinueObservation` unless criteria pass cleanly (→ Promote) OR persistent loss-bias / catastrophic DD (→ Abandon). Deterministic validator runs **after** the LLM output: re-checks paper-criteria for Promote (mirrors M3's hard-rules pattern), auto-fills `post_mortem_path` default for Abandon if LLM omits, sets `stale=True` if `days_in_paper > paper_phase_target_days`.
- **New `orchestrator/paper_flow.py::paper_analyze(strategy_id, session)`** — loads summary from disk, runs the agent through the validator, writes sidecar `data/strategies/<slug>/paper_analyst_<ar_id>.json`, sets `Strategy.metadata["paper_analyst_promote_recommended"]` or `["paper_analyst_abandon_recommended"]` flag. No state transition.
- **Two new POST endpoints:**
  - `POST /strategies/{id}/paper-analyze` — kicks `paper_analyze` via BackgroundTasks. 202 + AgentRun envelope. 422 if not in `PAPER_TRADING` or no on-disk data yet.
  - `POST /strategies/{id}/promote-live` — double-gated. Body `{human_approval: true, operator_note?: str}`. 422 unless body says yes AND metadata flag is set AND state is `PAPER_TRADING`. On pass: calls `transition_strategy(s, LIVE_TRADING, payload={human_approval: True, operator_note, ...})` — M2's lifecycle guard re-checks `human_approval==True` (double-gate). Creates an audit `AgentRun(agent_name="promote_live")`.
- **Migration 0006** — adds `Strategy.metadata: JSON nullable default {}`. Generic vehicle; future recommendation flags live here without new migrations.

**Tech Stack:** Python 3.13, SQLAlchemy 2.x async, pydantic-ai, FastAPI BackgroundTasks, pytest, alembic, pydantic-ai's `TestModel`/`FunctionModel` for agent tests. No new dependencies.

**Locked Decisions (in addition to M6a's A-H):**
- **(I) Flag storage = JSON metadata column.** Migration 0006 adds `Strategy.metadata` (JSON, nullable, default `{}`). Generic — future flags go here without further migrations.
- **(J) 3 hand-curated paper-criteria YAMLs at minimum.** equity, forex, crypto. Schema: top-level `required_all` and `hard_blockers`. NOT nested under a `paper_to_live` key (those exist in M2 YAMLs via the calibrator but are currently unread — we keep `data/criteria/paper/` as its own flat tree to avoid cross-coupling).
- **(K) Paper-Analyst never transitions state.** Only writes sidecar + sets metadata flag. The transition is a separate human-gated endpoint.
- **(L) Promote-live is double-gated.** Body `human_approval=true` AND prior `paper_analyst_promote_recommended` flag AND M2's `transition_strategy` re-checks `human_approval=True`. Three gates, only one of which the LLM can influence (and only positively — it can recommend, not approve).
- **(M) `paper_phase_target_days` is a soft warning.** When elapsed days exceed it, the validator forces `stale=True` on `ContinueObservation`. There is no auto-abandon. Humans decide.
- **(N) Concrete-before-generic.** `criteria_paper.py` does not share a base module with M2's `check_backtest_criteria`. Same shape, distinct file. The comparator parser is duplicated (~15 LOC). Extract only when a 3rd evaluator appears (M7 live-trading risk gates).

**Pre-checks (verified at session start):**
- agents HEAD = `fa0a92a` ✓
- agents `VIRTUAL_ENV= uv run pytest -q` = 282 passed ✓
- agents `VIRTUAL_ENV= uv run alembic current` = `0005 (head)` ✓
- agents `VIRTUAL_ENV= uv run python scripts/m6a_smoke.py` = `[m6a_smoke] PASSED` ✓ (idempotent)

**File-layout overview (agents-side adds, repo `~/Projekte/fwbg-agents/`):**
```
src/fwbg_agents/
  persistence/migrations/versions/0006_strategy_metadata.py   (new)
  persistence/models.py                                       (modify — Strategy gets .metadata: JSON)
  orchestrator/criteria_paper.py                              (new)
  agents/paper_analyst.py                                     (new)
  agents/prompts/paper_analyst.md                             (new)
  orchestrator/paper_flow.py                                  (new)
  api/strategies.py                                           (modify — 2 new POST endpoints)
data/criteria/paper/
  equity.yaml                                                 (new — hand-curated)
  forex.yaml                                                  (new — hand-curated)
  crypto.yaml                                                 (new — hand-curated)
scripts/
  m6b_smoke.py                                                (new — end-to-end)
tests/
  orchestrator/test_criteria_paper.py                         (new)
  agents/test_paper_analyst.py                                (new)
  orchestrator/test_paper_flow.py                             (new)
  api/test_strategies_paper_analyze.py                        (new)
  api/test_strategies_promote_live.py                         (new)
```

---

## Task 1 — Migration 0006: Strategy.metadata JSON column

**Files:**
- Create: `src/fwbg_agents/persistence/migrations/versions/0006_strategy_metadata.py`
- Modify: `src/fwbg_agents/persistence/models.py:120-147` — add `metadata: Mapped[dict] = mapped_column("metadata_", JSON, nullable=False, server_default="{}", default=dict)`. (Column name `metadata_` because SQLAlchemy reserves `metadata` on Base; Python attribute stays `.metadata` if we name the attribute differently — actually simpler: use Python attr `meta` mapped to DB column `metadata_`, since SQLAlchemy DeclarativeBase has `metadata` on the class itself. Verify with a probe in step 1a.)

**Step 1a — Probe SQLAlchemy `metadata` reservation**

```bash
cd ~/Projekte/fwbg-agents
VIRTUAL_ENV= uv run python -c "
from fwbg_agents.persistence.models import Strategy, Base
print('Base.metadata exists:', hasattr(Base, 'metadata'))
print('Strategy.metadata exists:', hasattr(Strategy, 'metadata'))
"
```
Expected: both `True`. Confirms we must use a different Python attribute name. **Decision: Python attribute = `metadata_json`, DB column = `metadata_json`. Same name both sides for clarity.**

**Step 1b — Update models.py**

```python
# in class Strategy, before `created_at`:
metadata_json: Mapped[dict] = mapped_column(
    "metadata_json", JSON, nullable=False, server_default="{}", default=dict
)
```

**Step 1c — Write migration**

```python
"""strategy_metadata_json

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-24
"""
from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"

def upgrade() -> None:
    with op.batch_alter_table("strategy") as batch:
        batch.add_column(sa.Column("metadata_json", sa.JSON(), nullable=False, server_default="{}"))

def downgrade() -> None:
    with op.batch_alter_table("strategy") as batch:
        batch.drop_column("metadata_json")
```

**Step 1d — Verify**

```bash
cd ~/Projekte/fwbg-agents
VIRTUAL_ENV= uv run alembic upgrade head
VIRTUAL_ENV= uv run alembic current   # expect: 0006 (head)
VIRTUAL_ENV= uv run pytest -q          # expect: 282 passed (no behaviour change)
```

**Step 1e — Commit**

```bash
git add src/fwbg_agents/persistence/migrations/versions/0006_strategy_metadata.py \
        src/fwbg_agents/persistence/models.py
git commit -m "feat(M6b): alembic 0006 — Strategy.metadata_json JSON column (generic vehicle for recommendation flags)"
```

---

## Task 2 — Hand-curated paper-criteria YAMLs (3 asset classes)

**Files:**
- Create: `data/criteria/paper/equity.yaml`
- Create: `data/criteria/paper/forex.yaml`
- Create: `data/criteria/paper/crypto.yaml`

**Schema** (flat, top-level `required_all` and `hard_blockers`, each a list of `{<metric>: "<op> <value>"}`):

```yaml
# data/criteria/paper/equity.yaml
required_all:
  - sharpe_paper: ">= 0.8"
  - win_rate: ">= 0.40"
  - trades_total: ">= 30"
hard_blockers:
  - max_dd_paper: "<= 0.25"
```

```yaml
# data/criteria/paper/forex.yaml
required_all:
  - sharpe_paper: ">= 1.0"
  - win_rate: ">= 0.45"
  - trades_total: ">= 30"
hard_blockers:
  - max_dd_paper: "<= 0.15"
```

```yaml
# data/criteria/paper/crypto.yaml
required_all:
  - sharpe_paper: ">= 0.6"
  - win_rate: ">= 0.35"
  - trades_total: ">= 20"
hard_blockers:
  - max_dd_paper: "<= 0.30"
```

**Rationale:** thresholds are looser than M2 backtest (which uses sharpe≥1.5 across the board) because paper performance is real-time and noisier than out-of-sample backtest. Crypto is loosest (volatility), forex strictest (most efficient market in our universe). Per [[feedback-no-data-derived-thresholds]] these are hand-curated — they do NOT come from a calibrator quantile.

**Step 2a — Write all three files** (no test needed yet — YAMLs are pure data, the loader tests them in Task 3).

**Step 2b — Sanity-check YAML parses**

```bash
cd ~/Projekte/fwbg-agents
VIRTUAL_ENV= uv run python -c "
import yaml, pathlib
for f in pathlib.Path('data/criteria/paper').glob('*.yaml'):
    d = yaml.safe_load(f.read_text())
    assert 'required_all' in d and 'hard_blockers' in d, f
    print(f.name, 'ok')
"
```
Expected: 3 lines `<class>.yaml ok`.

**Step 2c — Commit**

```bash
git add data/criteria/paper/
git commit -m "feat(M6b): hand-curated paper-criteria YAMLs for equity/forex/crypto"
```

---

## Task 3 — `orchestrator/criteria_paper.py` (loader + evaluator + tests)

**Files:**
- Create: `src/fwbg_agents/orchestrator/criteria_paper.py`
- Create: `tests/orchestrator/test_criteria_paper.py`

**Behaviour:**
- `load_paper_criteria(asset_class: str, *, criteria_dir: Path | None = None) -> dict` — loads `<criteria_dir>/<asset_class.lower()>.yaml`. Defaults to `<repo_root>/data/criteria/paper/`. Raises `FileNotFoundError` if asset_class has no YAML.
- `evaluate_paper_criteria(summary: PaperTradeSummary, criteria: dict) -> CriteriaEvalResult` — returns `CriteriaEvalResult(passed: bool, failures: list[str])`. `passed` is True iff ALL `required_all` entries are True AND ALL `hard_blockers` are True. Each failure entry is human-readable like `"sharpe_paper: 0.6 < 0.8"`.

**Step 3a — Write the failing tests**

```python
# tests/orchestrator/test_criteria_paper.py
import pytest
from pathlib import Path
from fwbg_agents.orchestrator.criteria_paper import (
    load_paper_criteria, evaluate_paper_criteria, CriteriaEvalResult
)
from fwbg_agents.tools.fwbg_paper_reader import PaperTradeSummary


def _make_summary(**overrides):
    base = dict(
        sharpe_paper=1.2, max_dd_paper=0.10, trades_total=50, trades_today=2,
        days_in_paper=45, win_rate=0.55, last_trade_at="2026-06-20T10:00:00Z",
        current_equity=10500.0, starting_equity=10000.0, equity_curve_sample=[],
    )
    base.update(overrides)
    return PaperTradeSummary(**base)


def test_load_paper_criteria_forex_returns_dict_with_required_keys():
    d = load_paper_criteria("forex")
    assert "required_all" in d
    assert "hard_blockers" in d


def test_load_paper_criteria_unknown_class_raises():
    with pytest.raises(FileNotFoundError):
        load_paper_criteria("nonexistent")


def test_evaluate_passes_when_all_metrics_clear_thresholds():
    criteria = {"required_all": [{"sharpe_paper": ">= 0.8"}], "hard_blockers": [{"max_dd_paper": "<= 0.25"}]}
    res = evaluate_paper_criteria(_make_summary(sharpe_paper=1.0, max_dd_paper=0.10), criteria)
    assert res.passed is True
    assert res.failures == []


def test_evaluate_fails_when_hard_blocker_breached():
    criteria = {"required_all": [{"sharpe_paper": ">= 0.8"}], "hard_blockers": [{"max_dd_paper": "<= 0.25"}]}
    res = evaluate_paper_criteria(_make_summary(sharpe_paper=1.0, max_dd_paper=0.30), criteria)
    assert res.passed is False
    assert any("max_dd_paper" in f for f in res.failures)
```

**Step 3b — Run, expect ImportError**

```bash
cd ~/Projekte/fwbg-agents
VIRTUAL_ENV= uv run pytest tests/orchestrator/test_criteria_paper.py -v
```
Expected: ImportError on `fwbg_agents.orchestrator.criteria_paper`.

**Step 3c — Implement**

```python
# src/fwbg_agents/orchestrator/criteria_paper.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

from fwbg_agents.tools.fwbg_paper_reader import PaperTradeSummary

_DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "criteria" / "paper"


@dataclass
class CriteriaEvalResult:
    passed: bool
    failures: list[str]


def load_paper_criteria(asset_class: str, *, criteria_dir: Path | None = None) -> dict[str, Any]:
    base = criteria_dir or _DEFAULT_DIR
    path = base / f"{asset_class.lower()}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no paper-criteria YAML for asset_class={asset_class!r}: {path}")
    return yaml.safe_load(path.read_text()) or {}


# Concrete copy of lifecycle._eval_comparator — see locked decision (N).
_OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        ">":  lambda a, b: a >  b, "<":  lambda a, b: a <  b,
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


def _eval(metric: str, value: float, expr: str) -> tuple[bool, str]:
    expr = expr.strip()
    for op in (">=", "<=", "==", "!=", ">", "<"):
        if expr.startswith(op):
            threshold = float(expr[len(op):].strip())
            ok = _OPS[op](value, threshold)
            return ok, f"{metric}: {value} {op} {threshold} -> {'pass' if ok else 'fail'}"
    raise ValueError(f"unparseable comparator: {expr!r}")


def evaluate_paper_criteria(summary: PaperTradeSummary, criteria: dict[str, Any]) -> CriteriaEvalResult:
    metrics = summary.model_dump() if hasattr(summary, "model_dump") else summary.__dict__
    failures: list[str] = []
    for section in ("required_all", "hard_blockers"):
        for entry in criteria.get(section, []) or []:
            for metric, expr in entry.items():
                if metric not in metrics or metrics[metric] is None:
                    failures.append(f"{metric}: missing from summary")
                    continue
                ok, _msg = _eval(metric, float(metrics[metric]), expr)
                if not ok:
                    failures.append(f"{metric}: {metrics[metric]} fails '{expr}'")
    return CriteriaEvalResult(passed=not failures, failures=failures)
```

**Step 3d — Verify green**

```bash
VIRTUAL_ENV= uv run pytest tests/orchestrator/test_criteria_paper.py -v
```
Expected: 4 passed.

**Step 3e — Commit**

```bash
git add src/fwbg_agents/orchestrator/criteria_paper.py tests/orchestrator/test_criteria_paper.py
git commit -m "feat(M6b): paper-criteria loader + evaluator (concrete parallel to M2 backtest-criteria)"
```

---

## Task 4 — Paper-Analyst agent + prompt + tests

**Files:**
- Create: `src/fwbg_agents/agents/paper_analyst.py`
- Create: `src/fwbg_agents/agents/prompts/paper_analyst.md`
- Create: `tests/agents/test_paper_analyst.py`

**Discriminated union output:**
```python
class PromotePaperToLive(BaseModel):
    decision: Literal["promote_paper_to_live"] = "promote_paper_to_live"
    rationale: str

class AbandonPaper(BaseModel):
    decision: Literal["abandon_paper"] = "abandon_paper"
    rationale: str
    post_mortem_path: str | None = None  # validator fills default if None

class ContinueObservation(BaseModel):
    decision: Literal["continue_observation"] = "continue_observation"
    rationale: str
    stale: bool = False  # validator forces True if days_in_paper > paper_phase_target_days

PaperAnalystOutput = Annotated[
    PromotePaperToLive | AbandonPaper | ContinueObservation,
    Discriminator("decision"),
]
```

**Validator behaviour** (runs AFTER LLM, mirrors M3 `validate_and_apply`):
- If `PromotePaperToLive` and `evaluate_paper_criteria(...).passed is False`: **raise** `PaperAnalystValidationError` (LLM cannot bypass hard rules).
- If `AbandonPaper` and `post_mortem_path is None`: fill `data/strategies/<slug>/paper_post_mortem.md`.
- If `ContinueObservation` and `summary.days_in_paper > strategy.paper_phase_target_days`: force `stale=True`.

**Step 4a — Write the prompt**

`src/fwbg_agents/agents/prompts/paper_analyst.md`:
```markdown
You are the Paper-Analyst for the fwbg trading system. A strategy has been running in paper-trading mode and you must decide its next step from real-time paper-trading telemetry.

You will receive:
- `summary`: PaperTradeSummary (sharpe_paper, max_dd_paper, trades_total, days_in_paper, win_rate, equity curve).
- `positions`: PaperPositions (currently-open positions with SL/TP).
- `paper_criteria`: hand-curated thresholds for this asset class.
- `paper_phase_target_days`: configured target duration of the paper phase.
- `paper_criteria_eval`: pre-computed CriteriaEvalResult against the summary.

Choose ONE of three decisions:

1. **promote_paper_to_live** — paper performance clearly clears the criteria AND no concerning recent behaviour. Only choose this when paper_criteria_eval.passed is True AND the equity curve trends up over the last 30+ days AND no catastrophic drawdown in the last 14 days.

2. **abandon_paper** — irrecoverable: persistent loss-bias (>50% losing trades for 30+ days), max-DD breach beyond hard_blockers, or correlated systematic failures. Write a brief `rationale` and let the system fill the post_mortem_path.

3. **continue_observation** — default. Choose when the strategy has not yet produced enough data, is borderline, or is trending positively but not yet clearing thresholds. Set `stale=true` if the paper phase has run longer than `paper_phase_target_days` without a clear signal.

Bias strongly toward continue_observation. Only promote when criteria pass cleanly. Only abandon when the evidence is unambiguous.

Output: structured JSON matching the discriminated union.
```

**Step 4b — Write tests (using pydantic-ai TestModel)**

```python
# tests/agents/test_paper_analyst.py
import pytest
from pydantic_ai.models.test import TestModel
from fwbg_agents.agents.paper_analyst import (
    PaperAnalyst, PaperAnalystValidationError,
    PromotePaperToLive, AbandonPaper, ContinueObservation,
)
from fwbg_agents.tools.fwbg_paper_reader import PaperTradeSummary
from fwbg_agents.orchestrator.criteria_paper import CriteriaEvalResult


def _summary(**o):
    base = dict(
        sharpe_paper=1.0, max_dd_paper=0.10, trades_total=50, trades_today=1,
        days_in_paper=45, win_rate=0.55, last_trade_at=None,
        current_equity=10500.0, starting_equity=10000.0, equity_curve_sample=[],
    )
    base.update(o)
    return PaperTradeSummary(**base)


def test_promote_passes_when_criteria_pass():
    model = TestModel(custom_output_args={"decision": "promote_paper_to_live", "rationale": "clean pass"})
    analyst = PaperAnalyst(model=model)
    out = analyst.analyze_sync(
        summary=_summary(),
        positions=None,
        paper_criteria={"required_all": [], "hard_blockers": []},
        paper_phase_target_days=90,
        paper_criteria_eval=CriteriaEvalResult(passed=True, failures=[]),
        strategy_slug="s",
    )
    assert isinstance(out, PromotePaperToLive)


def test_promote_rejected_when_criteria_fail():
    model = TestModel(custom_output_args={"decision": "promote_paper_to_live", "rationale": "x"})
    analyst = PaperAnalyst(model=model)
    with pytest.raises(PaperAnalystValidationError):
        analyst.analyze_sync(
            summary=_summary(),
            positions=None,
            paper_criteria={"required_all": [], "hard_blockers": []},
            paper_phase_target_days=90,
            paper_criteria_eval=CriteriaEvalResult(passed=False, failures=["sharpe_paper: 0.5 fails '>= 0.8'"]),
            strategy_slug="s",
        )


def test_abandon_fills_default_post_mortem_path_when_omitted():
    model = TestModel(custom_output_args={"decision": "abandon_paper", "rationale": "persistent loss"})
    analyst = PaperAnalyst(model=model)
    out = analyst.analyze_sync(
        summary=_summary(),
        positions=None,
        paper_criteria={"required_all": [], "hard_blockers": []},
        paper_phase_target_days=90,
        paper_criteria_eval=CriteriaEvalResult(passed=False, failures=[]),
        strategy_slug="abc",
    )
    assert isinstance(out, AbandonPaper)
    assert out.post_mortem_path is not None
    assert "abc" in out.post_mortem_path
    assert out.post_mortem_path.endswith("paper_post_mortem.md")


def test_continue_observation_forces_stale_when_days_exceed_target():
    model = TestModel(custom_output_args={"decision": "continue_observation", "rationale": "borderline", "stale": False})
    analyst = PaperAnalyst(model=model)
    out = analyst.analyze_sync(
        summary=_summary(days_in_paper=120),
        positions=None,
        paper_criteria={"required_all": [], "hard_blockers": []},
        paper_phase_target_days=90,
        paper_criteria_eval=CriteriaEvalResult(passed=False, failures=[]),
        strategy_slug="s",
    )
    assert isinstance(out, ContinueObservation)
    assert out.stale is True


def test_continue_observation_keeps_stale_false_when_under_target():
    model = TestModel(custom_output_args={"decision": "continue_observation", "rationale": "early", "stale": False})
    analyst = PaperAnalyst(model=model)
    out = analyst.analyze_sync(
        summary=_summary(days_in_paper=40),
        positions=None,
        paper_criteria={"required_all": [], "hard_blockers": []},
        paper_phase_target_days=90,
        paper_criteria_eval=CriteriaEvalResult(passed=False, failures=[]),
        strategy_slug="s",
    )
    assert isinstance(out, ContinueObservation)
    assert out.stale is False
```

**Step 4c — Implement** (sketch — full code follows the M3 Analyst pattern at `src/fwbg_agents/agents/analyst.py` for agent construction, deps wiring, `analyze_sync` vs `analyze` async):

```python
# src/fwbg_agents/agents/paper_analyst.py
from __future__ import annotations
from pathlib import Path
from typing import Annotated, Literal
from pydantic import BaseModel, Discriminator
from pydantic_ai import Agent
from fwbg_agents.agents.analyst import default_model  # reuse helper
from fwbg_agents.tools.fwbg_paper_reader import PaperTradeSummary, PaperPositions
from fwbg_agents.orchestrator.criteria_paper import CriteriaEvalResult


class PaperAnalystValidationError(Exception):
    pass


class PromotePaperToLive(BaseModel):
    decision: Literal["promote_paper_to_live"] = "promote_paper_to_live"
    rationale: str


class AbandonPaper(BaseModel):
    decision: Literal["abandon_paper"] = "abandon_paper"
    rationale: str
    post_mortem_path: str | None = None


class ContinueObservation(BaseModel):
    decision: Literal["continue_observation"] = "continue_observation"
    rationale: str
    stale: bool = False


PaperAnalystOutput = Annotated[
    PromotePaperToLive | AbandonPaper | ContinueObservation,
    Discriminator("decision"),
]


_PROMPT = (Path(__file__).parent / "prompts" / "paper_analyst.md").read_text()


class PaperAnalyst:
    def __init__(self, *, model=None):
        self.model = model or default_model()
        self.agent = Agent(self.model, output_type=PaperAnalystOutput, system_prompt=_PROMPT)

    def analyze_sync(self, *, summary, positions, paper_criteria, paper_phase_target_days,
                     paper_criteria_eval, strategy_slug, data_dir: Path | None = None):
        user_payload = {
            "summary": summary.model_dump() if hasattr(summary, "model_dump") else summary.__dict__,
            "positions": (positions.model_dump() if positions else None),
            "paper_criteria": paper_criteria,
            "paper_phase_target_days": paper_phase_target_days,
            "paper_criteria_eval": {"passed": paper_criteria_eval.passed,
                                    "failures": paper_criteria_eval.failures},
        }
        result = self.agent.run_sync(str(user_payload))
        out = result.output
        return self._validate(out, summary=summary, paper_phase_target_days=paper_phase_target_days,
                              paper_criteria_eval=paper_criteria_eval, strategy_slug=strategy_slug,
                              data_dir=data_dir)

    def _validate(self, out, *, summary, paper_phase_target_days, paper_criteria_eval,
                  strategy_slug, data_dir):
        if isinstance(out, PromotePaperToLive):
            if not paper_criteria_eval.passed:
                raise PaperAnalystValidationError(
                    f"Promote rejected — criteria failures: {paper_criteria_eval.failures}"
                )
        elif isinstance(out, AbandonPaper):
            if out.post_mortem_path is None:
                base = data_dir or Path("data") / "strategies" / strategy_slug
                out = out.model_copy(update={"post_mortem_path": str(base / "paper_post_mortem.md")})
        elif isinstance(out, ContinueObservation):
            if summary.days_in_paper > paper_phase_target_days and not out.stale:
                out = out.model_copy(update={"stale": True})
        return out
```

**Step 4d — Verify green**

```bash
VIRTUAL_ENV= uv run pytest tests/agents/test_paper_analyst.py -v
```
Expected: 5 passed.

**Step 4e — Commit**

```bash
git add src/fwbg_agents/agents/paper_analyst.py src/fwbg_agents/agents/prompts/paper_analyst.md tests/agents/test_paper_analyst.py
git commit -m "feat(M6b): PaperAnalyst pydantic-ai agent — Promote/Abandon/Continue with hard-rule validator"
```

---

## Task 5 — `orchestrator/paper_flow.py::paper_analyze` + tests

**Files:**
- Create: `src/fwbg_agents/orchestrator/paper_flow.py`
- Create: `tests/orchestrator/test_paper_flow.py`

**Behaviour of `paper_analyze(strategy_id, session, *, settings=None, analyst=None) -> AgentRun`:**
1. Load Strategy. Assert `current_state == PAPER_TRADING` else raise `PaperFlowError`.
2. Resolve `asset_class = strategy.asset_class` (top-level column — confirmed in models.py:132).
3. `load_paper_criteria(asset_class)` — raise if YAML missing.
4. `read_paper_summary(strategy.slug, settings.fwbg_data_dir)`. If None → raise `PaperFlowError("no on-disk paper-trading data yet")`.
5. `read_paper_positions(strategy.slug, settings.fwbg_data_dir)` — may be None, that's ok.
6. Create `AgentRun(agent_name="paper_analyst", status=RUNNING, strategy_id=strategy.id)`, commit, capture id.
7. `eval = evaluate_paper_criteria(summary, criteria)`.
8. `out = analyst.analyze_sync(...)`.
9. Write sidecar `<data_dir>/strategies/<slug>/paper_analyst_<ar.id>.json` containing `out.model_dump()`.
10. If `out` is `PromotePaperToLive`: set `strategy.metadata_json["paper_analyst_promote_recommended"] = True` and persist.
11. If `out` is `AbandonPaper`: set `strategy.metadata_json["paper_analyst_abandon_recommended"] = True` and `["paper_analyst_post_mortem_path"] = out.post_mortem_path` and persist.
12. If `out` is `ContinueObservation`: nothing on the Strategy (sidecar only).
13. Mark AgentRun `status=DONE`, `output_artifact_path=<sidecar path>`. Commit.
14. On exception inside the try: set `status=FAILED`, `error=str(exc)`, re-raise.
15. Return the AgentRun.

**Step 5a — Write tests** (use a stub `PaperAnalyst`-like object whose `analyze_sync` returns a pre-built output, and a fake `settings.fwbg_data_dir` pointing at a tmp-path with synthesised trades.jsonl + status.json):

```python
# tests/orchestrator/test_paper_flow.py — 6 tests:
# 1. raises PaperFlowError when strategy not in PAPER_TRADING
# 2. raises PaperFlowError when no on-disk data
# 3. promote outcome sets metadata_json["paper_analyst_promote_recommended"]=True + AgentRun DONE
# 4. abandon outcome sets metadata_json["paper_analyst_abandon_recommended"]=True + post_mortem_path
# 5. continue_observation outcome: metadata unchanged, sidecar only
# 6. exception inside analyst.analyze_sync marks AgentRun FAILED and re-raises
```

(Full test code follows the same fixture patterns as `tests/orchestrator/test_plugin_flow.py` — use `AsyncSession` test fixture, write tmp `trades.jsonl`/`status.json` via existing fwbg_paper_reader helpers in tests if available, else inline-write small JSONL.)

**Step 5b → 5d — Red, Implement, Green.** Implementation skeleton:

```python
# src/fwbg_agents/orchestrator/paper_flow.py
from __future__ import annotations
import json
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fwbg_agents.config import get_settings
from fwbg_agents.persistence.models import Strategy, AgentRun, AgentRunStatus, StrategyState
from fwbg_agents.tools.fwbg_paper_reader import read_paper_summary, read_paper_positions
from fwbg_agents.orchestrator.criteria_paper import load_paper_criteria, evaluate_paper_criteria
from fwbg_agents.agents.paper_analyst import (
    PaperAnalyst, PromotePaperToLive, AbandonPaper, ContinueObservation,
)
from datetime import datetime, timezone


class PaperFlowError(Exception):
    pass


async def paper_analyze(strategy_id: int, session: AsyncSession, *, settings=None, analyst=None) -> AgentRun:
    settings = settings or get_settings()
    analyst = analyst or PaperAnalyst()
    strategy = (await session.execute(select(Strategy).where(Strategy.id == strategy_id))).scalar_one_or_none()
    if strategy is None:
        raise PaperFlowError(f"strategy {strategy_id} not found")
    if strategy.current_state != StrategyState.PAPER_TRADING.value:
        raise PaperFlowError(f"strategy {strategy_id} not in PAPER_TRADING (state={strategy.current_state})")
    criteria = load_paper_criteria(strategy.asset_class)
    summary = read_paper_summary(strategy.slug, Path(settings.fwbg_data_dir))
    if summary is None:
        raise PaperFlowError(f"no on-disk paper-trading data for slug={strategy.slug}")
    positions = read_paper_positions(strategy.slug, Path(settings.fwbg_data_dir))

    now = datetime.now(timezone.utc)
    ar = AgentRun(agent_name="paper_analyst", status=AgentRunStatus.RUNNING.value,
                  strategy_id=strategy.id, started_at=now, created_at=now)
    session.add(ar)
    await session.commit()
    await session.refresh(ar)

    try:
        eval_res = evaluate_paper_criteria(summary, criteria)
        out = analyst.analyze_sync(
            summary=summary, positions=positions, paper_criteria=criteria,
            paper_phase_target_days=strategy.paper_phase_target_days,
            paper_criteria_eval=eval_res, strategy_slug=strategy.slug,
        )
        sidecar_dir = Path(settings.fwbg_data_dir).parent / "strategies" / strategy.slug \
            if False else Path("data") / "strategies" / strategy.slug
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        sidecar_path = sidecar_dir / f"paper_analyst_{ar.id}.json"
        sidecar_path.write_text(json.dumps(out.model_dump(), indent=2))

        meta = dict(strategy.metadata_json or {})
        if isinstance(out, PromotePaperToLive):
            meta["paper_analyst_promote_recommended"] = True
        elif isinstance(out, AbandonPaper):
            meta["paper_analyst_abandon_recommended"] = True
            meta["paper_analyst_post_mortem_path"] = out.post_mortem_path
        strategy.metadata_json = meta

        ar.status = AgentRunStatus.DONE.value
        ar.output_artifact_path = str(sidecar_path)
        ar.ended_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(ar)
        return ar
    except Exception as exc:
        ar.status = AgentRunStatus.FAILED.value
        ar.error = str(exc)
        ar.ended_at = datetime.now(timezone.utc)
        await session.commit()
        raise
```

**Note on sidecar dir:** keep it at `data/strategies/<slug>/` (agents repo root) — that's where existing analyst sidecars live (see `recommendations.py` for the pattern). Confirm via a quick grep in step 5c.

**Step 5e — Commit**

```bash
git add src/fwbg_agents/orchestrator/paper_flow.py tests/orchestrator/test_paper_flow.py
git commit -m "feat(M6b): paper_analyze orchestrator flow — Analyst → sidecar + metadata flag, no state transition"
```

---

## Task 6 — `POST /strategies/{id}/paper-analyze` endpoint + tests

**Files:**
- Modify: `src/fwbg_agents/api/strategies.py` — add endpoint + Pydantic response model.
- Create: `tests/api/test_strategies_paper_analyze.py`

**Behaviour:**
- Body: empty (or `{}`).
- 422 if strategy not in `PAPER_TRADING` (reuse `_require_state(strategy, [PAPER_TRADING])` helper — or inline check, smaller cost).
- 422 if `read_paper_summary` returns None (no on-disk data yet).
- 202 + `{agent_run_id: int, status: "scheduled"}` + `BackgroundTasks` runs `paper_flow.paper_analyze(strategy_id, fresh_session)`. (matches M5c reiterate-with-plugin envelope; honest about AR being PENDING at HTTP-return time)

**Step 6a — Tests (3):**

```python
# tests/api/test_strategies_paper_analyze.py
# 1. 422 when strategy in PROPOSED state
# 2. 422 when no on-disk paper data
# 3. 202 + AgentRun row created in RUNNING (eventually DONE after background completes — but for the API contract test, assert just 202 + AgentRun exists, mirror plugin_author endpoint test pattern from tests/api/test_plugins_author_flow.py)
```

**Step 6b → 6d — Red, Implement, Green.**

```python
# api/strategies.py append:
from fastapi import BackgroundTasks
from fwbg_agents.orchestrator.paper_flow import paper_analyze, PaperFlowError
from fwbg_agents.tools.fwbg_paper_reader import read_paper_summary

class PaperAnalyzeResponse(BaseModel):
    agent_run_id: int
    status: str

@router.post("/strategies/{strategy_id}/paper-analyze",
             response_model=PaperAnalyzeResponse, status_code=202)
async def post_paper_analyze(strategy_id: int, bg: BackgroundTasks,
                              session: AsyncSession = Depends(get_session)):
    s = (await session.execute(select(Strategy).where(Strategy.id == strategy_id))).scalar_one_or_none()
    if s is None:
        raise HTTPException(404, "strategy not found")
    if s.current_state != StrategyState.PAPER_TRADING.value:
        raise HTTPException(422, f"strategy must be in PAPER_TRADING (is {s.current_state})")
    if read_paper_summary(s.slug, Path(get_settings().fwbg_data_dir)) is None:
        raise HTTPException(422, "no on-disk paper-trading data yet")

    now = datetime.now(timezone.utc)
    ar = AgentRun(agent_name="paper_analyst", status=AgentRunStatus.PENDING.value,
                  strategy_id=s.id, started_at=now, created_at=now)
    session.add(ar)
    await session.commit()
    await session.refresh(ar)
    ar_id = ar.id

    async def _kick():
        async with get_session_context() as fresh:  # use existing helper, or write a one-shot
            try:
                await paper_analyze(s.id, fresh)
            except PaperFlowError:
                pass  # AgentRun already marked FAILED inside paper_analyze on exception
    bg.add_task(_kick)
    return PaperAnalyzeResponse(agent_run_id=ar_id, status="scheduled")
```

**Note:** verify the existing async-session-context helper name by reading `api/plugins.py:208`+ pattern. Adapt if it's spelled differently.

**Step 6e — Commit**

```bash
git add src/fwbg_agents/api/strategies.py tests/api/test_strategies_paper_analyze.py
git commit -m "feat(M6b): POST /strategies/{id}/paper-analyze — manual analyst trigger"
```

---

## Task 7 — `POST /strategies/{id}/promote-live` endpoint + tests

**Files:**
- Modify: `src/fwbg_agents/api/strategies.py` — add endpoint + body model.
- Create: `tests/api/test_strategies_promote_live.py`

**Body:** `{human_approval: bool, operator_note: str | None}`.

**Gates (all three must pass):**
1. `body.human_approval is True` — else 422.
2. `strategy.metadata_json.get("paper_analyst_promote_recommended") is True` — else 422.
3. `strategy.current_state == PAPER_TRADING` — else 422.

**On success:**
- Call `transition_strategy(session, s, StrategyState.LIVE_TRADING, payload={"human_approval": True, "operator_note": body.operator_note}, created_by="operator")` — M2 guard re-checks `human_approval` (third gate).
- Create audit `AgentRun(agent_name="promote_live", status=DONE, strategy_id=s.id)`.
- 200 + `{strategy_id, new_state: "live_trading", agent_run_id}`.

**Step 7a — Tests (5):**

```python
# tests/api/test_strategies_promote_live.py
# 1. 422 when body.human_approval=False
# 2. 422 when strategy in PROPOSED (wrong state)
# 3. 422 when metadata_json has no paper_analyst_promote_recommended flag
# 4. 200 happy path → strategy.current_state == LIVE_TRADING + AgentRun(agent_name="promote_live", status=DONE)
# 5. transition is audited in transition table (entity_type="strategy", from_state="paper_trading", to_state="live_trading", payload.human_approval=True)
```

**Step 7b → 7d — Red, Implement, Green.**

**Step 7e — Commit**

```bash
git add src/fwbg_agents/api/strategies.py tests/api/test_strategies_promote_live.py
git commit -m "feat(M6b): POST /strategies/{id}/promote-live — triple-gated human approval to LIVE_TRADING"
```

---

## Task 8 — `scripts/m6b_smoke.py` end-to-end smoke

**Files:**
- Create: `scripts/m6b_smoke.py`
- (No automated test for the script — manual smoke, mirrors `m6a_smoke.py`.)

**Stages:**
0. Cleanup: remove any prior `paper-smoke-test-001` Strategy + data/strategies/<slug>/ sidecars + data/paper-trades/<slug>/ files.
1. Seed: insert a Strategy in `PAPER_TRADING`, asset_class="forex", paper_phase_target_days=90.
2. Synthesise on-disk data at `<fwbg_data_dir>/account-trades/<slug>/`:
   - `trades.jsonl` with 50 entries, mix of winners/losers, win_rate≈0.55.
   - `status.json` with `current_equity=11000`, `starting_equity=10000`, sharpe≈1.2, max_dd≈0.10, days_in_paper=45.
   - `positions.json` with 1 open position (symbol=EURUSD, qty=1000, SL/TP set).
3. `POST /strategies/{id}/paper-analyze` → assert 202, capture `agent_run_id`.
4. Poll `GET /agents/runs/{ar_id}` until `status in {DONE, FAILED}` (≤30s).
5. Assert `data/strategies/<slug>/paper_analyst_<ar_id>.json` exists, parse it, assert `decision == "promote_paper_to_live"` (data was tuned to pass forex criteria).
6. Refresh Strategy from DB, assert `metadata_json["paper_analyst_promote_recommended"] is True`.
7. `POST /strategies/{id}/promote-live` with `{"human_approval": true, "operator_note": "m6b smoke"}` → assert 200, response contains `new_state: "live_trading"`.
8. Refresh Strategy, assert `current_state == "live_trading"`.
9. Idempotency-flag for re-runs: stage 0 must clean up the LIVE_TRADING strategy too (it transitions from paper, so the row is still there).
10. Print `[m6b_smoke] PASSED`.

**Run manually:**

```bash
cd ~/Projekte/fwbg-agents
VIRTUAL_ENV= uv run python scripts/m6b_smoke.py
```
Expected (final line): `[m6b_smoke] PASSED`.

**Step 8 — Commit**

```bash
git add scripts/m6b_smoke.py
git commit -m "feat(M6b): scripts/m6b_smoke.py — end-to-end paper-analyst + promote-live smoke"
```

---

## End-of-session housekeeping

1. **Design-doc Implementation-Status-Table:** add an M6b row with final commit SHAs + `✓ done`.
2. **This plan-doc:** flip status header to `✓ done` with final commit table.
3. **`project_fwbg_agents.md`:** append M6b block in the same shape as the M6a block (top of file).
4. **`reference_fwbg_agents_m6b_plan.md`:** new memory file with commit-table + locked decisions I-N.
5. **`MEMORY.md`:** add index line for the new reference memory.
6. **graphify:** commit-hook updates it automatically.
7. **Manual smoke against dev DB:** run `scripts/m6b_smoke.py` end-to-end twice — must end `[m6b_smoke] PASSED` both times (idempotent).

---

## Out-of-scope (next milestones)

- **M5d** — PluginAuthor planner/implementer split (parked, sketch in `project_fwbg_agents_m5d_sketch.md`).
- **M7** — Live-trading risk gates (position-size caps, daily-loss kill-switch, max-correlated-exposure). Likely triggers the eventual extraction of a shared `criteria_evaluator.py` module covering backtest + paper + live (3rd use case justifies it).
- **Dashboard wiring** for `POST /paper-analyze` button + `POST /promote-live` confirmation modal — fwbg-dashboard repo, separate session.
