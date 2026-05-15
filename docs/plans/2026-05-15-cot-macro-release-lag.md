# COT & Macro Release-Lag Validation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure COT (Commitment of Traders) and macro-economic data are joined to OHLCV bars using the *release date* (when the data became public), not the *report date* (the period the data covers), to eliminate lookahead bias in macro features.

**Architecture:** Each external dataset gets a mandatory `release_date` column (or `report_date + lag` rule) recorded at fetch time. The loader joins macro rows to OHLCV bars using `bar_timestamp >= release_date`, never `bar_timestamp >= report_date`. A unit-tested guard rejects any merge that violates this constraint.

**Tech Stack:** Python, pandas, pytest. Data sources: CFTC COT (Tuesday report, Friday release), various macro from FRED / Stooq / scripts under `scripts/`.

---

## Background

COT data publishes with a ~3-day lag: the report dated Tuesday is released Friday after 15:30 ET. The current implementation in `scripts/fetch_cot_data.py` resamples weekly COT data to daily via `ffill()`, which is correct *iff* the index already represents release dates. A scan of the fetch and merge code shows this is not enforced anywhere; the report date is being used as the index in some paths, which means COT values are joined to bars that occurred before the data was public.

Similar risk applies to FRED macro series (NFP, CPI, GDP) — each has its own release schedule. See [src/fwbg/data/loader.py:216-229](../../src/fwbg/data/loader.py#L216) where `prev_date.map()` is used for the macro merge: this is *partially* protective (uses previous day) but it's a calendar-day shift, not a true release-time shift.

Risk: macro features may carry information that a live trader could not have had, inflating in-sample and walk-forward metrics.

---

## Success Criteria

1. Every external dataset stored under `data/` carries an explicit `release_date` column.
2. The merge function refuses to attach a macro row to a bar whose timestamp is `< release_date`.
3. Unit test verifies "bar at T cannot see COT row with release_date > T".
4. Re-fetched COT data is verified manually against the CFTC release calendar for at least 5 sample dates.
5. Existing tests pass; expect a small metric delta in macro-using strategies (that's the bias being removed).

---

## Out of Scope

- Adding new macro data sources.
- Re-fetching historical COT going back years (only forward-going correctness is required; document the cutoff date).
- Order-of-magnitude restructuring of macro data storage.

---

## Task 1: Audit which macro datasets are affected

**Step 1: Inventory**

```bash
ls scripts/fetch_*.py
grep -rn "release\|report\|publication" scripts/fetch_*.py src/fwbg/data/loader.py
```

**Step 2: Document findings**

Create `docs/plans/2026-05-15-cot-macro-audit.md` (or just inline in this plan) listing for each script:
- What it fetches.
- What timestamp it uses as index.
- Whether that timestamp is report-date or release-date.

**Step 3: Commit the audit**

```bash
git add docs/plans/2026-05-15-cot-macro-audit.md
git commit -m "docs: audit macro fetch scripts for release-date handling"
```

---

## Task 2: Define the release-date contract

**Files:**
- Create: `src/fwbg/data/macro_contract.py`

**Step 1: Define the contract module**

```python
"""Release-date contract for external macro / COT data.

Any DataFrame stored under data/<source>/ that participates in feature merges
MUST include a `release_date` column (UTC tz-aware Timestamp). The merger
will refuse to attach a row whose `release_date` is greater than the target
bar's timestamp.
"""
from __future__ import annotations

import pandas as pd

RELEASE_COL = "release_date"


def assert_release_dates_present(df: pd.DataFrame, source_name: str) -> None:
    if RELEASE_COL not in df.columns:
        raise ValueError(
            f"{source_name}: missing required column '{RELEASE_COL}'. "
            f"Every macro dataset must declare when its rows became public."
        )
    if df[RELEASE_COL].isna().any():
        raise ValueError(f"{source_name}: '{RELEASE_COL}' contains NaT.")
    if not pd.api.types.is_datetime64_any_dtype(df[RELEASE_COL]):
        raise ValueError(f"{source_name}: '{RELEASE_COL}' must be datetime.")


def merge_respect_release(
    bars: pd.DataFrame,
    macro: pd.DataFrame,
    source_name: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """Asof-join `macro` onto `bars` respecting release_date.

    For each bar at time t, attach the macro row with the largest
    release_date <= t. Returns `bars` with `value_cols` added (NaN before
    first release).
    """
    assert_release_dates_present(macro, source_name)
    macro_sorted = macro.sort_values(RELEASE_COL)
    merged = pd.merge_asof(
        bars.sort_index().reset_index(),
        macro_sorted[[RELEASE_COL] + value_cols],
        left_on=bars.index.name or "T",
        right_on=RELEASE_COL,
        direction="backward",
    )
    return merged.set_index(bars.index.name or "T")
```

**Step 2: Commit**

```bash
git add src/fwbg/data/macro_contract.py
git commit -m "feat: introduce release-date contract for macro data"
```

---

## Task 3: Write the contract test

**Files:**
- Test: `tests/data/test_macro_release_contract.py` (create)

**Step 1: Write the test**

```python
import pandas as pd
import pytest

from fwbg.data.macro_contract import merge_respect_release


def test_bar_at_T_cannot_see_release_after_T():
    bars = pd.DataFrame(
        {"C": [1.0, 1.0, 1.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="D", name="T"),
    )
    macro = pd.DataFrame({
        "release_date": pd.to_datetime(["2024-01-02", "2024-01-05"]),
        "value": [100.0, 200.0],
    })
    merged = merge_respect_release(bars, macro, "test", ["value"])
    # Bar 2024-01-01: no release available yet → NaN
    # Bar 2024-01-02: release on same day → 100
    # Bar 2024-01-03: still only the first release → 100
    assert pd.isna(merged.loc["2024-01-01", "value"])
    assert merged.loc["2024-01-02", "value"] == 100.0
    assert merged.loc["2024-01-03", "value"] == 100.0


def test_missing_release_date_column_raises():
    bars = pd.DataFrame({"C": [1.0]}, index=pd.date_range("2024-01-01", periods=1, freq="D"))
    bad_macro = pd.DataFrame({"value": [1.0]}, index=[pd.Timestamp("2024-01-01")])
    with pytest.raises(ValueError, match="release_date"):
        merge_respect_release(bars, bad_macro, "bad", ["value"])
```

**Step 2: Run, expect PASS (contract module exists from Task 2)**

```bash
pytest tests/data/test_macro_release_contract.py -v
```

**Step 3: Commit**

```bash
git add tests/data/test_macro_release_contract.py
git commit -m "test: cover macro release-date contract"
```

---

## Task 4: Update COT fetcher to emit release_date

**Files:**
- Modify: `scripts/fetch_cot_data.py`

**Step 1: Identify report-date vs release-date in current code**

The CFTC publishes Tuesday's report Friday afternoon ET. Convention: release_date = report_date + 3 days at 15:30 ET → store as next-day UTC (16:00 UTC ≈ end of NY session).

**Step 2: Add release_date column**

```python
# After parsing CFTC CSV (report_date already exists):
df["release_date"] = (
    df["report_date"]
    + pd.Timedelta(days=3)  # Tue → Fri
    + pd.Timedelta(hours=21)  # 15:30 ET ≈ 20:30 UTC during DST; round to 21:00 UTC
)
df["release_date"] = df["release_date"].dt.tz_localize("UTC")
```

**Step 3: Drop the `ffill()` resample to daily — that pattern is what masked the bug**

Replace with: keep the weekly rows, let `merge_respect_release` do asof-join.

**Step 4: Run fetch on a small range manually and inspect output**

```bash
python scripts/fetch_cot_data.py --start 2024-01-01 --end 2024-02-01 --dry-run
```
(Add `--dry-run` flag if not present, or just inspect the resulting CSV.)

**Step 5: Manually verify 3 sample release dates against CFTC's published calendar at https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm**

Document the verification result in commit message.

**Step 6: Commit**

```bash
git add scripts/fetch_cot_data.py
git commit -m "feat: emit release_date in COT fetch (verified vs CFTC calendar)"
```

---

## Task 5: Migrate macro loader to use the contract

**Files:**
- Modify: `src/fwbg/data/loader.py` (the macro/COT join block around line 216-229)

**Step 1: Locate the current join**

```bash
grep -n "prev_date.map\|cot\|macro" src/fwbg/data/loader.py
```

**Step 2: Replace with `merge_respect_release`**

```python
# Before (illustrative):
df[macro_col] = df.index.map(macro_series.shift(1).to_dict())

# After:
from fwbg.data.macro_contract import merge_respect_release
df = merge_respect_release(df, macro_df, "cot", [macro_col])
```

**Step 3: Run all data-loading tests**

```bash
pytest tests/test_data_loader.py tests/test_data_loading.py tests/test_datasource_prepare.py -x
```

**Step 4: Run end-to-end on a strategy that uses macro features**

```bash
fwbg --assets EURUSD --strategy-file strategies/configs/<a-macro-strategy>.json
```

**Step 5: Commit**

```bash
git add src/fwbg/data/loader.py
git commit -m "fix: join macro/COT via release_date asof-merge"
```

---

## Task 6: Document the cutoff and known limitations

**Files:**
- Modify: `README.md` or create `docs/data/macro-release-dates.md`

**Step 1: Write a brief note**

- COT release_date is set to Friday 21:00 UTC. For pre-DST or pre-2007 data this is approximate; document the convention.
- Historical data fetched before this change does NOT have release_date and must be re-fetched or marked stale.
- The contract refuses ambiguous data — `release_date` NaN raises.

**Step 2: Commit**

```bash
git add docs/data/macro-release-dates.md README.md
git commit -m "docs: document macro release-date convention and limitations"
```
