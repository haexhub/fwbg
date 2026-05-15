# FRED Monthly OECD Release-Lag — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Apply the release-date contract introduced in
[2026-05-15-cot-macro-release-lag.md](2026-05-15-cot-macro-release-lag.md)
to the FRED monthly OECD long-term yield series (`DE10Y`, `JP10Y`,
`GB10Y`, `AU10Y`).  Today these series leak information by 4–8 weeks; the
audit at [2026-05-15-cot-macro-audit.md](2026-05-15-cot-macro-audit.md)
and [docs/data/macro-release-dates.md](../data/macro-release-dates.md)
documents the bias.

**Architecture:** Re-use `src/fwbg/data/macro_contract.py` and the
`release_date`-aware merge path already wired into
`src/fwbg/data/loader.py`.  This plan only changes the OECD fetcher and
adds tests; no new infrastructure required.

**Tech Stack:** Python, pandas, pytest.  Data source: FRED
(`https://fred.stlouisfed.org/series/IRLTLT01<COUNTRY>M156N`), which
republishes OECD Main Economic Indicators (MEI).

---

## Background

The FRED OECD long-term interest rate series (`IRLTLT01DEM156N`,
`IRLTLT01JPM156N`, `IRLTLT01GBM156N`, `IRLTLT01AUM156N`) are sourced from
OECD MEI and publish on the OECD's monthly release schedule.  OECD does
not pre-announce a precise release calendar in a machine-readable form,
but historically the value for reference month *M* is published in the
second half of month *M+1* (~5–6 weeks after the start of the reference
month).  See: <https://www.oecd.org/sdd/oecdmaineconomicindicatorsmei.htm>.

Today `scripts/fetch_macro_data.py` indexes by FRED's
`observation_date` — the **first day** of the reference month — then
forward-fills to daily.  Bars on the 15th of month *M* therefore see the
month-*M* yield, even though OECD only published it ~6 weeks later.

This change set adopts a conservative fixed buffer rather than fitting a
release calendar:

> **release_date = observation_month_end + 6 weeks at 21:00 UTC**

That places visibility ~5–7 weeks after the reference period closes —
slightly later than typical OECD publication, but never earlier.  The
small staleness penalty is the price for not modelling holiday slips.

---

## Success Criteria

1. `scripts/fetch_macro_data.py` emits a `release_date` column for every
   OECD monthly series.
2. Daily FRED Treasury series (`DGS2`, `DGS5`, `DGS30`) and hourly
   yfinance tickers (`DXY`, `VIX`) remain unchanged — out of scope.
3. The contract test from `tests/data/test_macro_release_contract.py` is
   already in place; this plan adds a release-date computation test
   mirroring `tests/data/test_cot_release_date.py`.
4. An end-to-end test in `tests/test_data_loading.py` shows that a bar in
   the middle of reference month *M* sees `NaN` for the OECD column when
   only the month-*M* row exists.
5. `docs/data/macro-release-dates.md` is updated to mark OECD monthly as
   "fixed" and document the chosen buffer.

---

## Out of Scope

- A live OECD release-calendar feed (would require scraping or an API
  this codebase does not call yet).
- FRED daily Treasury (`DGS*`) release-time correction — separate
  follow-up.
- Extending the generic `DataSource` interface to surface release dates
  uniformly — separate follow-up.

---

## Task 1: Add the release-date computation helper

**Files:**
- Modify: `scripts/fetch_macro_data.py`

**Step 1: Add a top-level helper mirroring `compute_release_date` in `fetch_cot_data.py`**

```python
# Buffer pinned in docs/data/macro-release-dates.md
OECD_RELEASE_BUFFER = pd.Timedelta(weeks=6, hours=21)


def compute_oecd_release_date(observation_date: pd.Timestamp) -> pd.Timestamp:
    """Conservative release date for OECD monthly long-term yields.

    Returns ``observation_month_end + 6 weeks at 21:00 UTC``.  Real OECD
    publication is usually 4–6 weeks after the reference month begins;
    the conservative buffer never declares data public before it is.
    """
    ts = pd.Timestamp(observation_date)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    month_end = ts.normalize() + pd.offsets.MonthEnd(0)
    return pd.Timestamp(month_end) + OECD_RELEASE_BUFFER
```

**Step 2: Commit**

```bash
git add scripts/fetch_macro_data.py
git commit -m "feat: add OECD monthly release-date helper"
```

---

## Task 2: Pin the helper against sample dates

**Files:**
- Create: `tests/data/test_fred_monthly_release_date.py`

**Step 1: Write the test**

```python
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_macro_data import compute_oecd_release_date  # noqa: E402


@pytest.mark.parametrize(
    "observation, expected_release",
    [
        # January 2024 month-end (Jan 31) + 6 weeks = March 13 21:00 UTC.
        ("2024-01-01", "2024-03-13 21:00:00+00:00"),
        ("2024-02-01", "2024-04-10 21:00:00+00:00"),
        # June 2024 → end of June (Jun 30) + 6w = Aug 11 21:00 UTC.
        ("2024-06-01", "2024-08-11 21:00:00+00:00"),
    ],
)
def test_oecd_release_date_matches_convention(observation, expected_release):
    assert compute_oecd_release_date(pd.Timestamp(observation)) == pd.Timestamp(
        expected_release
    )


def test_release_lands_after_reference_month():
    """No matter the input day, release_date is strictly after the
    reference month ends."""
    for obs in ["2024-01-15", "2024-07-31", "2025-02-28"]:
        ts = pd.Timestamp(obs)
        release = compute_oecd_release_date(ts)
        month_end = ts.normalize() + pd.offsets.MonthEnd(0)
        assert release > pd.Timestamp(month_end).tz_localize("UTC")
```

**Step 2: Run, expect PASS**

```bash
PYTHONPATH="$PWD/src:$PWD/packages/fwbg-premium/src" \
  pytest tests/data/test_fred_monthly_release_date.py -v
```

**Step 3: Commit**

```bash
git add tests/data/test_fred_monthly_release_date.py
git commit -m "test: pin OECD monthly release-date convention"
```

---

## Task 3: Emit `release_date` in the monthly fetcher path

**Files:**
- Modify: `scripts/fetch_macro_data.py`

**Step 1: Distinguish the monthly OECD series from daily DGS**

The current `_download_fred_yield` treats all FRED series the same.
Branch on the series id: `IRLTLT01*` are monthly OECD; everything else is
daily.  Only the OECD path needs a `release_date` column.

**Step 2: Inject the column after the existing CSV parse**

```python
if "IRLTLT01" in series_id:
    df["release_date"] = df.index.to_series().apply(compute_oecd_release_date)
```

**Step 3: Stop the daily ffill for OECD series only**

The OECD-monthly path is the one that needs to drop `df.resample("D").ffill()` because the downstream `merge_respect_release` now handles
sparse rows correctly.  The DGS daily path keeps its current shape so this
plan touches only what it must.

**Step 4: Commit**

```bash
git add scripts/fetch_macro_data.py
git commit -m "fix: emit release_date and stop ffill for OECD monthly yields"
```

---

## Task 4: End-to-end integration test

**Files:**
- Modify: `tests/test_data_loading.py`

**Step 1: Add a new test next to `test_release_date_column_blocks_pre_release_lookahead`**

The test should set up a CSV that mimics the post-fix OECD output (one
row per month with `release_date`) and verify a mid-month bar in the
reference month sees `NaN`, while a bar after the conservative buffer
sees the value.

**Step 2: Run, expect PASS (the loader path already handles release_date)**

**Step 3: Commit**

```bash
git add tests/test_data_loading.py
git commit -m "test: verify OECD monthly bar sees NaN before release buffer"
```

---

## Task 5: Update the limitations doc

**Files:**
- Modify: `docs/data/macro-release-dates.md`

**Step 1: Move "FRED monthly OECD long-term yields" from "Known
limitations" to "What is fixed"**

Record the buffer choice (`6 weeks + 21:00 UTC`), why it was picked
(conservative; never declares data public before OECD has published it),
and where the helper lives.

**Step 2: Commit**

```bash
git add docs/data/macro-release-dates.md
git commit -m "docs: mark OECD monthly release-lag as fixed"
```

---

## Verification

Re-fetch the affected series and confirm bars before the release buffer
see NaN end-to-end:

```bash
python scripts/fetch_macro_data.py
PYTHONPATH="$PWD/src:$PWD/packages/fwbg-premium/src" \
  pytest tests/data/ tests/test_data_loading.py -q
```

Strategy artefacts cached against the old daily-ffill OECD schema must be
re-trained — same caveat as the COT change.
