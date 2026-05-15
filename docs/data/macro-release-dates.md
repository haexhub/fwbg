# Macro / COT Release-Date Convention

External datasets (COT, FRED, etc.) describe a *report period* but become
public on a later *release date*.  Joining features on the report date
leaks future information into backtests.  This document captures the
convention we now enforce and the bias sources that are still open.

## Contract

Every macro DataFrame that flows through the data-loading pipeline must
carry a `release_date` column.  The merge layer
([`src/fwbg/data/macro_contract.py`](../../src/fwbg/data/macro_contract.py))
asof-joins each row onto bars whose timestamp is `>= release_date`.

Bars whose timestamp predates the first release see `NaN` for the
corresponding feature.

## What is fixed by the current implementation

| Source | Fetcher | Convention |
|--------|--------|-----------|
| CFTC TFF (COT, weekly) | `scripts/fetch_cot_data.py` | `release_date = report_date + Fri 21:00 UTC` (Tue→Fri lag, 21:00 UTC chosen as the DST-safe upper bound of 15:30 ET).  Holiday delays (e.g. Thanksgiving) are not modelled — see "Known limitations" below. |

CFTC reports re-fetched with the new script carry the column natively.
CSVs predating this change do not — they must be re-fetched or marked
stale.  The merge function raises `ValueError` on `NaT`, missing column,
or non-datetime dtype, so silent regressions are rejected.

## Known limitations / out of scope

These bias sources are documented but **not** fixed by this change set.
They warrant follow-up plans of their own.

### FRED monthly OECD long-term yields (`DE10Y`, `JP10Y`, `GB10Y`, `AU10Y`)

OECD long-term interest rates are published with a 4–8 week lag.  The
current fetcher (`scripts/fetch_macro_data.py`) indexes by FRED's
`observation_date`, which is the first day of the *reference month*, then
forward-fills to daily.  That leaks information by up to two months.

Fixing this requires either (a) the OECD release calendar or (b) a
conservative fixed buffer (e.g. `report_month_end + 8 weeks`).  Both need
their own design decision; not bundled here.

### FRED daily Treasury yields (`DGS2`, `DGS5`, `DGS30`)

FRED publishes daily Treasury rates roughly one business day after the
observation date.  For daily-bar strategies this is masked by the
1-calendar-day shift in `loader.py`; for HOUR-resolution strategies the
intraday window between observation close and FRED release is still a
forward-leak source.  Negligible in most backtests, but not zero.

### Generic `DataSource.load()` path

`src/fwbg/data/loader.py` exposes the release-date contract only when a
source CSV carries the `release_date` column.  Sources that do not
(everything except COT today) keep the legacy 1-calendar-day shift via
`prev_date.map()`.  A future task is to extend the `DataSource` interface
to surface release dates uniformly so the legacy fallback can be retired.

### Bar-clock vs. release-clock skew

Bars are stored tz-naive in Europe/Berlin local time
(`fetch_all_sources.py:30-42`).  Release dates are tz-naive UTC after
normalization in the loader.  Numerical comparison therefore leaks up to
~1 hour around the release moment.  For weekly COT data with a 3-day lag,
that 1-hour skew is immaterial; for any future sub-daily release calendar
we should pin a common tz convention.

### `cot_positioning` plugin's defensive `.shift(1)`

`packages/fwbg-premium/.../cot_positioning/__init__.py:67-71` shifts every
derived COT feature by one bar.  With the upstream release-date fix this
shift is now redundant, but harmless (1-hour additional delay).  Removing
it is a follow-up clean-up; leaving it in is the safer default while the
contract beds in.

## Re-fetching historical data

After upgrading, re-run:

```bash
python scripts/fetch_cot_data.py
```

This rewrites every `data/forexsb/COT_*_DAY.csv` with the new schema
(weekly rows, `release_date` column).  Strategy artefacts cached against
the old daily-ffill schema must be re-trained.
