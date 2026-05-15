# COT & Macro Fetch Scripts — Release-Date Audit

> Audit produced for plan [2026-05-15-cot-macro-release-lag.md](2026-05-15-cot-macro-release-lag.md), Task 1.

## Scope

Every fetch script under `scripts/` plus the downstream merge in `src/fwbg/data/loader.py` was inspected to determine whether the timestamp index reflects the **release date** (when data became public) or the **report date** (the period the data covers).

## Per-script findings

### `scripts/fetch_cot_data.py` — CFTC TFF (weekly)

- **What it fetches:** Asset-Manager net positions for 7 FX futures (EURUSD, USDJPY, GBPUSD, USDCAD, AUDUSD, USDCHF, NZDUSD). Source: CFTC TFF historical zips.
- **Index used:** `Report_Date_as_YYYY-MM-DD` (column from the CFTC CSV) → renamed to `Datetime`. This is the **report date** (Tuesday of the reporting week).
- **Release-aware?** ❌ **NO.** Lines 134–137 sort by report date, deduplicate, then `df.resample("D").ffill()`. Result: a Tuesday-dated row is forward-filled to Wed/Thu/Fri — but CFTC only releases the report **Friday ~15:30 ET**. Tue/Wed/Thu bars see the report 3 days early.
- **Severity:** HIGH. Direct lookahead, ~3 calendar days, affects 7 FX symbols.

### `scripts/fetch_macro_data.py` — yfinance hourly (DXY, VIX)

- **What it fetches:** DXY (`DX-Y.NYB`) and VIX (`^VIX`) at 1h interval, ~2 years history.
- **Index used:** yfinance bar timestamp (hour close).
- **Release-aware?** ✅ Effectively yes — hourly market data is available at the bar close. No release lag concern.

### `scripts/fetch_macro_data.py` — FRED daily Treasury yields (DGS2, DGS5, DGS30)

- **What it fetches:** US Treasury constant-maturity rates (daily series).
- **Index used:** FRED `observation_date` (the trading day the rate was *observed*, e.g., 2024-01-02).
- **Release-aware?** ⚠️ **PARTIAL.** FRED publishes the daily rate the next business day. For a 2024-01-02 observation, the value is on FRED roughly 2024-01-03 ~16:00 ET. The current `prev_date` shift in `loader.py:216-229` masks this for daily bars but not for HOUR-resolution backtests within day D.
- **Severity:** MEDIUM for HOUR-resolution, LOW for DAY-resolution.

### `scripts/fetch_macro_data.py` — FRED monthly OECD yields (DE10Y, JP10Y, GB10Y, AU10Y)

- **What it fetches:** OECD long-term interest rates (monthly, `IRLTLT01xxxM156N` series).
- **Index used:** FRED `observation_date` = first day of the reference month.
- **Release-aware?** ❌ **NO.** OECD monthly stats are released with a 4–8 week lag after the reference period ends. The script then `resample("D").ffill()`, so daily bars on, e.g., 2024-02-15 see the February rate even though OECD published it in March/April.
- **Severity:** HIGH for any strategy keying off `macro_DE10Y`/`macro_JP10Y`/etc.

### `scripts/fetch_all_sources.py` — OHLCV via yfinance / stooq

- **What it fetches:** Asset OHLCV bars (Yahoo + Stooq).
- **Index used:** Market bar timestamp.
- **Release-aware?** ✅ Yes — bar timestamp **is** the moment the bar closes (and hence becomes public).

### `scripts/fetch_mt5_source.py` — MT5 broker data

- Not inspected in detail; this script fetches broker bars and is in the same category as `fetch_all_sources.py` (bar timestamps are release-aware).

## Downstream merge audit

### `src/fwbg/data/loader.py:213-229` — generic macro/source merge

```python
prev_date = pd.Series(date_series, index=df.index).map(
    lambda d: pd.Timestamp(d) - pd.Timedelta(days=1)
)
for prefix, raw_df in result.data.items():
    if "Close" in raw_df.columns:
        lookup = raw_df["Close"].to_dict()
        df[f"macro_{prefix}"] = prev_date.map(
            lambda d, lk=lookup: lk.get(d, np.nan)
        ).ffill()
```

- A **fixed 1-calendar-day shift**. Adequate for daily series that publish next-day, **inadequate** for weekly COT (3-day lag) and monthly FRED yields (4–8 week lag).
- The `DataSource` interface this code consumes returns DataFrames with a `Close` column keyed by timestamp; there is no `release_date` channel today. Adopting the contract requires either extending `DataSource` or routing macro/COT through a separate code path.

### `packages/fwbg-premium/.../data_loading/cot_positioning/__init__.py:67-71`

```python
# Shift all COT features by 1 bar (lookahead prevention)
for col in cot_feature_cols:
    df[col] = df[col].shift(1)
```

- A **1-bar shift** (1 hour for H1 bars). Completely inadequate vs. the 3-day Tue→Fri release lag.
- Correct upstream fix: have the base `macro_cot_*` column already be release-date-correct (via the new contract). Then the `.shift(1)` here can stay as defense-in-depth or be removed.

## Recommended scope for the rest of the plan

| Item | Action |
|------|--------|
| `fetch_cot_data.py` | Emit `release_date = report_date + Fri 21:00 UTC`, drop the daily ffill. **Covered by Task 4.** |
| `fetch_macro_data.py` FRED daily | Emit `release_date = observation_date + 1 BDay`. **Add to Task 4 (suggested).** |
| `fetch_macro_data.py` FRED monthly OECD | Hard to pin a single release date; OECD does publish a calendar but it varies. **Out of scope** — flag in docs (Task 6). |
| `loader.py:216-229` | Migrate to the contract for COT/macro_cot prefix specifically; leave generic daily-prev_date path until `DataSource` exposes `release_date`. **Covered by Task 5, possibly narrower than plan implied.** |
| `cot_positioning` plugin | After Task 5 the upstream column is release-correct; `.shift(1)` can stay as redundant guard or be removed in a follow-up. **Out of scope for this plan.** |

## Sample CFTC release-calendar dates to verify against in Task 4

| Report date (Tue) | Expected release (Fri) |
|------------------|------------------------|
| 2024-01-02 | 2024-01-05 |
| 2024-02-06 | 2024-02-09 |
| 2024-06-25 | 2024-06-28 |
| 2024-11-26 | 2024-11-29 |
| 2025-03-04 | 2025-03-07 |

(Source: <https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm>.)
