"""
Fetches CFTC Commitment of Traders (COT) data for FX futures.

Downloads the "Traders in Financial Futures" (TFF) report from CFTC
and extracts non-commercial positioning for major FX pairs.

Output: data/forexsb/COT_{SYMBOL}_DAY.csv per symbol.  Each CSV contains
one row per weekly report with a `release_date` column declaring when the
row became public (Friday ~21:00 UTC, see ``RELEASE_LAG`` below).
Downstream consumers must asof-join on ``release_date`` rather than the
report date — see ``src/fwbg/data/macro_contract.py``.

Source: https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm
URL pattern (2017+): https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip
Archive (2006-2016): https://www.cftc.gov/files/dea/history/fin_fut_txt_2006_2016.zip
"""
import io
import os
import zipfile

import pandas as pd
import requests

DATA_PATH = "./data/forexsb"
os.makedirs(DATA_PATH, exist_ok=True)

CFTC_BASE = "https://www.cftc.gov/files/dea/history"

# Release-date convention for the TFF report.
# CFTC publishes Tuesday's report Friday ~15:30 ET.  15:30 EDT = 19:30 UTC,
# 15:30 EST = 20:30 UTC.  Round up to 21:00 UTC so we never declare data
# public before it actually is.  Documented in
# docs/data/macro-release-dates.md.
RELEASE_LAG = pd.Timedelta(days=3, hours=21)

# Map CFTC contract names to FX symbols
COT_FX_MAPPING = {
    "EURO FX": "EURUSD",
    "JAPANESE YEN": "USDJPY",
    "BRITISH POUND": "GBPUSD",
    "CANADIAN DOLLAR": "USDCAD",
    "AUSTRALIAN DOLLAR": "AUDUSD",
    "SWISS FRANC": "USDCHF",
    "NEW ZEALAND DOLLAR": "NZDUSD",
}


def _build_urls(start_year=2006, end_year=None):
    """Build list of CFTC download URLs."""
    if end_year is None:
        end_year = pd.Timestamp.now().year

    urls = []
    # Archive: 2006-2016 combined
    if start_year <= 2016:
        urls.append(f"{CFTC_BASE}/fin_fut_txt_2006_2016.zip")

    # Individual years: 2017+
    for year in range(max(start_year, 2017), end_year + 1):
        urls.append(f"{CFTC_BASE}/fut_fin_txt_{year}.zip")

    return urls


def _download_zip(url):
    """Download and parse a CFTC TFF ZIP file."""
    print(f"  {url.split('/')[-1]}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)

    return df


def compute_release_date(report_date: pd.Timestamp) -> pd.Timestamp:
    """Return the UTC instant a CFTC TFF report becomes public.

    The report dated ``report_date`` (a Tuesday) is released the following
    Friday at ~15:30 ET; we conservatively use Friday 21:00 UTC.  Holiday
    delays (e.g. Thanksgiving) are not modelled — see
    ``docs/data/macro-release-dates.md``.
    """
    ts = pd.Timestamp(report_date)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    # Normalise to start-of-day before adding the lag so the result lands
    # at exactly Fri 21:00 UTC regardless of the input's time component.
    return ts.normalize() + RELEASE_LAG


def _extract_fx_positions(df):
    """Extract asset manager net positioning for FX contracts.

    TFF reports use Asset_Mgr (institutional) instead of NonComm.
    Net position = Asset_Mgr_Long - Asset_Mgr_Short (smart money flow).

    Each returned row carries a ``release_date`` column (UTC) declaring when
    the row became public.  Consumers must merge on ``release_date``.
    """
    # TFF uses Report_Date_as_YYYY-MM-DD
    date_col = "Report_Date_as_YYYY-MM-DD"
    if date_col not in df.columns:
        # Fallback for older format
        date_col = "Report_Date_as_MM_DD_YYYY"

    results = {}

    for contract_name, symbol in COT_FX_MAPPING.items():
        mask = df["Market_and_Exchange_Names"].str.contains(
            contract_name, case=False, na=False
        )
        subset = df[mask].copy()
        if subset.empty:
            continue

        subset["date"] = pd.to_datetime(subset[date_col], format="mixed")
        subset = subset.sort_values("date")

        # Asset Manager positioning (institutional smart money)
        am_long = subset["Asset_Mgr_Positions_Long_All"].astype(float)
        am_short = subset["Asset_Mgr_Positions_Short_All"].astype(float)

        out = pd.DataFrame({
            "Datetime": subset["date"].values,
            "Close": (am_long - am_short).values,
            "release_date": [compute_release_date(d) for d in subset["date"].values],
        })
        out = out.set_index("Datetime")

        if symbol in results:
            results[symbol] = pd.concat([results[symbol], out])
        else:
            results[symbol] = out

    return results


def fetch_cot_data(start_year=2006):
    """Download CFTC TFF reports and save per-symbol CSVs."""
    print("=== COT Data Download ===")
    urls = _build_urls(start_year)
    all_data = []

    for url in urls:
        try:
            df = _download_zip(url)
            all_data.append(df)
            print(f"    {len(df)} rows")
        except Exception as e:
            print(f"    Error: {e}")

    if not all_data:
        print("No COT data downloaded")
        return

    combined = pd.concat(all_data, ignore_index=True)
    results = _extract_fx_positions(combined)

    print(f"\n=== Saving {len(results)} symbols ===")
    for symbol, df in sorted(results.items()):
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        # NOTE: weekly rows are kept verbatim — downstream asof-joins on
        # release_date.  Forward-filling to daily would re-introduce the
        # bias this change set out to fix.

        filename = f"{DATA_PATH}/COT_{symbol}_DAY.csv"
        df.to_csv(filename)
        print(f"  {symbol}: {len(df)} weekly rows ({df.index[0].date()} to {df.index[-1].date()})")


if __name__ == "__main__":
    fetch_cot_data()
