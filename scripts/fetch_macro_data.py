"""
Fetch macro data for FWBG optimizer.

Sources:
- yfinance: DXY, VIX (hourly, ~2y history)
- FRED: International bond yields (monthly, 25+ years history)
  OECD monthly series carry a ``release_date`` column (see
  ``compute_oecd_release_date``) and are NOT forward-filled; the loader
  asof-joins them on ``release_date``.  Daily Treasury series are
  forward-filled to daily as before.
"""
import io
import os
import urllib.request

import pandas as pd
import yfinance as yf

DATA_PATH = "./data/forexsb"
os.makedirs(DATA_PATH, exist_ok=True)

# Hourly macro tickers via yfinance
HOURLY_TICKERS = {
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
}

# Bond yields via FRED (daily/monthly, forward-filled to daily)
FRED_YIELD_SERIES = {
    # US Treasury yields (daily)
    "US2Y": "DGS2",             # US 2Y Treasury
    "US5Y": "DGS5",             # US 5Y Treasury
    "US30Y": "DGS30",           # US 30Y Treasury
    # International yields (monthly, forward-filled)
    "DE10Y": "IRLTLT01DEM156N",  # Germany 10Y
    "JP10Y": "IRLTLT01JPM156N",  # Japan 10Y
    "GB10Y": "IRLTLT01GBM156N",  # UK 10Y
    "AU10Y": "IRLTLT01AUM156N",  # Australia 10Y
}

# Conservative OECD publication buffer, pinned in docs/data/macro-release-dates.md.
# The audit puts the real lag at 4-8 weeks after the reference period ends;
# we use the upper bound so data is never declared public before it was.
OECD_RELEASE_BUFFER = pd.Timedelta(weeks=8, hours=21)


def compute_oecd_release_date(observation_date) -> pd.Timestamp:
    """Conservative release date for OECD monthly long-term yields.

    Returns ``observation_month_end + 8 weeks at 21:00 UTC``.  FRED stamps
    these series with the first day of the reference month, so the buffer
    is anchored on that month's end, not on the observation date.
    """
    ts = pd.Timestamp(observation_date)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    month_end = ts.normalize() + pd.offsets.MonthEnd(0)
    return month_end + OECD_RELEASE_BUFFER


def _is_oecd_monthly(series_id: str) -> bool:
    return series_id.startswith("IRLTLT01")


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=2000-01-01"


def _download_yfinance(name, ticker, interval, suffix):
    """Download a single ticker from yfinance and save as CSV."""
    try:
        print(f"  {name} ({ticker}, {interval})...")
        period = "2y" if interval == "1h" else "max"
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[["Close"]].copy()
            df.index.name = "Datetime"

            filename = f"{DATA_PATH}/{name}_{suffix}.csv"
            df.to_csv(filename)
            print(f"    {len(df)} rows saved")
        else:
            print(f"    No data")
    except Exception as e:
        print(f"    Error: {e}")


def _download_fred_yield(name, series_id):
    """Download a yield series from FRED (OECD monthly: sparse + release_date)."""
    try:
        print(f"  {name} ({series_id})...")
        url = FRED_CSV_URL.format(series_id=series_id)
        resp = urllib.request.urlopen(url, timeout=15)
        data = resp.read().decode()

        df = pd.read_csv(io.StringIO(data), parse_dates=["observation_date"],
                         index_col="observation_date")
        df.columns = ["Close"]

        # Drop missing values (FRED uses "." for missing)
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna()

        if _is_oecd_monthly(series_id):
            # Keep one row per month and declare when it became public.
            # Forward-filling by observation month would leak 4-8 weeks;
            # the loader asof-joins on release_date instead.
            df["release_date"] = [compute_oecd_release_date(d) for d in df.index]
        else:
            # Daily Treasury series: forward-fill weekends/holidays.
            df = df.resample("D").ffill()

        df.index.name = "Datetime"
        filename = f"{DATA_PATH}/{name}_DAY.csv"
        df.to_csv(filename)
        print(f"    {len(df)} daily rows ({df.index[0].date()} to {df.index[-1].date()})")
    except Exception as e:
        print(f"    Error: {e}")


def update_macro_data():
    """Fetch hourly macro data (yfinance) and daily yields (FRED)."""
    print("=== Hourly Data (yfinance) ===")
    for name, ticker in HOURLY_TICKERS.items():
        _download_yfinance(name, ticker, "1h", "HOUR")

    print("\n=== Bond Yields (FRED) ===")
    for name, series_id in FRED_YIELD_SERIES.items():
        _download_fred_yield(name, series_id)


if __name__ == "__main__":
    update_macro_data()
