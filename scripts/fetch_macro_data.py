"""
Fetch macro data for FWBG optimizer.

Sources:
- yfinance: DXY, VIX (hourly, ~2y history)
- FRED: International bond yields (monthly, 25+ years history)
  Forward-filled to daily for DataSource compatibility.
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
    """Download monthly yield from FRED and forward-fill to daily."""
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

        # Resample monthly → daily via forward-fill
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
