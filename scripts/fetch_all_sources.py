import yfinance as yf
import pandas as pd
import json
import glob
import os
import time
import random
import requests
import io
import warnings
from datetime import datetime, timedelta

warnings.simplefilter(action="ignore", category=FutureWarning)

# Defaults
START_DATE_DEFAULT = "2015-01-01"
LOCAL_TZ = "Europe/Berlin"


def setup_folders():
    for src in ["yahoo", "stooq"]:
        os.makedirs(f"data/{src}", exist_ok=True)


def extract_symbol(epic):
    parts = epic.split(".")
    return parts[2].upper() if len(parts) >= 3 else epic.upper()


def check_and_convert_tz(df, name):
    if df.empty:
        return df
    try:
        if df.index.tz is not None:
            df.index = df.index.tz_convert(LOCAL_TZ).tz_localize(None)
        else:
            df.index = (
                df.index.tz_localize("UTC").tz_convert(LOCAL_TZ).tz_localize(None)
            )
    except Exception as e:
        print(f"⚠️ TZ-Sync Warnung {name}: {e}")
    return df


def fetch_stooq(epic, resolution):
    if "HOUR" in resolution or "MINUTE" in resolution:
        print(
            f"⏩ Stooq übersprungen für {epic}: Stooq bietet kostenlos nur DAY-Daten."
        )
        return

    symbol = extract_symbol(epic).lower()
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    headers = {"User-Agent": "Mozilla/5.0"}
    filename = f"{epic.replace('.', '_')}_{resolution}.csv"
    path = f"data/stooq/{filename}"

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            if not df.empty and "Close" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
                df = check_and_convert_tz(df, symbol)
                df[["Open", "High", "Low", "Close"]].to_csv(path)
                print(f"✅ Stooq: {filename}")
    except Exception as e:
        print(f"❌ Stooq Fehler {symbol}: {e}")


def fetch_yahoo(epic, resolution):
    symbol = extract_symbol(epic)
    ticker = f"{symbol}=X" if len(symbol) == 6 else symbol

    # --- YAHOO LIMIT LOGIK ---
    if "HOUR" in resolution:
        interval = "1h"
        # Yahoo erlaubt max 730 Tage für 1h
        start = (datetime.now() - timedelta(days=729)).strftime("%Y-%m-%d")
    elif "MINUTE_15" in resolution:
        interval = "15m"
        # Yahoo erlaubt max 60 Tage für 15m
        start = (datetime.now() - timedelta(days=59)).strftime("%Y-%m-%d")
    else:
        interval = "1d"
        start = START_DATE_DEFAULT

    filename = f"{epic.replace('.', '_')}_{resolution}.csv"
    path = f"data/yahoo/{filename}"

    try:
        df = yf.download(
            ticker, start=start, interval=interval, progress=False, auto_adjust=True
        )
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = check_and_convert_tz(df, ticker)
            df[["Open", "High", "Low", "Close"]].dropna().to_csv(path)
            print(f"✅ Yahoo: {filename} (Start: {start})")
        else:
            print(f"⚠️ Yahoo: {ticker} lieferte keine Daten für {interval}.")
    except Exception as e:
        print(f"❌ Yahoo Fehler {ticker}: {e}")


if __name__ == "__main__":
    setup_folders()
    targets = set()
    for acc_path in glob.glob("accounts/*.json"):
        with open(acc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            bot_cfg = data.get("bot", {})
            for name, a_cfg in data.get("assets", {}).items():
                # Falls Asset keine eigene Source hat, nimm Bot-Source
                src = a_cfg.get("data_source", bot_cfg.get("data_source", "yahoo"))
                res = a_cfg.get("resolution", bot_cfg.get("resolution", "HOUR"))
                targets.add((src, a_cfg["epic"], res))

    for src, epic, res in targets:
        if src.lower() == "stooq":
            fetch_stooq(epic, res)
        else:
            fetch_yahoo(epic, res)
        time.sleep(random.randint(1, 3))
