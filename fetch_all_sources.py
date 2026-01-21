import yfinance as yf
import pandas as pd
import yaml
import os
import time
import random
import requests
import io

# Konfiguration
START_DATE = "2015-01-01"


def setup_folders():
    for src in ["yahoo", "stooq"]:
        os.makedirs(f"data/{src}", exist_ok=True)


def fetch_stooq_direct(mapping):
    print("\n--- ⚪ Starte Stooq Direct-Download ---")
    headers = {"User-Agent": "Mozilla/5.0"}

    for epic, yahoo_ticker in mapping.items():
        # Stooq Ticker Format: eurusd, usdjpy, etc. (kleingeschrieben, ohne =X)
        stooq_ticker = yahoo_ticker.replace("=X", "").lower()
        safe_name = epic.replace(".", "_")

        # Direkter CSV-Download Link von Stooq
        url = f"https://stooq.com/q/d/l/?s={stooq_ticker}&i=d"

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text))

                if not df.empty and len(df) > 100:
                    # Stooq Spalten: Date, Open, High, Low, Close, Volume
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)
                    df = df.sort_index()

                    # Nur die benötigten Spalten speichern
                    clean_df = df[["Open", "High", "Low", "Close"]]
                    clean_df.to_csv(f"data/stooq/{safe_name}.csv")
                    print(f"✅ Stooq: {stooq_ticker.upper()} ({len(clean_df)} Zeilen)")
                else:
                    print(
                        f"⚠️ Stooq: {stooq_ticker.upper()} lieferte keine ausreichenden Daten."
                    )
            else:
                print(
                    f"❌ Stooq Server Fehler für {stooq_ticker}: Status {response.status_code}"
                )

        except Exception as e:
            print(f"❌ Fehler bei Stooq-Download {stooq_ticker}: {e}")

        time.sleep(random.randint(2, 4))  # Freundlich bleiben


def fetch_yahoo_safe(mapping):
    print("\n--- 🔵 Starte Yahoo Safe-Fetch ---")
    for epic, ticker in mapping.items():
        safe_name = epic.replace(".", "_")
        try:
            df = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                clean_df = df[["Open", "High", "Low", "Close"]].dropna()
                clean_df.to_csv(f"data/yahoo/{safe_name}.csv")
                print(f"✅ Yahoo: {ticker} ({len(clean_df)} Zeilen)")
            else:
                print(f"⚠️ Yahoo: {ticker} leer.")
        except Exception as e:
            print(f"❌ Yahoo Fehler bei {ticker}: {e}")

        time.sleep(random.randint(3, 6))


if __name__ == "__main__":
    setup_folders()
    with open("mapping.yaml", "r") as f:
        m = yaml.safe_load(f).get("markets", {})

    # Erst Stooq, dann Yahoo
    fetch_stooq_direct(m)
    print("\n⏳ Pause zwischen den Quellen...")
    time.sleep(10)
    fetch_yahoo_safe(m)
