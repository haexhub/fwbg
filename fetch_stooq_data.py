import pandas_datareader.data as web
import os
import yaml
import time
from datetime import datetime


def fetch_from_stooq():
    if not os.path.exists("mapping.yaml"):
        print("❌ mapping.yaml fehlt!")
        return

    with open("mapping.yaml", "r") as f:
        mapping = yaml.safe_load(f).get("markets", {})

    if not os.path.exists("data"):
        os.makedirs("data")

    # Stooq Symbole sind oft einfach das Paar (z.B. EURUSD)
    # Manchmal braucht man das Präfix '^' oder '.f'
    print("🚀 Starte Download von Stooq (historische Daten)...")

    start = datetime(2010, 1, 1)
    end = datetime.now()

    for epic, yahoo_ticker in mapping.items():
        # Konvertiere Yahoo-Ticker (EURUSD=X) zu Stooq-Ticker (EURUSD)
        stooq_ticker = yahoo_ticker.replace("=X", "").lower()

        print(f"📥 Lade {stooq_ticker} von Stooq...")
        try:
            # Stooq Abfrage
            df = web.DataReader(stooq_ticker, "stooq", start, end)

            if df.empty:
                print(f"   ⚠️ Keine Daten für {stooq_ticker} bei Stooq gefunden.")
                continue

            # Stooq liefert Daten oft absteigend (neueste zuerst), wir brauchen aufsteigend
            df = df.sort_index()

            # Umbenennen der Spalten, damit sie zum Bot passen
            # Stooq nutzt: Open, High, Low, Close, Volume
            clean_df = df[["Open", "High", "Low", "Close"]].copy()

            safe_name = epic.replace(".", "_")
            file_path = f"data/{safe_name}.csv"

            # Falls Yahoo-Daten schon da sind, können wir sie mergen (optional)
            # Hier überschreiben wir sie einfach mit der sauberen Stooq-Historie
            clean_df.to_csv(file_path)

            print(f"   ✅ Erfolg! {len(clean_df)} Tage geladen.")
            print(
                f"   📅 Bereich: {clean_df.index[0].date()} bis {clean_df.index[-1].date()}"
            )

            # Kurze Pause gegen Rate-Limiting
            time.sleep(1)

        except Exception as e:
            print(f"   ❌ Fehler bei Stooq-Download für {stooq_ticker}: {e}")


if __name__ == "__main__":
    fetch_from_stooq()
