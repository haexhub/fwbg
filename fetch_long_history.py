import yfinance as yf
import pandas as pd
import yaml
import os
import time


def fetch_all():
    if not os.path.exists("mapping.yaml"):
        print("❌ mapping.yaml nicht gefunden!")
        return

    with open("mapping.yaml", "r") as f:
        mapping = yaml.safe_load(f).get("markets", {})

    if not os.path.exists("data"):
        os.makedirs("data")

    print(f"🚀 Starte Download-Prozess für {len(mapping)} Paare...")

    for epic, ticker in mapping.items():
        print(f"\n🔍 Prüfe {ticker}...")
        try:
            # Wir versuchen es erst mit einer kleineren Periode, falls 5y hakt
            # auto_adjust=True sorgt für saubere Kurse
            df = yf.download(
                ticker, period="max", interval="1d", progress=False, auto_adjust=True
            )

            if df is None or df.empty:
                print(
                    f"   ⚠️ Yahoo liefert keine Daten für {ticker}. Prüfe Ticker-Format!"
                )
                continue

            # Bereinigung der MultiIndex-Spalten (Yahoo Spezialität)
            if isinstance(df.columns, pd.MultiIndex):
                # Wir nehmen nur die erste Ebene der Spaltennamen
                df.columns = df.columns.get_level_values(0)

            # Sicherstellen, dass die benötigten Spalten da sind
            required = ["Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required):
                print(
                    f"   ❌ Spalten fehlen für {ticker}. Vorhanden: {list(df.columns)}"
                )
                continue

            clean_df = df[required].copy()

            # Entferne Zeilen mit NaN-Werten in den Kernspalten
            clean_df = clean_df.dropna()

            if len(clean_df) < 100:
                print(f"   ⚠️ Datensatz zu klein ({len(clean_df)} Zeilen).")
                continue

            safe_name = epic.replace(".", "_")
            file_path = f"data/{safe_name}.csv"
            clean_df.to_csv(file_path)

            print(f"   ✅ Erfolg: {len(clean_df)} Tage gespeichert in {file_path}")
            print(
                f"   📅 Zeitraum: {clean_df.index[0].date()} bis {clean_df.index[-1].date()}"
            )

            # Kurze Pause für die API-Freundlichkeit
            time.sleep(1)

        except Exception as e:
            print(f"   ❌ Kritischer Fehler bei {ticker}: {e}")


if __name__ == "__main__":
    fetch_all()
