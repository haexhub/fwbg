import yfinance as yf
import pandas as pd
import os

# --- SETUP ---
DATA_PATH = "./data/forexsb"
os.makedirs(DATA_PATH, exist_ok=True)


def update_macro_data():
    """Zieht DXY und VIX und bereinigt das Multi-Index-Format von yfinance."""
    macros = {"DXY": "DX-Y.NYB", "VIX": "^VIX"}

    for name, ticker in macros.items():
        try:
            print(f"📡 Lade {name} ({ticker})...")
            # auto_adjust=True für bereinigte Kurse
            df = yf.download(ticker, period="2y", interval="1h", auto_adjust=True)

            if not df.empty:
                # KRITISCH: Multi-Index-Header entfernen (Ticker-Zeile löschen)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Nur die 'Close' Spalte behalten
                df = df[["Close"]].copy()

                # Index (Datum) Name setzen
                df.index.name = "Datetime"

                # Speichern
                filename = f"{DATA_PATH}/{name}_HOUR.csv"
                df.to_csv(filename)
                print(f"✅ {filename} erfolgreich gespeichert.")
            else:
                print(f"⚠️ Keine Daten für {name} empfangen.")
        except Exception as e:
            print(f"❌ Fehler bei {name}: {e}")


if __name__ == "__main__":
    update_macro_data()
