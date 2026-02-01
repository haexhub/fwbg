import pandas as pd
import os
import glob

SOURCE_DIR = "data/downloads/forexsb"
TARGET_DIR = "data/forexsb"

# Mapping für korrekte Namen
SYMBOL_MAP = {
    "DEUIDXEUR": "DAX",
    "USA30IDXUSD": "DOW30",
    "USA500IDXUSD": "SPX500",
    "USATECHIDXUSD": "NAS100",
    "GBRIDXGBP": "FTSE100",
    "BRENTCMDUSD": "BRENT",
    "XAUUSD": "GOLD",
    "XAGUSD": "SILVER",
}

# ForexSB Timeframe Mapping
TIMEFRAME_MAP = {
    "M1": "MINUTE_1",
    "M5": "MINUTE_5",
    "M15": "MINUTE_15",
    "M30": "MINUTE_30",
    "H1": "HOUR",
    "H4": "HOUR_4",
    "D1": "DAY",
}

os.makedirs(TARGET_DIR, exist_ok=True)


def repair_and_import(timeframes=None):
    """
    Importiert ForexSB Daten für die angegebenen Timeframes.

    Args:
        timeframes: Liste von Timeframes (z.B. ["H1", "M15"]) oder None für alle
    """
    if timeframes is None:
        timeframes = ["H1"]  # Default: nur Hourly

    print(f"🔧 ForexSB Importer: Verarbeite Timeframes {timeframes}...")

    for tf in timeframes:
        if tf not in TIMEFRAME_MAP:
            print(f"⚠️ Unbekannter Timeframe: {tf}")
            continue

        output_suffix = TIMEFRAME_MAP[tf]
        files = glob.glob(os.path.join(SOURCE_DIR, f"*_{tf}.csv"))

        if not files:
            print(f"⚠️ Keine {tf} Dateien gefunden in {SOURCE_DIR}")
            continue

        print(f"\n📊 Verarbeite {len(files)} {tf} Dateien...")

        for file_path in files:
            raw_name = os.path.basename(file_path).replace(f"_{tf}.csv", "")
            clean_name = SYMBOL_MAP.get(raw_name, raw_name)

            try:
                # Erst Header lesen um Format zu erkennen
                df_header = pd.read_csv(file_path, nrows=0)
                header_cols = list(df_header.columns)

                # Spezialfall: Simple Format (T,C) - z.B. VIX
                if header_cols == ["T", "C"] or (len(header_cols) == 2 and "T" in header_cols):
                    df = pd.read_csv(file_path)
                    df.columns = ["Time", "Close"]
                    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
                    df = df.dropna(subset=["Time"])
                    # Für einfache Close-only Dateien: Open=High=Low=Close, Volume=0
                    df["Open"] = df["Close"]
                    df["High"] = df["Close"]
                    df["Low"] = df["Close"]
                    df["Volume"] = 0
                    df = df[["Time", "Open", "High", "Low", "Close", "Volume"]]

                else:
                    # Standard ForexSB Format
                    # Schritt 1: Datei roh einlesen, um das Trennzeichen-Problem zu umgehen
                    # Wir überspringen die allererste Zeile (wo oft der Asset-Name steht)
                    df = pd.read_csv(file_path, skiprows=1, header=None)

                    # Schritt 2: Wenn die Datei einen Index am Anfang hat (0, 1, 2...),
                    # dann sind unsere Daten in den Spalten 1 bis 6.
                    # Wir prüfen dynamisch, wie viele Spalten wir haben.
                    if df.shape[1] >= 7:
                        df = df.iloc[:, [1, 2, 3, 4, 5, 6]]
                    elif df.shape[1] >= 6:
                        df = df.iloc[:, [0, 1, 2, 3, 4, 5]]
                    else:
                        print(f"⚠️ Überspringe {raw_name}: nur {df.shape[1]} Spalten")
                        continue

                    df.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]

                    # Schritt 3: Zeitformat säubern
                    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
                    df = df.dropna(subset=["Time"])

                target_path = os.path.join(TARGET_DIR, f"{clean_name}_{output_suffix}.csv")
                df.to_csv(target_path, index=False)
                print(f"✅ {clean_name}_{output_suffix}.csv ({len(df)} Zeilen)")

            except Exception as e:
                print(f"❌ Fehler bei {raw_name}: {e}")


if __name__ == "__main__":
    import sys
    # Default: H1, oder Timeframes als Argumente übergeben (z.B. python forexsb_importer.py H1 M15)
    timeframes = sys.argv[1:] if len(sys.argv) > 1 else ["H1"]
    repair_and_import(timeframes)
