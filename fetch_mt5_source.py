import MetaTrader5 as mt5
import pandas as pd
import json
import os
import glob
from datetime import datetime

# --- KONFIGURATION ---
LOCAL_TZ = "Europe/Berlin"


def fetch_from_mt5(epic, resolution, count=20000):
    """Holt historische Daten direkt aus dem MT5 Terminal."""

    # 1. Symbol extrahieren (z.B. EURUSD)
    parts = epic.split(".")
    symbol = parts[2] if len(parts) >= 3 else epic
    # MT5 Symbole haben oft kein Präfix, evtl. Suffix (z.B. EURUSD.m) - hier anpassen falls nötig

    # 2. Timeframe Mapping
    tf = mt5.TIMEFRAME_D1
    if "HOUR" in resolution:
        tf = mt5.TIMEFRAME_H1
    elif "MINUTE_15" in resolution:
        tf = mt5.TIMEFRAME_M15

    # 3. Daten abrufen
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

    if rates is None or len(rates) == 0:
        # Versuch mit alternativem Symbol (manche Broker nutzen EURUSD, andere EURUSD.pro)
        print(f"⚠️ {symbol} nicht gefunden, versuche Suche...")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)

    # Spalten auf unseren Standard bringen
    clean_df = df[["open", "high", "low", "close"]].copy()
    clean_df.columns = ["Open", "High", "Low", "Close"]

    # Speichern
    filename = f"{epic.replace('.', '_')}_{resolution}.csv"
    path = f"data/mt5/{filename}"
    os.makedirs("data/mt5", exist_ok=True)
    clean_df.to_csv(path)

    print(f"✅ MT5: {filename} ({len(clean_df)} Zeilen geladen)")
    return path


if __name__ == "__main__":
    if not mt5.initialize():
        print("❌ MT5 Initialization failed. Ist das Terminal offen?")
        quit()

    # Hole Assets aus Accounts
    for acc_path in glob.glob("accounts/*.json"):
        with open(acc_path, "r") as f:
            data = json.load(f)
            for name, a_cfg in data["assets"].items():
                res = a_cfg.get("resolution", data["bot"]["resolution"])
                fetch_from_mt5(a_cfg["epic"], res)

    mt5.shutdown()
