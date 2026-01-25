#!/bin/bash
# Setup-Script für FWBG Optimizer
# Verwendung: bash setup.sh
#
# Dieses Script:
# 1. Erstellt Python Virtual Environment
# 2. Installiert alle Dependencies
# 3. Erstellt benötigte Verzeichnisse
# 4. Lädt Makro-Indikatoren herunter (VIX, DXY, Bonds, etc.)
# 5. Lädt Asset-Daten von Yahoo Finance (Fallback wenn keine ForexSB-Daten)
# 6. Erstellt Strategie-Template

set -e

echo "============================================"
echo "     FWBG Optimizer - Full Setup"
echo "============================================"
echo ""

# Farben für Ausgabe
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Prüfe Python3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 nicht gefunden. Bitte Python 3.10+ installieren.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}Python Version: $PYTHON_VERSION${NC}"

# 1. Virtual Environment
echo ""
echo -e "${BLUE}=== 1/6 Virtual Environment ===${NC}"
if [ ! -d ".venv" ]; then
    echo "Erstelle neues Virtual Environment..."
    python3 -m venv .venv
    echo -e "${GREEN}Virtual Environment erstellt${NC}"
else
    echo -e "${YELLOW}Virtual Environment existiert bereits${NC}"
fi

# Aktivieren
source .venv/bin/activate
echo "Virtual Environment aktiviert"

# 2. Dependencies installieren
echo ""
echo -e "${BLUE}=== 2/6 Python Dependencies ===${NC}"
echo "Upgrade pip..."
pip install --upgrade pip -q

echo "Installiere requirements.txt..."
pip install -r requirements.txt -q
echo -e "${GREEN}Dependencies installiert${NC}"

# 3. Verzeichnisse erstellen
echo ""
echo -e "${BLUE}=== 3/6 Verzeichnisse ===${NC}"
mkdir -p data/forexsb
mkdir -p data/downloads/forexsb
mkdir -p data/yahoo
mkdir -p data/stooq
mkdir -p strategies
mkdir -p accounts/main_demo/plots
mkdir -p accounts/main_demo/history
echo -e "${GREEN}Verzeichnisse erstellt${NC}"

# 4. ForexSB-Daten importieren
echo ""
echo -e "${BLUE}=== 4/6 ForexSB-Daten ===${NC}"
RAW_FOREXSB_COUNT=$(ls data/downloads/forexsb/*_H1.csv 2>/dev/null | wc -l)
if [ "$RAW_FOREXSB_COUNT" -gt 0 ]; then
    echo "ForexSB Rohdaten gefunden: $RAW_FOREXSB_COUNT Dateien"
    echo "Importiere und konvertiere..."
    python3 forexsb_importer.py H1
    echo -e "${GREEN}ForexSB-Daten importiert${NC}"
else
    echo -e "${YELLOW}WICHTIG: ForexSB Rohdaten fehlen!${NC}"
    echo ""
    echo "Bitte kopiere die ForexSB CSV-Dateien nach:"
    echo "  data/downloads/forexsb/"
    echo ""
    echo "Die Daten können hier heruntergeladen werden:"
    echo "  https://forexsb.com/historical-forex-data"
    echo ""
    echo "  1. Symbol auswählen"
    echo "  2. Format: 'Excel (CSV)'"
    echo "  3. Period: 'H1' (und optional M15)"
    echo "  4. 'Load data' klicken"
    echo "  5. Alle Dateien herunterladen"
    echo "  6. CSV-Dateien nach data/downloads/forexsb/ kopieren"
    echo ""
    read -p "Drücke Enter wenn die Daten kopiert wurden (oder Ctrl+C zum Abbrechen)..."

    # Nochmal prüfen nach Enter
    RAW_FOREXSB_COUNT=$(ls data/downloads/forexsb/*_H1.csv 2>/dev/null | wc -l)
    if [ "$RAW_FOREXSB_COUNT" -gt 0 ]; then
        echo ""
        echo "ForexSB Rohdaten gefunden: $RAW_FOREXSB_COUNT Dateien"
        echo "Importiere und konvertiere..."
        python3 forexsb_importer.py H1
        echo -e "${GREEN}ForexSB-Daten importiert${NC}"
    else
        echo -e "${RED}Keine ForexSB-Daten gefunden. Yahoo-Fallback wird verwendet.${NC}"
    fi
fi

# 5. Makro-Daten herunterladen
echo ""
echo -e "${BLUE}=== 5/6 Makro-Daten Download ===${NC}"
python3 << 'PYTHON_SCRIPT'
import yfinance as yf
import pandas as pd
import os
import time
import glob
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

DATA_PATH = "./data/forexsb"
os.makedirs(DATA_PATH, exist_ok=True)

# ============================================
# PRÜFE EXISTIERENDE FOREXSB-DATEN
# ============================================
existing_hourly = glob.glob(f"{DATA_PATH}/*_HOUR.csv")
existing_assets = set()
for f in existing_hourly:
    name = os.path.basename(f).replace("_HOUR.csv", "")
    # Ignoriere Makro-Indikatoren
    if not any(x in name for x in ["VIX", "DXY", "TNX", "TYX", "XL", "TLT", "HYG", "LQD", "DAY", "SKEW"]):
        existing_assets.add(name)

print(f"Gefundene Asset-Dateien: {len(existing_assets)}")
if existing_assets:
    print(f"  Beispiele: {list(existing_assets)[:5]}")

# ============================================
# ALLE TRADING ASSETS
# ============================================
ASSETS_HOURLY = {
    # FOREX - Majors
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    # FOREX - Crosses
    "EURGBP": "EURGBP=X",
    "EURCAD": "EURCAD=X",
    "EURCHF": "EURCHF=X",
    "EURJPY": "EURJPY=X",
    "EURNZD": "EURNZD=X",
    "EURAUD": "EURAUD=X",
    "GBPJPY": "GBPJPY=X",
    "GBPCHF": "GBPCHF=X",
    "GBPAUD": "GBPAUD=X",
    "GBPNZD": "GBPNZD=X",
    "GBPCAD": "GBPCAD=X",
    "AUDJPY": "AUDJPY=X",
    "AUDNZD": "AUDNZD=X",
    "AUDCAD": "AUDCAD=X",
    "AUDCHF": "AUDCHF=X",
    "NZDJPY": "NZDJPY=X",
    "NZDCAD": "NZDCAD=X",
    "NZDCHF": "NZDCHF=X",
    "CADJPY": "CADJPY=X",
    "CADCHF": "CADCHF=X",
    "CHFJPY": "CHFJPY=X",
    # Commodities
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "BRENT": "BZ=F",
    "WTI": "CL=F",
    # Indizes
    "SPX500": "^GSPC",
    "NAS100": "^NDX",
    "DOW30": "^DJI",
    "DAX": "^GDAXI",
    "FTSE100": "^FTSE",
}

# Makro-Indikatoren (Daily) - Immer herunterladen
MACRO_DAILY = {
    "VIX_DAY": "^VIX",
    "VVIX_DAY": "^VVIX",
    "SKEW_DAY": "^SKEW",
    "VXN_DAY": "^VXN",
    "TNX_DAY": "^TNX",
    "TYX_DAY": "^TYX",
    "FVX_DAY": "^FVX",
    "IRX_DAY": "^IRX",
    "DXY_DAY": "DX-Y.NYB",
    "GOLD_FUT_DAY": "GC=F",
    "OIL_FUT_DAY": "CL=F",
    "SILVER_FUT_DAY": "SI=F",
    "SPX_DAY": "^GSPC",
    "NASDAQ_DAY": "^IXIC",
    "DOW_DAY": "^DJI",
    "RUSSELL_DAY": "^RUT",
    "NIKKEI_DAY": "^N225",
    "HANGSENG_DAY": "^HSI",
    "FTSE_DAY": "^FTSE",
    "DAX_IDX_DAY": "^GDAXI",
    "XLF_DAY": "XLF",
    "XLE_DAY": "XLE",
    "XLK_DAY": "XLK",
    "XLU_DAY": "XLU",
    "XLP_DAY": "XLP",
    "TLT_DAY": "TLT",
    "HYG_DAY": "HYG",
    "LQD_DAY": "LQD",
}

# Makro-Indikatoren (Hourly)
MACRO_HOURLY = {
    "DXY_HOUR": "DX-Y.NYB",
    "VIX_HOUR": "^VIX",
}


def safe_get_columns(df):
    """Sicher Spalten aus DataFrame extrahieren (MultiIndex handling)."""
    if df is None or len(df) == 0:
        return []
    if isinstance(df.columns, pd.MultiIndex):
        return list(df.columns.get_level_values(0))
    return list(df.columns)


def download_hourly_ohlc(name, ticker, period="2y"):
    """Lädt Hourly OHLC-Daten."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, period=period, interval="1h", auto_adjust=True, progress=False)

        if df is None or len(df) == 0:
            return 0

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        cols = list(df.columns)
        required = ["Open", "High", "Low", "Close"]
        if not all(c in cols for c in required):
            return 0

        df = df[required].copy()
        df.index.name = "Time"
        df = df.dropna()

        if len(df) > 0:
            filename = f"{DATA_PATH}/{name}_HOUR.csv"
            df.to_csv(filename)
            return len(df)
    except Exception:
        pass
    return 0


def download_daily_close(name, ticker):
    """Lädt Daily Close-Daten."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)

        if df is None or len(df) == 0:
            return 0

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        cols = list(df.columns)
        if "Close" not in cols:
            return 0

        df = df[["Close"]].copy()
        df.index.name = "Datetime"
        df = df.dropna()

        if len(df) > 0:
            filename = f"{DATA_PATH}/{name}.csv"
            df.to_csv(filename)
            return len(df)
    except Exception:
        pass
    return 0


def download_hourly_close(name, ticker):
    """Lädt Hourly Close-Daten."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, period="2y", interval="1h", auto_adjust=True, progress=False)

        if df is None or len(df) == 0:
            return 0

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        cols = list(df.columns)
        if "Close" not in cols:
            return 0

        df = df[["Close"]].copy()
        df.index.name = "Datetime"
        df = df.dropna()

        if len(df) > 0:
            filename = f"{DATA_PATH}/{name}.csv"
            df.to_csv(filename)
            return len(df)
    except Exception:
        pass
    return 0


# ============================================
# MAKRO-INDIKATOREN (immer aktualisieren)
# ============================================
print("\n" + "="*50)
print("MAKRO-INDIKATOREN (Daily)")
print("="*50)
macro_success = 0
total = len(MACRO_DAILY)
for i, (name, ticker) in enumerate(MACRO_DAILY.items(), 1):
    days = download_daily_close(name, ticker)
    if days > 0:
        print(f"  [{i:2}/{total}] {name}: {days} Tage")
        macro_success += 1
    else:
        print(f"  [{i:2}/{total}] {name}: -")
    time.sleep(0.2)
print(f"Erfolgreich: {macro_success}/{total}")

print("\n" + "="*50)
print("MAKRO-INDIKATOREN (Hourly)")
print("="*50)
hourly_success = 0
total = len(MACRO_HOURLY)
for i, (name, ticker) in enumerate(MACRO_HOURLY.items(), 1):
    hours = download_hourly_close(name, ticker)
    if hours > 0:
        print(f"  [{i:2}/{total}] {name}: {hours} Stunden")
        hourly_success += 1
    time.sleep(0.2)
print(f"Erfolgreich: {hourly_success}/{total}")

# ============================================
# ASSET-DATEN (nur wenn nicht vorhanden)
# ============================================
missing_assets = {k: v for k, v in ASSETS_HOURLY.items() if k not in existing_assets}

if missing_assets:
    print("\n" + "="*50)
    print(f"TRADING ASSETS - Yahoo Fallback ({len(missing_assets)} fehlend)")
    print("="*50)
    print("(ForexSB-Daten haben Priorität wenn vorhanden)")
    success_count = 0
    total = len(missing_assets)
    for i, (name, ticker) in enumerate(missing_assets.items(), 1):
        bars = download_hourly_ohlc(name, ticker)
        if bars > 0:
            print(f"  [{i:2}/{total}] {name}: {bars} Bars (Yahoo)")
            success_count += 1
        else:
            print(f"  [{i:2}/{total}] {name}: -")
        time.sleep(0.3)
    print(f"Erfolgreich: {success_count}/{total}")
else:
    print("\n" + "="*50)
    print("TRADING ASSETS")
    print("="*50)
    print(f"Alle {len(existing_assets)} Assets bereits vorhanden (ForexSB)")

# Zusammenfassung
print("\n" + "="*50)
print("ZUSAMMENFASSUNG")
print("="*50)
total_files = len([f for f in os.listdir(DATA_PATH) if f.endswith('.csv')])
print(f"Gesamt: {total_files} CSV-Dateien in {DATA_PATH}/")
PYTHON_SCRIPT

# 6. Strategie-Template erstellen
echo ""
echo -e "${BLUE}=== 6/6 Strategie-Template ===${NC}"
if [ ! -f "strategies/default.json" ]; then
    cat > strategies/default.json << 'EOF'
{
    "name": "Default Strategy",
    "description": "Standard Walk-Forward Optimierung mit Plateau-Erkennung",
    "version": "1.0",
    "parameters": {
        "walk_forward_folds": 8,
        "oos_size": 4000,
        "min_trades": 200,
        "corr_threshold": 0.75,
        "feature_stability_min": 3
    },
    "grids": {
        "FOREX": {
            "tp": [15, 20, 25, 30, 40, 50, 60, 80],
            "sl": [15, 20, 25, 30, 40, 50, 60, 80],
            "ct": [0.52, 0.55, 0.58, 0.60, 0.65, 0.70]
        },
        "INDEX": {
            "tp": [20, 30, 50, 70, 100, 150],
            "sl": [20, 30, 50, 70, 100, 150],
            "ct": [0.52, 0.55, 0.60, 0.65, 0.70]
        },
        "COMMODITY": {
            "tp": [20, 30, 40, 60, 80, 100],
            "sl": [20, 30, 40, 60, 80, 100],
            "ct": [0.52, 0.55, 0.58, 0.62, 0.65, 0.70]
        }
    }
}
EOF
    echo -e "${GREEN}Strategie-Template erstellt${NC}"
else
    echo -e "${YELLOW}Strategie-Template existiert bereits${NC}"
fi

# Finale Zusammenfassung
echo ""
echo "============================================"
echo -e "${GREEN}     SETUP ABGESCHLOSSEN!${NC}"
echo "============================================"
echo ""
echo "Daten-Verzeichnis:"
ls -la data/forexsb/*.csv 2>/dev/null | wc -l | xargs -I {} echo "  {} CSV-Dateien vorhanden"
echo ""
echo -e "${BLUE}Optimizer starten:${NC}"
echo "  source .venv/bin/activate"
echo "  python -m optimizer --strategy-file strategies/default.json"
echo ""
echo -e "${BLUE}Andere Befehle:${NC}"
echo "  python -m optimizer --runs              # Runs anzeigen"
echo "  python -m optimizer --compare ID1 ID2  # Runs vergleichen"
echo "  TIMEFRAME=MINUTE_15 python -m optimizer ...  # Anderer Timeframe"
echo ""
