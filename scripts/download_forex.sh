#!/bin/bash

# Konfiguration
START_DATE="2014-01-01"
END_DATE="2026-03-04"
TIMEFRAME="m15"       # m1, m15, h1, d1
FORMAT="csv"
OUT_DIR="./data/raw_forex"

mkdir -p $OUT_DIR

# Dukascopy Forex Paare
# Format: "LibraryID:Dateiname"
declare -a assets=(
    # --- Majors ---
    "eurusd:EURUSD"
    "gbpusd:GBPUSD"
    "usdjpy:USDJPY"
    "usdchf:USDCHF"
    "audusd:AUDUSD"
    "usdcad:USDCAD"
    "nzdusd:NZDUSD"

    # --- Crosses ---
    "eurgbp:EURGBP"
    "eurjpy:EURJPY"
    "gbpjpy:GBPJPY"
    "euraud:EURAUD"
    "eurchf:EURCHF"
    "audjpy:AUDJPY"
    "gbpaud:GBPAUD"
    "cadjpy:CADJPY"
)

echo "Starte Forex Download..."
echo "--------------------------------------------------------"

for pair in "${assets[@]}"; do
    KEY="${pair%%:*}"
    NAME="${pair##*:}"

    echo "Lade $NAME (ID: $KEY)..."

    npx dukascopy-node \
        -i "$KEY" \
        -from "$START_DATE" \
        -to "$END_DATE" \
        -t "$TIMEFRAME" \
        -f "$FORMAT" \
        -dir "$OUT_DIR" \
        -s

    if [ $? -eq 0 ]; then
        FOUND=$(find "$OUT_DIR" -name "${KEY}*.csv" | head -n 1)

        if [ -f "$FOUND" ]; then
            mv "$FOUND" "$OUT_DIR/${NAME}_${TIMEFRAME}.csv"
            echo "✅ $NAME erfolgreich."
        else
            echo "⚠️  Download lief durch, aber Datei nicht gefunden ($KEY)."
        fi
    else
        echo "❌ Fehler bei $NAME ($KEY)."
    fi

    sleep 2
done

echo "--------------------------------------------------------"
echo "Fertig."
