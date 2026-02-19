#!/bin/bash

# Konfiguration
START_DATE="2014-01-01"
END_DATE="2026-02-19"
TIMEFRAME="m15"       # m1, m15, h1, d1
FORMAT="csv"
OUT_DIR="./data/raw_indices"

mkdir -p $OUT_DIR

# DIE RICHTIGE LISTE (Länder-Codes statt Index-Namen)
# Format: "LibraryID:Dateiname"
declare -a assets=(
    # --- USA (Prefix: usa) ---
    "usa500idxusd:US500_SPX"      # S&P 500
    "usatechidxusd:US100_NDX"     # Nasdaq 100
    "usa30idxusd:US30_DJI"        # Dow Jones Industrial

    # --- Europa (Prefix: deu, gbr, fra, eus) ---
    "deuidxeur:DE40_DAX"          # DAX 40
    "gbridxgbp:UK100_FTSE"        # FTSE 100
    "fraidxeur:FR40_CAC"          # CAC 40
    "eusidxeur:EU50_STOXX"        # EURO STOXX 50

    # --- Asien / Pazifik (Prefix: jpn, hkg, aus) ---
    "jpnidxjpy:JP225_NIKKEI"      # Nikkei 225
    "hkgidxhkd:HK50_HSI"          # Hang Seng
    "ausidxaud:AU200_ASX"         # ASX 200
    
    # --- Volatilität ---
    "volidxusd:US_VIX"            # US Volatility Index
)

echo "Starte Download (Verifizierte IDs)..."
echo "--------------------------------------------------------"

for pair in "${assets[@]}"; do
    KEY="${pair%%:*}"
    NAME="${pair##*:}"
    
    echo "Lade $NAME (ID: $KEY)..."
    
    # Wir fangen Fehler ab, falls ein Download doch scheitert
    npx dukascopy-node \
        -i "$KEY" \
        -from "$START_DATE" \
        -to "$END_DATE" \
        -t "$TIMEFRAME" \
        -f "$FORMAT" \
        -dir "$OUT_DIR" \
        -s
    
    # Check Status
    if [ $? -eq 0 ]; then
        # Datei finden und umbenennen
        # Die Library erstellt Dateien im Format: 'usa500idxusd-m15.csv'
        FOUND=$(find "$OUT_DIR" -name "${KEY}*.csv" | head -n 1)
        
        if [ -f "$FOUND" ]; then
            mv "$FOUND" "$OUT_DIR/${NAME}_${TIMEFRAME}.csv"
            echo "✅ $NAME erfolgreich."
        else
            echo "⚠️  Download lief durch, aber Datei nicht gefunden ($KEY)."
        fi
    else
        echo "❌ Fehler bei $NAME ($KEY) - ID existiert evtl. nicht im Feed."
    fi
    
    sleep 2
done

echo "--------------------------------------------------------"
echo "Fertig."