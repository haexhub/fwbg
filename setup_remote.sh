#!/bin/bash
# Setup-Script für Remote-Rechner (64GB RAM)
# Verwendung: bash setup_remote.sh

set -e

echo "=== FWBG Optimizer Setup ==="

# Python venv erstellen
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Aktivieren und Dependencies installieren
source .venv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verzeichnisse erstellen
mkdir -p data/forexsb
mkdir -p test_results
mkdir -p strategies
mkdir -p accounts/main_demo/plots

echo ""
echo "=== Setup complete ==="
echo ""
echo "Nächste Schritte:"
echo "1. Kopiere Daten nach data/forexsb/"
echo "2. Aktiviere venv: source .venv/bin/activate"
echo "3. Starte Run: python -m optimizer --strategy-file strategies/symmetric_grid.json"
echo ""
echo "Für mehr CPU-Cores (bei 64GB RAM), editiere optimizer/main.py:"
echo "  Zeile ~98: mp.cpu_count() * 0.50  (statt 0.25)"
