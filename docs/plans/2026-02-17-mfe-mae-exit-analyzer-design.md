# MFE/MAE Exit Analyzer — Design

## Ziel

CLI-Tool (`fwbg analyze`) das für ein Asset oder eine Asset-Klasse die optimalen TP/SL-Bereiche
per MFE/MAE-Analyse (Maximum Favorable/Adverse Excursion) ermittelt. Ergebnisse werden als
individuelle JSON-Dateien pro Asset geschrieben — konsumierbar sowohl von CLI als auch vom
fwbg-dashboard (WebUI).

## Motivation

Der aktuelle Grid-Search iteriert blind über vordefinierte TP/SL-Kombinationen. Bei BRENT
mit ATR-basierten Exits z.B. nur 11 Trades in 2 von 8 Folds — die TP/SL-Ranges passen nicht
zum Asset. MFE/MAE-Analyse liefert datengetriebene Grid-Vorschläge basierend auf der
tatsächlichen Preisstruktur.

## CLI Interface

```bash
# Einzelnes Asset
fwbg analyze BRENT_HOUR.csv --strategy-file strategies/exploration_scalping.json

# Asset-Klasse (jedes Asset einzeln)
fwbg analyze --asset-class COMMODITY --strategy-file strategies/exploration_scalping.json

# Optional
fwbg analyze BRENT_HOUR.csv --max-bars 96       # Lookforward-Window (default: 48)
fwbg analyze BRENT_HOUR.csv --output-dir ./out   # Output-Verzeichnis (default: test_results/analyze/)
```

Strategy-Datei liefert Exit-Strategie-Typ (fixed/atr_based) und exit_params. Ohne Strategy →
Default ATR-based mit atr_period=14.

## Output

### JSON-Datei pro Asset

Pfad: `test_results/analyze/{SYMBOL}_{TIMEFRAME}.json`

```json
{
  "symbol": "BRENT",
  "timeframe": "HOUR",
  "data_file": "BRENT_HOUR.csv",
  "bars_analyzed": 42000,
  "max_bars_forward": 48,
  "exit_strategy": "atr_based",
  "exit_params": { "atr_period": 7 },
  "analyzed_at": "2026-02-17T16:30:00",

  "mfe_mae": {
    "long": {
      "mfe_percentiles": { "25": 0.42, "50": 0.81, "75": 1.45, "90": 2.31, "95": 3.12 },
      "mae_percentiles": { "25": 0.28, "50": 0.55, "75": 1.02, "90": 1.78, "95": 2.45 }
    },
    "short": {
      "mfe_percentiles": { "25": 0.40, "50": 0.78, "75": 1.38, "90": 2.20, "95": 2.98 },
      "mae_percentiles": { "25": 0.30, "50": 0.58, "75": 1.08, "90": 1.85, "95": 2.52 }
    }
  },

  "capture_matrix": [
    {
      "tp": 0.5, "sl": 0.8,
      "win_rate_long": 0.58, "win_rate_short": 0.55,
      "avg_trades_per_month": 142, "rrr": 0.625,
      "edge_long": 0.04, "edge_short": 0.01
    }
  ],

  "suggested_grid": {
    "tp": [0.5, 0.8, 1.2],
    "sl": [0.8, 1.0, 1.5],
    "reasoning": "MFE P50=0.81, P75=1.45 -> TP range 0.5-1.2. MAE P75=1.02, P90=1.78 -> SL range 0.8-1.5"
  }
}
```

Percentile-Werte in ATR-Multiplikatoren (atr_based) bzw. Spread-Multiplikatoren/Pips (fixed).

### CLI Terminal-Output

Formatierte Tabellen mit:
1. MFE/MAE Percentile-Tabelle (Long + Short)
2. Top-10 Capture-Matrix-Einträge (sortiert nach Edge)
3. Suggested Grid als JSON-Snippet

## Algorithmus

### 1. MFE/MAE Berechnung (Numba)

Für jeden Bar i, schaue max_bars vorwärts:
- Entry = Open[i+1] (nächster Bar, wie in der echten Simulation)
- MFE Long = max(High[j] for j in [i+1, i+1+max_bars]) - entry
- MAE Long = entry - min(Low[j] for j in [i+1, i+1+max_bars])
- MFE/MAE Short analog gespiegelt

### 2. ATR-Normalisierung

Rohe Preisdistanzen / ATR[i] -> ATR-Multiplikatoren. ATR berechnet über die
konfigurierte atr_period. Bei fixed Exit-Strategie: Division durch Spread statt ATR.

### 3. Capture-Rate-Berechnung (Numba)

Für jede TP/SL-Kombination aus einem feinen Raster (z.B. 0.1 bis 5.0 in 0.1-Schritten):
- Pro Bar: Iteriere vorwärts, prüfe ob TP oder SL zuerst erreicht wird
- Bei gleichzeitigem Hit im selben Bar: konservativ als Loss werten (wie in compute_targets_numba)
- Zähle Win-Rate Long, Win-Rate Short, Anzahl Trades
- Edge = WinRate * TP - (1-WinRate) * SL (mathematische Erwartung pro Trade in ATR-Einheiten)

### 4. Grid-Vorschlag

Basierend auf MFE/MAE-Percentilen:
- TP-Kandidaten: P50, P60, P75 der MFE (dort werden 50-75% der Moves eingefangen)
- SL-Kandidaten: P75, P85, P90 der MAE (Schutz vor 75-90% der Adverse Moves)
- Filtere Kombinationen mit negativem Edge
- Wähle 3-4 TP und 3-4 SL Werte mit bester Balance aus Capture-Rate und Edge

## Dateistruktur

```
src/fwbg/analysis/
    __init__.py
    mfe_mae.py           # Numba: compute_mfe_mae(), compute_capture_rates()
    exit_analyzer.py      # Orchestrierung, JSON-Output, CLI-Tabellen

src/fwbg/cli/            # Bestehendes CLI erweitern um analyze-Subcommand

tests/analysis/
    test_mfe_mae.py       # Unit-Tests Numba-Funktionen + Integration
```

### mfe_mae.py (~150 Zeilen)
- `compute_mfe_mae(open_, high, low, max_bars)` — Numba, gibt 4 Arrays zurück
- `compute_capture_rates(open_, high, low, atr, tp_values, sl_values, max_bars)` — Numba,
  gibt Win-Rates pro TP/SL-Kombination zurück

### exit_analyzer.py (~250 Zeilen)
- `analyze_asset(data_file, exit_strategy, exit_params, max_bars)` — Hauptfunktion
- `_compute_percentiles(mfe, mae, atr)` — Percentile-Berechnung nach ATR-Normalisierung
- `_suggest_grid(percentiles, capture_rates)` — Grid-Vorschlag-Logik
- `_format_terminal_output(result)` — Tabellen-Formatierung
- `_write_json(result, output_path)` — JSON-Serialisierung

### CLI-Integration
- Neues Subcommand `analyze` in bestehender CLI-Struktur
- Argumente: positional asset file, --asset-class, --strategy-file, --max-bars, --output-dir
