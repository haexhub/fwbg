# Rule-Based Strategy Engine

**Status:** Roadmap
**Priorität:** Nice-to-have (nach ML-Exploration)
**Kontext:** Dashboard-Integration, manuelle Hypothesen-Falsifizierung

## Motivation

- ML-Exploration läuft, aber Trader wollen auch manuell Strategien testen
- Dashboard (TradingView-Nachbau) soll visuellen Strategy Builder bekommen
- Regelbasierte Strategien als Baseline für ML-Vergleich
- Schnelle Falsifizierung von Trading-Hypothesen mit Walk-Forward-Validation

## Konzept

Neuer Model-Typ `rule_based` neben `xgboost`. Ersetzt nur die Prediction-Stufe, gesamte Pipeline bleibt identisch:

```
Indicators → Rule Engine → Signal (0/1) → Simulation → Grid Search → Validation
```

Kein CT (Confidence Threshold) nötig, da Signale binär sind.

## Strategy JSON Format

```json
{
  "model": {
    "type": "rule_based",
    "rules": {
      "long": {
        "all": [
          {"column": "C", "op": ">", "ref": "trend_ema_21"},
          {"column": "trend_ema_21", "op": ">", "ref": "trend_sma_50"},
          {"column": "C", "op": "between", "ref": ["trend_sma_50", "trend_ema_21"]},
          {"column": "trend_ema_slope_21", "op": ">", "value": 0}
        ]
      },
      "short": {
        "all": [
          {"column": "C", "op": "<", "ref": "trend_ema_21"},
          {"column": "trend_ema_21", "op": "<", "ref": "trend_sma_50"},
          {"column": "C", "op": "between", "ref": ["trend_ema_21", "trend_sma_50"]},
          {"column": "trend_ema_slope_21", "op": "<", "value": 0}
        ]
      }
    }
  }
}
```

### Operatoren

| Op | Beschreibung | Beispiel |
|----|-------------|---------|
| `>`, `<`, `>=`, `<=`, `==` | Vergleich mit Spalte (`ref`) oder Wert (`value`) | `C > trend_ema_21` |
| `between` | Wert liegt zwischen zwei Spalten | `C between [sma_50, ema_21]` |
| `crosses_above` | Crossover (prev <= ref, now > ref) | `ema_8 crosses_above ema_21` |
| `crosses_below` | Crossunder | `ema_8 crosses_below ema_21` |

### Logik-Kombination

- `all`: Alle Bedingungen müssen zutreffen (AND)
- `any`: Mindestens eine Bedingung (OR)
- Verschachtelbar: `{"all": [cond1, {"any": [cond2, cond3]}]}`

## Implementierung

### 1. `RuleBasedModel` Klasse

Selbes Interface wie `XGBoostModel`:
- `fit()` → No-op (keine Trainingsphase)
- `predict(X) → np.array` → Evaluiert Regeln, gibt 0/1 zurück
- Kein Feature-Selection nötig (Trader wählt Features explizit)

### 2. Änderungen in `nested_cv.py`

- Model-Typ Dispatch: `if model_type == "rule_based": model = RuleBasedModel(rules)`
- CT-Grid überspringen (oder CT fest auf 0.5 setzen, da Signal binär)
- Feature-Selection überspringen (Trader definiert Features über Regeln)

### 3. Dashboard Strategy Builder

- Visueller Editor: Bedingungen zusammenklicken
- Dropdown: verfügbare Spalten (aus Indicator-Plugins)
- Preview: Signale auf Chart anzeigen
- Run: Walk-Forward über API starten
- Ergebnis: Sharpe, DSR, PBO, Equity Curve

## Was identisch bleibt

- Indicator-Pipeline (Features werden normal berechnet)
- Exit Strategy (ATR-based / fixed)
- Grid Search (TP/SL, timeout_bars, regime_filter)
- Simulation (numba_core)
- Walk-Forward Validation
- Overfitting-Metriken (DSR, PBO)
- Ergebnis-Format

## Abgrenzung

- Kein Backtesting-Framework für beliebige Strategien
- Nur Strategien die auf unsere Indicator-Features zugreifen
- Kein State zwischen Bars (kein "halte Position bis X")
- Entry-only Regeln — Exits über TP/SL/Timeout wie bei ML
