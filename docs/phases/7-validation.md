# Phase 7: Validation & Statistische Tests

## Zweck

Die Validation-Phase prüft die Robustheit gefundener Strategien durch Walk-Forward Cross-Validation und multiple statistische Tests. Ziel: Sicherstellen, dass ein gefundener Edge kein Zufallsprodukt oder Overfitting ist.

---

## Walk-Forward Validation

FWBG verwendet **Nested Cross-Validation** mit expandierenden Fenstern — das Standardverfahren für Zeitreihen-basierte ML-Strategien.

### Fold-Struktur

```
Daten:  |──────────────────────────────────────────|
        t=0                                      t=T

Fold 1: |====TRAIN====|==TEST==|
Fold 2: |========TRAIN========|==TEST==|
Fold 3: |============TRAIN============|==TEST==|
        ...
Fold N: |==================TRAIN==================|==TEST==|
```

Jeder Fold hat mehr Trainingsdaten als der vorherige (expandierendes Fenster). Das Test-Set ist immer in der Zukunft relativ zum Training.

### Nested CV (Inner + Outer)

```
Outer Fold (Walk-Forward):
  ├── Train Split
  │   └── Inner CV (Grid Search):
  │       ├── Inner Fold 1: Train | Val
  │       ├── Inner Fold 2: Train | Val
  │       └── Inner Fold 3: Train | Val
  │       → Beste TP/SL/CT-Kombination
  └── Test Split
      → Evaluation mit bester Kombination
```

- **Outer Folds:** Walk-Forward Evaluation (expandierend)
- **Inner Folds:** Grid Search innerhalb jedes Outer Folds
- Der beste Grid-Kandidat aus dem Inner CV wird auf dem Outer Test-Split evaluiert

### Konfiguration

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `folds` | `8` | Anzahl Outer Folds |
| `oos_size` | `4000` | Out-of-Sample Bars pro Fold |
| `n_inner_folds` | `3` | Anzahl Inner Folds für Grid Search |
| `embargo_bars` | `100` | Embargo-Bars zwischen Train und Test (Purging) |
| `sample_weights` | `false` | Trade-Duration-basierte Sample Weights |
| `probability_calibration` | `false` | Wahrscheinlichkeitskalibrierung der Predictions |
| `calibration_method` | `"isotonic"` | Kalibrierungsmethode ("isotonic" oder "sigmoid") |

### Time-Series Purging (Embargo)

Zwischen Train und Test wird eine Lücke von `embargo_bars` Bars eingefügt. Das verhindert Information Leakage durch Trades die über die Train/Test-Grenze hinausgehen.

```
|====TRAIN====|###EMBARGO###|==TEST==|
```

### Sample Weights

Bei `sample_weights: true` werden Trades nach ihrer Dauer gewichtet. Längere Trades erhalten höheres Gewicht, da sie mehr Information enthalten. Die Trade-Durations werden von der Exit Strategy via `return_durations=True` berechnet.

---

## Early Pruning

Zweiphasiger Grid Search — reduziert die Rechenzeit bei großen Grids.

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `early_pruning.enabled` | `false` | Aktivieren |
| `early_pruning.keep_ratio` | `0.5` | Top-Anteil der überlebenden Combos |
| `early_pruning.min_survivors` | `10` | Mindestens N Combos überleben |

**Ablauf:**
1. **Phase 1 (Screening):** Alle Combos auf Inner Fold 0 evaluieren
2. **Pruning:** Untere Hälfte nach PnL entfernen
3. **Phase 2 (Full Eval):** Nur Survivors auf allen Inner Folds evaluieren

```json
"validation": {
  "early_pruning": {
    "enabled": true,
    "keep_ratio": 0.5,
    "min_survivors": 10
  }
}
```

---

## Statistische Tests

### 1. Monte Carlo Permutation Test

Testet ob die beobachtete Win-Rate signifikant besser als Zufall ist:

- **1000 Permutationen** der Trade-Ergebnisse → Null-Verteilung
- **p-value < 0.05** → Edge ist statistisch signifikant
- Zusätzlich: **500 Equity-Simulationspfade** für Bankruptcy-Rate

### 2. Deflated Sharpe Ratio (DSR)

*Bailey & López de Prado (2014)*

Korrigiert den beobachteten Sharpe Ratio für **Multiple Testing** — je mehr Grid-Kombinationen getestet werden, desto wahrscheinlicher findet man zufällig einen hohen Sharpe.

```
DSR = Φ((SR_obs - E[max(SR)]) / σ(SR))
```

- **E[max(SR)]** — Erwarteter maximaler Sharpe unter Null-Hypothese
- **σ(SR)** — Standardabweichung des Sharpe-Schätzers (berücksichtigt Skewness/Kurtosis)
- **DSR > 0.95** → Sharpe ist auch nach Multiple-Testing-Korrektur signifikant

### 3. Probability of Backtest Overfitting (PBO)

*Bailey, Borwein, López de Prado, Zhu (2017)*

Misst die Wahrscheinlichkeit, dass die beste In-Sample-Strategie Out-of-Sample schlecht abschneidet.

**Methode: Combinatorial Symmetric Cross-Validation (CSCV)**
- Bei 8 Walk-Forward Folds: **C(8,4) = 70** mögliche IS/OOS-Splits
- Für jeden Split: Prüft ob der beste IS-Combo auch OOS gut rankt
- **PBO > 0.50** → Wahrscheinlich Overfitting

### 4. Feature Stability

Analysiert die Konsistenz der Feature-Selektion über alle Walk-Forward Folds (siehe [Phase 4: Feature Selection](4-feature-selection.md)).

---

## Signifikanz-Schwellenwerte

| Metrik | Gut | Schlecht | Bedeutung |
|--------|-----|---------|-----------|
| p-value | < 0.05 | >= 0.05 | Edge ist (nicht) zufällig |
| DSR | > 0.95 | < 0.50 | Sharpe übersteht (nicht) Multiple-Testing |
| PBO | < 0.20 | > 0.50 | Strategie ist (wahrscheinlich) nicht overfittet |

---

## Ergebnis-Interpretation

### Ergebnis-Struktur

```json
{
  "status": "significant",
  "overfitting": {
    "dsr": {
      "dsr": 0.982,
      "observed_sr": 1.85,
      "expected_max_sr": 2.51,
      "n_strategies": 144,
      "is_significant": true
    },
    "pbo": {
      "pbo": 0.12,
      "n_cscv_splits": 70,
      "is_overfit": false,
      "degradation": 0.88,
      "logit_mean": 1.45
    }
  }
}
```

### Status-Werte

| Status | Bedeutung |
|--------|-----------|
| `significant` | Statistisch signifikanter Edge gefunden |
| `not_significant` | Kein Edge (p-value >= 0.05) |
| `no_candidates` | Keine validen Kandidaten nach Filtern |

---

## Live Bias Detection

Während der Optimierung werden in Echtzeit Bias-Checks durchgeführt:

- **Mean Bias Ratio:** Misst Abweichung der Fold-Performance vom Mittelwert
- **Extreme Folds:** Folds mit ungewöhnlich hoher/niedriger Performance
- **Win-Rate Konsistenz:** Standardabweichung der Win-Rate über Folds
- **System-weiter Check:** Am Ende des gesamten Runs über alle Assets

Detaillierte Dokumentation: [Live Bias Detection](../LIVE_BIAS_DETECTION.md)

---

## Weiterführende Dokumentation

- [Robust Validation Guide](../ROBUST_VALIDATION_GUIDE.md) — Sample-Bias Detection im Detail
- [Live Bias Detection](../LIVE_BIAS_DETECTION.md) — Echtzeit-Bias-Checks
- [Strategy Configuration](../../strategies/README.md) — Validation-Parameter
