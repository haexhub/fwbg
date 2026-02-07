# Robust Validation Framework - Benutzerhandbuch

## Problem: Sample Bias in Backtests

### Was ist Sample Bias?

Sample Bias tritt auf wenn die **Test-Periode zufällig besonders günstig** für eine Strategie ist, was zu überhöhten Erwartungen führt.

**Symptome:**
- Holdout-Performance **deutlich besser** als Inner Validation (>2x)
- Unrealistisch hohe Win-Rates (z.B. 83% bei RRR=0.5)
- Performance bricht auf neuen Daten zusammen

**Beispiel (AUDUSD):**
```
Inner Validation PnL: 55.4
Holdout PnL: 129.0  (2.33x besser!)
Win Rate: 83.1%
→ Sample Bias! Holdout-Periode war "lucky"
```

---

## Lösung: Robust Walk-Forward Validation

Statt **1x Holdout** (anfällig für Sample Bias):
```
[====== Training 80% ======][= Holdout 20% =]
```

Nutze **5x Walk-Forward Folds**:
```
Fold 0: [==== Train ====][= Test =]
Fold 1: [======== Train ========][= Test =]
Fold 2: [============ Train ============][= Test =]
Fold 3: [================ Train ================][= Test =]
Fold 4: [==================== Train ====================][= Test =]
```

**Vorteile:**
- ✅ Performance über **mehrere Zeitperioden** gemessen
- ✅ Sample Bias **automatisch erkannt** (wenn ein Fold extrem abweicht)
- ✅ Robustheit-Metriken (Std-Dev, Worst-Case, Consistency)
- ✅ Mehr Trades für statistische Signifikanz

---

## Verwendung

### 1. Robust Validation ausführen

```python
from fwbg.optimization.robust_validation import run_robust_validation
from fwbg.core.context import SimulationContext

# Setup
ctx = SimulationContext(
    symbol="EURUSD",
    asset_class="FOREX",
    point=0.0001,
    spread=0.0003,
    # ... weitere Parameter
)

strategy_config = {
    "tp": 10,
    "sl": 20,
    "ct": 0.65,
}

# Run Robust Validation (statt altem nested_cv)
result = run_robust_validation(
    df=your_dataframe_with_features,
    strategy_config=strategy_config,
    ctx=ctx,
    n_walk_forward_folds=5,  # 5 verschiedene Test-Perioden
    verbose=True
)

# Prüfe Robustheit
if result.is_robust(min_total_trades=500, max_win_rate_std=0.15):
    print("✓ Config ist robust!")
else:
    print("❌ Config ist NICHT robust")
```

### 2. Sample Bias Detection

```python
from test_sample_bias_detection import SampleBiasDetector

# Automatische Checks
bias_check = SampleBiasDetector.run_full_check(
    inner_val_pnl=55.4,
    holdout_pnl=129.0,
    win_rate=0.831,
    rrr=0.5,
    n_trades=195,
)

print(bias_check['verdict'])
# Output: "WARNING - High Risk of Sample Bias"
```

### 3. Integration in Workflow

```python
# Walk-Forward Validation (in process.py bereits integriert)
result = run_robust_validation(df, strategy_config, ctx)
# → Robust gegen Sample Bias durch multiple Test-Perioden
```

---

## Ergebnisse interpretieren

### RobustValidationResult

```python
result = run_robust_validation(...)

# Aggregierte Metriken über alle Folds
print(f"Mean Win Rate: {result.mean_win_rate*100:.1f}%")
print(f"Std Win Rate: {result.std_win_rate*100:.1f}%")
print(f"Range: [{result.min_win_rate*100:.1f}% - {result.max_win_rate*100:.1f}%]")
print(f"Total Trades: {result.total_trades}")
print(f"Consistency Score: {result.consistency_score:.2f}")  # 0-1, höher = besser

# Sample Bias Detection
if result.sample_bias_detected:
    print("⚠️  WARNING: Sample bias detected in some folds!")
    print(f"Holdout/Inner ratios: {result.holdout_vs_inner_ratios}")

# Robustheit-Check
if result.is_robust(
    min_total_trades=500,    # Minimum Trades gesamt
    max_win_rate_std=0.15,   # Maximum Std-Dev der Win-Rate
    max_bias_ratio=2.0       # Maximum Holdout/Inner Ratio
):
    print("✓ ROBUST")
else:
    print("❌ NOT ROBUST")
```

### Was sind gute Werte?

| Metrik | Robust | Grenzfall | Problematisch |
|--------|---------|-----------|---------------|
| Total Trades | >500 | 300-500 | <300 |
| Win-Rate Std-Dev | <0.10 | 0.10-0.15 | >0.15 |
| Consistency Score | >0.80 | 0.60-0.80 | <0.60 |
| Holdout/Inner Ratio | 0.7-1.3 | 1.3-2.0 | >2.0 |

**Beispiel - Robust:**
```
Mean Win Rate: 68.5% ± 8.2%
Total Trades: 850
Consistency: 0.85
Holdout/Inner: [0.95, 1.12, 0.88, 1.05, 0.92]
→ ✓ ROBUST
```

**Beispiel - Nicht Robust (Sample Bias):**
```
Mean Win Rate: 75.2% ± 18.3%
Total Trades: 220
Consistency: 0.42
Holdout/Inner: [0.85, 3.20, 1.05, 0.75, 1.15]
→ ❌ PROBLEMATISCH (Fold 1 hat 3.2x bias!)
```

---

## Automatische Tests

Die Sample Bias Detection läuft **automatisch** bei jedem Backtest wenn du das Robust Validation Framework nutzt.

### Tests die ausgeführt werden:

1. **Holdout vs Inner Bias Check**
   - Warnt wenn Holdout >2x besser als Inner
   - Deutet auf "lucky period" hin

2. **Unrealistic Win-Rate Check**
   - Berechnet Break-Even Win-Rate für gegebenes RRR
   - Warnt wenn Win-Rate unrealistisch hoch

3. **Insufficient Trades Check**
   - Warnt wenn <500 Trades gesamt
   - Zu wenig für statistische Signifikanz

4. **Fold Consistency Check**
   - Prüft Std-Dev über Folds
   - Warnt wenn Performance zu inkonsistent

### Pytest Integration

```bash
# Run unit tests
pytest tests/test_sample_bias_detection.py -v

# Erwartete Output:
# test_obvious_sample_bias PASSED
# test_normal_holdout_performance PASSED
# test_unrealistic_winrate PASSED
# test_full_check_audusd_case PASSED
# ...
```

---

## Best Practices

### DO ✓

- **Nutze 5+ Walk-Forward Folds** für robuste Validierung
- **Minimum 500 Trades** über alle Folds für Signifikanz
- **Prüfe Consistency Score** - sollte >0.70 sein
- **Monitore Holdout/Inner Ratio** - sollte 0.7-1.3 sein
- **Teste auf mehreren Assets** - nicht nur einem

### DON'T ✗

- **Nicht nur 1x Holdout** verwenden (Sample Bias!)
- **Nicht <300 Trades** akzeptieren (zu wenig Daten)
- **Nicht Outlier-Folds ignorieren** (zeigen Probleme!)
- **Nicht Win-Rates >80%** ohne kritische Prüfung glauben
- **Nicht Forward-Test überspringen** (finale Validierung!)

### Walk-Forward Validation nutzen

```python
# Walk-Forward Validation in process.py
result = run_robust_validation(df, config, ctx, n_walk_forward_folds=8)

# Prüfe Robustheit
if not result.is_robust():
    print("WARNING: Config nicht robust, weitere Tests nötig")

# Prüfe Sample Bias
if result.sample_bias_detected:
    print("WARNING: Sample bias erkannt, Ergebnisse mit Vorsicht")
```

---

## FAQ

### Q: Warum war AUDUSD so gut (83% WR)?

**A:** Sample Bias. Der Holdout-Zeitraum war zufällig extrem günstig:
- Inner PnL: 55.4
- Holdout PnL: 129.0 (2.33x besser!)
- Mit Robust Validation über 5 Folds: Durchschnitt wäre wahrscheinlich ~68-72%

### Q: Wie viele Folds sollte ich nutzen?

**A:**
- **Minimum:** 3 Folds (schnelle Tests)
- **Empfohlen:** 5 Folds (gute Balance)
- **Optimal:** 7-10 Folds (wenn genug Daten vorhanden)

### Q: Was wenn Consistency Score niedrig ist?

**A:** Niedrige Consistency (<0.60) bedeutet:
- Strategie ist regime-dependent (funktioniert nur in bestimmten Marktbedingungen)
- Parameters sind nicht robust
- → Mehr Feature Engineering oder andere Parameter nötig

### Q: Ersetzt das Forward-Testing?

**A:** **NEIN!** Walk-Forward Validation ist eine wichtige Qualitätssicherung, aber:
- Forward-Test auf NEUEN Daten ist IMMER nötig
- Walk-Forward reduziert Sample Bias, eliminiert ihn aber nicht vollständig
- Echte Validierung = Live/Paper Trading

---

## Nächste Schritte

1. ✅ **Installiere neue Tests:** `pytest tests/test_sample_bias_detection.py`
2. ✅ **Teste auf AUDUSD:** Lauf `test_robust_validation.py` Script
3. ✅ **Integriere in Workflow:** Ersetze `nested_cv` durch `run_robust_validation`
4. ✅ **Monitore Metriken:** Prüfe `is_robust()` und `sample_bias_detected`
5. ✅ **Forward-Test:** Teste beste Configs auf komplett neuen Daten

---

## Kontakt & Support

Bei Fragen oder Problemen:
- GitHub Issues: https://github.com/your-repo/issues
- Dokumentation: `/docs`
- Tests: `/tests/test_sample_bias_detection.py`
