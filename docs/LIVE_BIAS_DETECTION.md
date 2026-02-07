# Live Bias Detection

## Übersicht

Das System führt **nach jedem prozessierten Asset** automatisch Bias-Checks durch und gibt sofort Feedback.

## Features

### 1. Real-Time Checks während Optimization

Nach jedem Asset werden folgende Checks durchgeführt:

- **Mean Bias Ratio**: Durchschnitt über alle Folds (sollte ~1.0 sein)
- **Extreme Folds**: Einzelne Folds mit >2.0x ratio
- **Win-Rate Konsistenz**: Std-Dev über Folds
- **Unrealistic Win-Rate**: Vergleich mit Break-Even
- **Trade Count**: Genug Trades für Signifikanz (≥500)

### 2. Output während des Runs

**Bei einem OK Asset:**
```
✓ EURUSD: No bias detected (mean=1.02x, std_wr=4.5%)
```

**Bei Warnings:**
```
⚠️  GBPUSD: Warnings
    - Mean bias ratio 1.35x > 1.3x (monitor)
    - 1/8 folds have >2.0x bias: ['2.15']
```

**Bei kritischen Issues:**
```
🚨 AUDUSD: BIAS DETECTED!
    - Mean bias ratio 1.52x > 1.5x (systematic bias!)
    - Unrealistic WR: 78.5% (breakeven=66.7%, excess=11.8%)
    Bias Ratios: ['0.95x', '1.08x', '2.45x', '1.15x', '0.92x', '1.88x', '1.05x', '0.98x']
```

### 3. Ergebnisse in JSON

Jede `grid_details/{SYMBOL}.json` enthält jetzt:

```json
{
  "symbol": "EURUSD",
  "status": "ok",
  "walk_forward": {
    "n_folds": 8,
    "mean_bias_ratio": 1.02,
    "bias_ratios": [0.95, 1.08, 0.92, 1.12, 0.98, 1.05, 0.89, 1.01],
    ...
  },
  "bias_check": {
    "symbol": "EURUSD",
    "bias_check": "ok",
    "mean_bias_ratio": 1.02,
    "bias_ratios": [0.95, 1.08, 0.92, 1.12, 0.98, 1.05, 0.89, 1.01],
    "mean_win_rate": 0.695,
    "std_win_rate": 0.045,
    "total_trades": 1200,
    "issues": [],
    "warnings": []
  }
}
```

## Thresholds

| Metrik | OK | Warning | Critical |
|--------|-----|---------|----------|
| Mean Bias Ratio | <1.3x | 1.3-1.5x | >1.5x |
| Einzelne Folds | <2.0x | 2.0-3.0x | >3.0x |
| WR Std-Dev | <0.10 | 0.10-0.15 | >0.15 |
| WR Excess | <10% | 10-15% | >15% |
| Total Trades | >500 | 300-500 | <300 |

## System-Wide Check

Am Ende aller Assets wird ein System-wide Check durchgeführt:

```
================================================================================
SYSTEMATIC BIAS CHECK
================================================================================
✓ System OK: Only 1/9 assets biased (<10%)
  OK: 7/9
  Warnings: 1/9
  Biased: 1/9
================================================================================
```

**Thresholds:**
- **OK**: <10% der Assets biased
- **WARNING**: 10-20% der Assets biased
- **CRITICAL**: >20% der Assets biased (systematisches Problem!)

## Manueller Check

Du kannst die Bias-Checks auch manuell auf bestehenden Results laufen lassen:

```bash
# Auf latest results
python scripts/check_bias_on_results.py

# Auf spezifischem Directory
python scripts/check_bias_on_results.py test_results/20260207_123456/grid_details
```

## Was tun bei Bias?

### Einzelnes Asset biased (1-2 von 9)
→ **Normal**, kann Zufall sein
→ **Action**: Asset beobachten, ggf. mehr Daten sammeln

### Mehrere Assets biased (3-4 von 9)
→ **Warning**, könnte systematisch sein
→ **Action**:
  - Prüfe ob bestimmte Feature Groups betroffen
  - Prüfe ob bestimmte Zeiträume betroffen
  - Erhöhe Test-Set Size (4000 → 6000 bars)

### Viele Assets biased (>5 von 9)
→ **CRITICAL**, systematisches Problem!
→ **Action**:
  1. Prüfe Lookahead Bias in Feature Calculation
  2. Prüfe Data Leakage in Preprocessing
  3. Prüfe Walk-Forward Implementation
  4. Run Lookahead-Tests: `pytest tests/test_no_bias_in_system.py::TestNoLookaheadBias -v`

## Code-Referenzen

- **Bias Checks**: [src/fwbg/optimization/bias_checks.py](../src/fwbg/optimization/bias_checks.py)
- **Integration**: [src/fwbg/optimization/process.py](../src/fwbg/optimization/process.py#L596-L602)
- **Manual Check**: [scripts/check_bias_on_results.py](../scripts/check_bias_on_results.py)
- **Tests**: [tests/test_no_bias_in_system.py](../tests/test_no_bias_in_system.py)

## Example Output

Nach 10h Run würdest du sehen:

```
Processing EURUSD...
✓ EURUSD: No bias detected (mean=1.02x, std_wr=4.5%)

Processing GBPUSD...
⚠️  GBPUSD: Warnings
    - Mean bias ratio 1.35x > 1.3x (monitor)

Processing AUDUSD...
🚨 AUDUSD: BIAS DETECTED!
    - Mean bias ratio 1.52x > 1.5x (systematic bias!)

...

================================================================================
SYSTEMATIC BIAS CHECK
================================================================================
🚨 SYSTEMATIC BIAS! 5/9 assets (56%) have mean_bias >1.5x
  OK: 2/9
  Warnings: 2/9
  Biased: 5/9

Biased Assets:
  - AUDUSD: 1.52x (ratios: ['0.95x', '1.08x', '2.45x', ...])
  - EURCAD: 1.78x (ratios: ['1.15x', '1.92x', '1.65x', ...])
  ...
================================================================================
```

→ In diesem Fall: **STOP** und untersuche systematisches Problem!
