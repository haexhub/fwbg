# Lookahead Bias Fixes - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all lookahead bias issues in indicator modules to prevent future data leakage.

**Architecture:** Alle Indikatoren MÜSSEN `shift_features()` am Ende verwenden. Rolling-Berechnungen die den aktuellen Wert mit einschließen müssen mit `.shift(1)` korrigiert werden. Double-Shift durch manuelles Shiften vor `shift_features()` muss entfernt werden.

**Tech Stack:** Python, pandas, ta-lib

---

## Hintergrund: Lookahead Bias Pattern

### Pattern 1: Double-Shift
```python
# FALSCH - Double Shift:
df[col] = df[col].shift(1)  # Manueller shift
features_df = shift_features(features, df.index)  # Nochmal shift = 2x!

# RICHTIG:
features_df = shift_features(features, df.index)  # Nur 1x shift
```

### Pattern 2: Rolling mit aktuellem Wert
```python
# FALSCH - Aktueller Wert ist im Fenster:
is_new_high = (df["H"] >= df["H"].rolling(20).max())  # H[i] vs max(H[i-19:i+1])

# RICHTIG - Nur vergangene Werte:
rolling_max = df["H"].rolling(20).max().shift(1)  # max(H[i-20:i])
is_new_high = (df["H"] >= rolling_max)  # H[i] vs max vergangener Werte
```

### Pattern 3: pct_change / diff ohne Shift
```python
# FALSCH - pct_change nutzt C[i] und C[i-1]:
features["x"] = df["C"].pct_change()  # Enthält C[i]!

# RICHTIG - Mit shift_features am Ende wird das automatisch geshiptet
# ODER wenn Zwischenergebnis genutzt wird:
returns = df["C"].pct_change().shift(1)  # Jetzt C[i-1] vs C[i-2]
```

---

## Task 1: Fix risk/__init__.py - Remove Double-Shift

**Problem:** Lines 273-278 shiften manuell alle `risk_*` und `corr_*` Spalten, aber die Funktion verwendet KEIN `shift_features()` am Ende. Das ist KEIN Double-Shift - aber inkonsistent mit dem Pattern in anderen Modulen.

**Analyse:** Das Modul schreibt direkt in df statt in features dict, daher manueller shift. Korrekte Lösung: Refactoring auf features dict + shift_features().

**Files:**
- Modify: `src/fwbg/builtins/indicators/risk/__init__.py:107-280`

**Step 1: Analyze the issue**

Das risk-Modul hat zwei Probleme:
1. Features werden direkt in `df` geschrieben statt in ein features dict
2. Am Ende wird manuell geshiptet (Zeilen 273-278)

Die korrekte Lösung ist das Refactoring auf das Standard-Pattern mit features dict.

**Step 2: Fix - Refactor to use shift_features pattern**

Ersetze den manuellen Shift-Block und nutze das standard Pattern:

```python
# Line 22: Add import
from fwbg.plugins.indicator import shift_features, safe_divide

# Lines 107-278: Refactor compute() to use features dict throughout
# instead of writing directly to df, then use shift_features at the end
```

**Step 3: Run tests**

```bash
python -m pytest tests/ -k "risk" -v
```

**Step 4: Commit**

```bash
git add src/fwbg/builtins/indicators/risk/__init__.py
git commit -m "fix(risk): refactor to use shift_features pattern"
```

---

## Task 2: Fix microstructure/__init__.py - Remove Double-Shift

**Problem:** Lines 144-148 shiften manuell alle `micro_*` Spalten, ABER das Modul nutzt KEIN `shift_features()`. Das ist KEIN Double-Shift aber inkonsistent.

**Analyse:** Ähnlich wie risk - direktes Schreiben in df, manueller shift am Ende.

**Files:**
- Modify: `src/fwbg/builtins/indicators/microstructure/__init__.py`

**Step 1: Analyze**

Das Modul:
- Schreibt Features direkt in df (z.B. `df["micro_wick_imbalance"] = ...`)
- Shiftet manuell am Ende (Lines 144-148)
- Nutzt NICHT shift_features()

**Step 2: Fix - Refactor to features dict + shift_features**

```python
# Change from:
df["micro_wick_imbalance"] = (upper_wick - lower_wick) / bar_range_safe

# To:
features = {}
features["micro_wick_imbalance"] = (upper_wick - lower_wick) / bar_range_safe
# ... all other features ...

# At end:
# REMOVE manual shift loop
features_df = shift_features(features, df.index)
return pd.concat([df, features_df], axis=1)
```

**Step 3: Run tests**

```bash
python -m pytest tests/ -k "micro" -v
```

**Step 4: Commit**

```bash
git add src/fwbg/builtins/indicators/microstructure/__init__.py
git commit -m "fix(microstructure): refactor to use shift_features pattern"
```

---

## Task 3: Fix macro_surprise/__init__.py - Remove Double-Shift

**Problem:** Lines 170-174 shiften manuell, aber kein shift_features() am Ende.

**Files:**
- Modify: `src/fwbg/builtins/indicators/macro_surprise/__init__.py`

**Step 1: Fix - Refactor to features dict + shift_features**

Gleiche Änderung wie bei microstructure.

**Step 2: Run tests**

```bash
python -m pytest tests/ -k "macro" -v
```

**Step 3: Commit**

```bash
git add src/fwbg/builtins/indicators/macro_surprise/__init__.py
git commit -m "fix(macro_surprise): refactor to use shift_features pattern"
```

---

## Task 4: Fix cross_features/__init__.py - BB Squeeze Self-Comparison

**Problem:** Lines 88-92 vergleichen `x.iloc[-1]` mit dem Percentile des gleichen Fensters:

```python
bb_width_percentile = bb_width.rolling(100).apply(
    lambda x: (x.iloc[-1] <= np.percentile(x, 20)) if len(x) > 0 else 0
)
```

Der aktuelle Wert (x.iloc[-1]) ist im Fenster enthalten und wird mit sich selbst verglichen.

**Files:**
- Modify: `src/fwbg/builtins/indicators/cross_features/__init__.py:88-92`

**Step 1: Fix - Exclude current value from percentile calculation**

```python
# FALSCH:
bb_width_percentile = bb_width.rolling(100).apply(
    lambda x: (x.iloc[-1] <= np.percentile(x, 20)) if len(x) > 0 else 0
)

# RICHTIG - Vergleiche aktuellen Wert mit historischem Percentile:
bb_width_hist_pct = bb_width.rolling(100).apply(
    lambda x: np.percentile(x[:-1], 20) if len(x) > 1 else np.nan
)
features["cross_bb_squeeze"] = (bb_width <= bb_width_hist_pct.shift(1)).astype(int)
```

**Step 2: Run tests**

```bash
python -m pytest tests/ -k "cross" -v
```

**Step 3: Commit**

```bash
git add src/fwbg/builtins/indicators/cross_features/__init__.py
git commit -m "fix(cross_features): fix BB squeeze self-comparison lookahead"
```

---

## Task 5: Fix structure/__init__.py - Event Detection Lookahead

**Problem 1:** Lines 219-223 vergleichen aktuellen H/L mit rolling max/min das den aktuellen Wert enthält:

```python
rolling_high = df["H"].rolling(period).max()  # Enthält H[i]!
is_new_high = (df["H"] >= rolling_high).astype(int)  # H[i] >= max inkl. H[i] = True!
```

**Problem 2:** Lines 181-183 nutzen `close_series.shift(window)` für path efficiency - das ist korrekt, ABER `close_series.diff()` nicht.

**Files:**
- Modify: `src/fwbg/builtins/indicators/structure/__init__.py`

**Step 1: Fix Event Detection**

```python
# FALSCH:
rolling_high = df["H"].rolling(period).max()
is_new_high = (df["H"] >= rolling_high).astype(int)

# RICHTIG - Rolling max OHNE aktuellen Wert:
rolling_high = df["H"].rolling(period).max().shift(1)
is_new_high = (df["H"] >= rolling_high).astype(int)
# Jetzt: H[i] >= max(H[i-period:i-1])
```

Gleiche Änderung für:
- `rolling_low` (Line 220)
- `atr_mean` für vol_spike (Line 250-251)

**Step 2: Run tests**

```bash
python -m pytest tests/ -k "structure" -v
```

**Step 3: Commit**

```bash
git add src/fwbg/builtins/indicators/structure/__init__.py
git commit -m "fix(structure): fix event detection rolling lookahead"
```

---

## Task 6: Fix multi_timeframe/__init__.py - H4 Trend Uses Current Close

**Problem:** Lines 69-76 nutzen `df["C"]` (aktueller Close) im H4 Trend:

```python
h4_close = df["C"]  # Aktueller Close!
h4_open = df["O"].shift(h4_bars - 1)
features["mtf_h4_trend"] = safe_divide(h4_close - h4_open, h4_range)
```

Das ist OK weil am Ende `shift_features()` genutzt wird - ABER die Rolling-Max/Min für h4_high/h4_low enthält den aktuellen Wert.

**Files:**
- Modify: `src/fwbg/builtins/indicators/multi_timeframe/__init__.py`

**Step 1: Fix H4 High/Low Rolling**

```python
# FALSCH:
h4_high = df["H"].rolling(h4_bars).max()  # Enthält H[i]
h4_low = df["L"].rolling(h4_bars).min()   # Enthält L[i]

# Das ist kompliziert: Für MTF wollen wir die LETZTEN 4 Bars inkl. aktueller.
# ABER: Da am Ende shift_features() genutzt wird, wird alles um 1 geshiptet.
# Also bei Bar i sehen wir Features von Bar i-1, die auf i-1 bis i-4 berechnet wurden.
# Das ist KORREKT!

# ABER: Die EMA-Berechnungen nutzen df["C"] direkt:
h4_ema = ta.trend.ema_indicator(df["C"], window=period * h4_bars)
features[f"mtf_h4_ema{period}_dist"] = (df["C"] - h4_ema) / df["C"]
# Da shift_features am Ende kommt, ist das OK.
```

Nach weiterer Analyse: Das Modul nutzt `shift_features()` korrekt am Ende. Die einzige potentielle Issue ist dass Rolling-Fenster den aktuellen Wert einschließen, aber da alles am Ende geshiptet wird, ist das Feature von Bar i = berechnet bis Bar i-1.

**KEIN FIX NÖTIG** - Das Modul ist korrekt.

---

## Task 7: Fix ichimoku/__init__.py - Verify Correctness

**Analyse:** Das Modul nutzt `shift_features()` am Ende (Line 154). Alle Features werden vorher berechnet und dann geshiptet. Das ist korrekt.

Die ta-lib Ichimoku-Berechnung ist internal und wir haben keinen Einfluss darauf. Die Features werden am Ende alle korrekt geshiptet.

**KEIN FIX NÖTIG** - Das Modul ist korrekt.

---

## Task 8: Fix price_action/__init__.py - Range Expansion Rolling Mean

**Problem:** Line 127-128 nutzen rolling mean für Range Expansion:

```python
avg_range = current_range.rolling(20).mean()  # Enthält aktuellen Range!
features["pa_range_expansion"] = safe_divide(current_range, avg_range)
```

Da `shift_features()` am Ende genutzt wird (Line 166), wird das Feature geshiptet. ABER: Das rolling(20).mean() enthält den aktuellen Wert.

Bei Bar i:
- avg_range[i] = mean(range[i-19:i+1])
- Nach shift: Feature[i] = berechnet für Bar i-1 = range[i-1] / mean(range[i-20:i])

Das ist KORREKT weil shift_features() am Ende kommt!

**KEIN FIX NÖTIG** - Das Modul ist korrekt.

---

## Task 9: Fix trend/__init__.py - Efficiency Ratio

**Analyse:** Lines 94-97:

```python
for period in [10, 20, 50]:
    change = abs(df["C"] - df["C"].shift(period))
    volatility = abs(df["C"].diff()).rolling(period).sum()
    features[f"trend_er_{period}"] = safe_divide(change, volatility)
```

- `df["C"].shift(period)` ist korrekt (vergangener Close)
- `df["C"].diff()` nutzt aktuellen Close
- Da `shift_features()` am Ende genutzt wird, ist das OK

**KEIN FIX NÖTIG** - Das Modul ist korrekt.

---

## Task 10: Fix distribution/__init__.py - Rolling Skew/Kurt

**Analyse:** Lines 63-67:

```python
returns = df["C"].pct_change()
for period in windows:
    features[f"dist_skew_{period}"] = returns.rolling(period).skew()
    features[f"dist_kurt_{period}"] = returns.rolling(period).kurt()
```

- `pct_change()` nutzt aktuellen Close
- `rolling().skew()` enthält aktuellen Return
- Da `shift_features()` am Ende genutzt wird (Line 108), wird alles korrekt geshiptet

**KEIN FIX NÖTIG** - Das Modul ist korrekt.

---

## Zusammenfassung der benötigten Fixes

Nach detaillierter Analyse:

| Modul | Problem | Fix Nötig? |
|-------|---------|------------|
| risk | Manueller shift statt shift_features | JA - Refactor |
| microstructure | Manueller shift statt shift_features | JA - Refactor |
| macro_surprise | Manueller shift statt shift_features | JA - Refactor |
| cross_features | BB Squeeze Self-Comparison | JA - Fix Lambda |
| structure | Event Rolling inkl. aktuell | JA - Add .shift(1) |
| multi_timeframe | Nutzt shift_features korrekt | NEIN |
| ichimoku | Nutzt shift_features korrekt | NEIN |
| price_action | Nutzt shift_features korrekt | NEIN |
| trend | Nutzt shift_features korrekt | NEIN |
| distribution | Nutzt shift_features korrekt | NEIN |

**5 Module brauchen Fixes, 5 sind korrekt.**

---

## Implementierungsreihenfolge

1. **Task 1-3: Refactor risk, microstructure, macro_surprise** auf shift_features Pattern
2. **Task 4: Fix cross_features** BB Squeeze
3. **Task 5: Fix structure** Event Detection Rolling

Nach jedem Fix: Tests laufen lassen um sicherzustellen dass keine Regression.

---

## Verifikation

Nach allen Fixes einen vollständigen Optimizer-Run mit einem Asset starten und prüfen ob die Ergebnisse realistischer sind (nicht mehr 88% Win-Rate).

```bash
timeout 120 python -m optimizer --asset EURUSD --strategy test_success_params --quick
```
