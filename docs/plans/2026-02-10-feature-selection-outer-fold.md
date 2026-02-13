# Feature Selection auf Outer Fold verschieben

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Feature Selection (Boruta) einmal pro Feature-Gruppe pro Outer Fold statt pro Grid-Kombination ausführen.

**Architecture:** Feature Selection wird aus `run_inner_cv` nach `_process_feature_group` verschoben. Die selektierten Features werden einmal berechnet und an alle Grid-Kombinationen durchgereicht. Dies reduziert die Anzahl der Boruta-Läufe von ~2560 pro Outer Fold auf ~10 (Anzahl Feature-Gruppen).

**Tech Stack:** Python, XGBoost, Boruta Feature Selection

---

## Problem-Analyse

**Aktueller Flow:**
```
process.py: Outer Fold Loop
  └─► grid_search.py: _process_feature_group (pro Feature-Gruppe)
        └─► Loop über TP/SL-Kombinationen (256 Combos)
              └─► nested_cv.py: run_inner_cv
                    └─► Loop über Inner Folds (5 Folds)
                          └─► select_features_from_fold() ← HIER IST BORUTA (FALSCH!)
```

**Performance-Impact:**
- 10 Feature-Gruppen × 256 TP/SL-Kombos = 2560 Boruta-Läufe pro Outer Fold
- 8 Outer Folds = **20.480 Boruta-Läufe pro Asset**
- Bei 5 Assets = 102.400 Boruta-Läufe → **74+ Stunden!**

**Gewünschter Flow:**
```
process.py: Outer Fold Loop
  └─► grid_search.py: _process_feature_group (pro Feature-Gruppe)
        └─► select_features_for_group() ← BORUTA HIER (1x pro Feature-Gruppe)
        └─► Loop über TP/SL-Kombinationen (256 Combos)
              └─► nested_cv.py: run_inner_cv (NUR Training/Evaluation, keine Feature Selection)
```

**Erwartete Verbesserung:**
- 10 Feature-Gruppen × 1 Boruta = 10 Boruta-Läufe pro Outer Fold
- 8 Outer Folds = **80 Boruta-Läufe pro Asset**
- Reduktion um Faktor **256x**

---

## Task 1: Neue Funktion `select_features_for_group` erstellen

**Files:**
- Modify: `src/fwbg/optimization/grid_search.py`

**Step 1: Schreibe die neue Funktion nach den Imports**

Nach Zeile ~20 (nach den Imports) einfügen:

```python
def select_features_for_group(
    inner_folds: list,
    group_features: list,
    ctx,
    sym: str,
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """
    Führt Feature Selection einmal pro Feature-Gruppe durch.

    Verwendet den ERSTEN Inner Fold für Feature Selection.
    Dies passiert VOR dem Grid-Search und reduziert Boruta-Läufe drastisch.

    Args:
        inner_folds: Liste von (train_df, val_df) Tuples
        group_features: Features dieser Gruppe
        ctx: SimulationContext
        sym: Symbol für Logging

    Returns:
        Tuple von (selected_features_long, selected_features_short)
    """
    from .nested_cv import select_features_from_fold, compute_targets

    if not inner_folds or len(group_features) < 3:
        return None, None

    # Verwende ersten Inner Fold für Feature Selection
    train_df, _ = inner_folds[0]

    # Berechne Targets mit Default TP/SL (Median der Grid-Werte)
    # Feature Selection ist unabhängig von TP/SL!
    default_tp = ctx.grid_tp[len(ctx.grid_tp) // 2] if ctx.grid_tp else 20
    default_sl = ctx.grid_sl[len(ctx.grid_sl) // 2] if ctx.grid_sl else 30

    targets_long, targets_short, has_long, has_short = compute_targets(
        train_df, default_tp, default_sl, ctx, timeout_bars=None
    )

    selected_long = None
    selected_short = None

    if has_long:
        selected_long, _ = select_features_from_fold(
            train_df, targets_long, group_features, ctx.min_trades,
            feature_selection=ctx.feature_selection,
            max_features=ctx.max_features,
            min_z_score=ctx.min_z_score,
        )
        if selected_long:
            log(2, f"  Feature Selection (Long): {len(selected_long)} Features ausgewählt", sym)

    if has_short:
        selected_short, _ = select_features_from_fold(
            train_df, targets_short, group_features, ctx.min_trades,
            feature_selection=ctx.feature_selection,
            max_features=ctx.max_features,
            min_z_score=ctx.min_z_score,
        )
        if selected_short:
            log(2, f"  Feature Selection (Short): {len(selected_short)} Features ausgewählt", sym)

    return selected_long, selected_short
```

**Step 2: Import Tuple hinzufügen**

Am Anfang der Datei, bei den Imports:

```python
from typing import Tuple, Optional, List
```

**Step 3: Commit**

```bash
git add src/fwbg/optimization/grid_search.py
git commit -m "feat: add select_features_for_group function"
```

---

## Task 2: Feature Selection in `_process_feature_group` aufrufen

**Files:**
- Modify: `src/fwbg/optimization/grid_search.py:145-285`

**Step 1: Feature Selection VOR dem Grid-Loop aufrufen**

In `_process_feature_group`, nach Zeile 217 (nach `grid_offset = fg_idx * grid_per_fg`) einfügen:

```python
    # === FEATURE SELECTION (einmal pro Feature-Gruppe) ===
    # Boruta läuft hier EINMAL statt für jede TP/SL-Kombination
    selected_features_long, selected_features_short = select_features_for_group(
        inner_folds, group_features, ctx, sym
    )

    if not selected_features_long and not selected_features_short:
        log(2, f"  Feature-Gruppe '{feature_group}': Keine Features selektiert - übersprungen", sym)
        return [], []

    # Reduzierte Feature-Liste für Grid-Search
    effective_features = selected_features_long or selected_features_short or group_features
    log(2, f"  Feature-Gruppe '{feature_group}': {len(effective_features)} selektierte Features für Grid-Search", sym)
```

**Step 2: Selektierte Features an Kombinationen übergeben**

Ändere die Combo-Erstellung (Zeile ~240-245) um die selektierten Features zu übergeben:

```python
            for timeout_bars in timeout_values:
                combos.append((
                    tp, sl, timeout_bars, combo_idx,
                    group_features, inner_folds, ctx, regime_config,
                    feature_group, grid_offset, total_grid_combos, inner_df,
                    selected_features_long, selected_features_short  # NEU
                ))
                combo_idx += 1
```

**Step 3: Commit**

```bash
git add src/fwbg/optimization/grid_search.py
git commit -m "feat: call feature selection once per feature group"
```

---

## Task 3: Wrapper-Funktion anpassen

**Files:**
- Modify: `src/fwbg/optimization/grid_search.py:95-143`

**Step 1: `_process_tp_sl_combo_wrapper` erweitern**

Die Funktion muss die neuen Parameter entpacken:

```python
def _process_tp_sl_combo_wrapper(args):
    """Wrapper für sequentielle TP/SL-Kombination Verarbeitung."""
    (tp, sl, timeout_bars, combo_idx,
     group_features, inner_folds, ctx, regime_config,
     feature_group, grid_offset, total_grid_combos, inner_df,
     selected_features_long, selected_features_short) = args  # ERWEITERT

    from .nested_cv import compute_targets_cached, slice_targets_for_fold

    global_grid_pos = grid_offset + combo_idx + 1

    # Berechne Targets für diese Kombination
    cached_targets = None
    has_preprocessing = ctx.preprocessing and len(ctx.preprocessing) > 0

    if inner_df is not None and not has_preprocessing:
        full_targets_long, full_targets_short = compute_targets_cached(
            inner_df, tp, sl, ctx, timeout_bars,
            exit_strategy_mode=ctx.exit_strategy,
        )
        cached_targets = {}
        for fold_idx, (train_df_fold, _) in enumerate(inner_folds):
            targets_long_fold = slice_targets_for_fold(full_targets_long, train_df_fold.index, inner_df.index)
            targets_short_fold = slice_targets_for_fold(full_targets_short, train_df_fold.index, inner_df.index)
            cached_targets[fold_idx] = (targets_long_fold, targets_short_fold)

    candidate, grid_result = _process_single_grid_combo(
        tp, sl, timeout_bars,
        group_features, inner_folds, ctx, regime_config,
        global_grid_pos, total_grid_combos,
        cached_targets=cached_targets,
        selected_features_long=selected_features_long,  # NEU
        selected_features_short=selected_features_short,  # NEU
    )

    return candidate, grid_result, combo_idx
```

**Step 2: Commit**

```bash
git add src/fwbg/optimization/grid_search.py
git commit -m "feat: pass pre-selected features to grid combo wrapper"
```

---

## Task 4: `_process_single_grid_combo` anpassen

**Files:**
- Modify: `src/fwbg/optimization/grid_search.py:23-93`

**Step 1: Neue Parameter hinzufügen**

```python
def _process_single_grid_combo(
    tp: int,
    sl: int,
    timeout_bars,
    group_features: list,
    inner_folds: list,
    ctx,
    regime_config: dict,
    global_grid_pos: int,
    total_grid_combos: int,
    cached_targets=None,
    selected_features_long: list = None,  # NEU
    selected_features_short: list = None,  # NEU
) -> tuple:
```

**Step 2: Features an `run_inner_cv` durchreichen**

```python
    inner_result = run_inner_cv(
        inner_folds, group_features, tp, sl, ctx,
        global_grid_pos, total_grid_combos,
        timeout_bars=timeout_bars,
        cached_targets=cached_targets,
        selected_features_long=selected_features_long,  # NEU
        selected_features_short=selected_features_short,  # NEU
    )
```

**Step 3: Commit**

```bash
git add src/fwbg/optimization/grid_search.py
git commit -m "feat: pass pre-selected features to run_inner_cv"
```

---

## Task 5: `run_inner_cv` anpassen - Feature Selection überspringen wenn Features übergeben

**Files:**
- Modify: `src/fwbg/optimization/nested_cv.py:819-950`

**Step 1: Neue Parameter hinzufügen**

```python
def run_inner_cv(
    inner_folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    group_features: List[str],
    tp: int,
    sl: int,
    ctx: SimulationContext,
    global_grid_pos: int,
    total_grid_combos: int,
    timeout_bars: int = None,
    cached_targets: Optional[Dict] = None,
    selected_features_long: Optional[List[str]] = None,  # NEU
    selected_features_short: Optional[List[str]] = None,  # NEU
) -> Dict[str, Any]:
```

**Step 2: Feature Selection Logik anpassen (Zeile ~931-947)**

Ersetze den bestehenden Feature Selection Block:

```python
        # Feature-Auswahl: Verwende übergebene Features oder mache lokale Selection
        if selected_features_long is None and selected_features_short is None:
            # Fallback: Feature Selection hier (sollte nicht mehr vorkommen)
            if fold_idx == 0:  # Nur auf erstem Fold
                if has_long:
                    selected_features_long, _ = select_features_from_fold(
                        train_df, targets_long, group_features, ctx.min_trades,
                        feature_selection=ctx.feature_selection,
                        max_features=ctx.max_features,
                        min_z_score=ctx.min_z_score,
                    )
                if has_short:
                    selected_features_short, _ = select_features_from_fold(
                        train_df, targets_short, group_features, ctx.min_trades,
                        feature_selection=ctx.feature_selection,
                        max_features=ctx.max_features,
                        min_z_score=ctx.min_z_score,
                    )
        # Wenn Features bereits übergeben wurden, verwende sie direkt (KEIN Boruta hier!)
```

**Step 3: Commit**

```bash
git add src/fwbg/optimization/nested_cv.py
git commit -m "feat: skip feature selection in run_inner_cv when features pre-selected"
```

---

## Task 6: Tests ausführen

**Step 1: Vorhandene Tests laufen lassen**

```bash
pytest tests/test_grid_search_feature_groups.py -v
```

**Step 2: Schnellen Integrationstest machen**

```bash
OPTIMIZER_LOG=2 timeout 120 fwbg --strategy-file strategies/exploration.json --asset-class COMMODITY --asset BRENT
```

Erwartete Ausgabe sollte zeigen:
- "Feature Selection (Long): X Features ausgewählt" (einmal pro Feature-Gruppe)
- Grid-Search sollte deutlich schneller durchlaufen

**Step 3: Commit nach erfolgreichen Tests**

```bash
git add -A
git commit -m "test: verify feature selection optimization works"
```

---

## Zusammenfassung der Änderungen

| Datei | Änderung |
|-------|----------|
| `grid_search.py` | Neue `select_features_for_group()` Funktion |
| `grid_search.py` | Feature Selection VOR Grid-Loop in `_process_feature_group` |
| `grid_search.py` | Selektierte Features durch Wrapper durchreichen |
| `nested_cv.py` | `run_inner_cv` akzeptiert pre-selektierte Features |
| `nested_cv.py` | Feature Selection wird übersprungen wenn Features übergeben |

**Performance-Verbesserung:**
- Vorher: 2560 Boruta-Läufe pro Outer Fold
- Nachher: 10 Boruta-Läufe pro Outer Fold
- **Faktor 256x schneller!**
