# Regime Cluster Plugin + Plugin Dependencies

## Ziel

Zwei Änderungen:
1. **`regime_cluster` Indikator-Plugin** — Composite Regime Score mit Quantil-Clustering statt binärer Schwellenwerte
2. **Plugin Dependencies** — `depends_on` Attribut mit topologischer Sortierung und Validierung

## Architektur-Prinzip

Der Kern (Simulation, Grid-Search, Walk-Forward, Target-Berechnung) läuft komplett ohne Plugins. Plugins erweitern nur. `depends_on` wirkt ausschließlich zwischen Plugins die in der aktuellen Pipeline konfiguriert sind.

---

## Teil 1: Plugin Dependencies

### BasePlugin-Erweiterung

`src/fwbg/pipeline/base.py`:

```python
class BasePlugin(ABC):
    name: str
    phase: PluginPhase
    version: str = "0.1.0"
    stateful: bool = False
    cacheable: bool = True
    depends_on: List[str] = []   # Plugin-Namen die vorher laufen müssen
```

### PipelineRunner-Änderungen

`src/fwbg/pipeline/runner.py` — in `_initialize()`:

1. Für jede Phase: alle konfigurierten Plugins sammeln
2. **Validierung**: Für jedes Plugin prüfen ob alle `depends_on`-Einträge in der Phase-Config vorhanden sind. Fehlend → `ValueError` mit klarer Meldung:
   ```
   ValueError: Plugin 'regime_cluster' depends on 'regime',
   but 'regime' is not in the pipeline. Either add 'regime'
   to pipeline.indicators or remove 'regime_cluster'.
   ```
3. **Topologische Sortierung** (Kahn's Algorithm): Plugins innerhalb einer Phase nach Dependencies ordnen
4. Bei Zyklen → `ValueError`

### Verhalten

- Bestehende Plugins ohne `depends_on` (leere Liste) sind unbeeinflusst
- Reihenfolge in der Strategy-JSON wird durch Auto-Sort überschrieben
- Phase-übergreifende Reihenfolge bleibt via `PHASE_ORDER`

---

## Teil 2: regime_cluster Plugin

### Konzept

Statt einzelner binärer Schwellen (ADX >= 25, VIX <= 30) einen Composite Score aus mehreren orthogonalen Marktstruktur-Inputs berechnen, per Rolling-Quantilen in Regime-Zonen einteilen (0/1/2), und als einzelne Column im Bitmask-Regime-Filter nutzen.

### Plugin-Eigenschaften

```python
@register_indicator("regime_cluster")
class RegimeClusterIndicator(BaseIndicator):
    name = "regime_cluster"
    version = "1.0.0"
    group = "regime"
    depends_on = ["regime", "volatility"]  # harte Dependencies
    stateful = False
    cacheable = True
```

### Parameter

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `zscore_window` | int | `200` | Rolling-Window für Z-Scoring der Inputs |
| `quantile_window` | int | `500` | Rolling-Window für Quantil-Berechnung |
| `n_regimes` | int | `3` | Anzahl Regime-Cluster (Quantil-Bins) |

### Kern-Inputs (5, orthogonal)

| Input | Vorzeichen | Misst |
|-------|-----------|-------|
| `regime_hurst_200` | + | Persistenz/Trending |
| `regime_entropy_100` | − (flip) | Vorhersagbarkeit |
| `regime_vr_200_5` | + (als vr−1.0) | Momentum vs Mean-Reversion |
| `vol_atr_pct_14_rank` | + | Volatilitäts-Level |
| `regime_hurst_divergence` | + | Regime-Shift-Signal |

### Optionale Inputs (wenn Spalte vorhanden)

| Input | Vorzeichen | Misst |
|-------|-----------|-------|
| `regime_risk_composite` | + | Makro Risk-On/Off |

### Berechnungsschritte

1. **Inputs sammeln** — Kern-Inputs aus df lesen, optionale nur wenn vorhanden
2. **Vorzeichenkorrektur** — Entropy flippen, VR zentrieren (vr − 1.0)
3. **Rolling Z-Score** — Pro Input: `(x - rolling_mean) / (rolling_std + 1e-10)`
4. **Composite Score** — Gleichgewichteter Durchschnitt aller z-scored Inputs
5. **Quantil-Clustering** — Rolling Quantile (window=500):
   - Score <= Q(1/3) → Cluster 0 (ungünstig)
   - Score <= Q(2/3) → Cluster 1 (neutral)
   - Score > Q(2/3) → Cluster 2 (günstig)
6. **shift_features()** — Alle Outputs um 1 Bar shiften (Lookahead Prevention)

### Score-Semantik

- **Hoher Score** = günstig für direktionales Trading (trending, persistent, niedrige Entropy)
- **Niedriger Score** = ungünstig (choppy, random, mean-reverting)

### Cluster-Labels (stabil, deterministisch)

- `0` = ungünstige Phase (unteres Drittel)
- `1` = neutrale Phase (mittleres Drittel)
- `2` = günstige Phase (oberes Drittel)

### Output-Features

| Feature | Typ | Beschreibung |
|---------|-----|--------------|
| `rclust_score` | float | Composite Regime Score |
| `rclust_cluster` | int (0/1/2) | Regime-Cluster Label |
| `rclust_score_chg` | float | Score-Änderung über 24 Bars |
| `rclust_n_inputs` | int | Anzahl verfügbarer Inputs |

### Grid-Nutzung

```json
"regime_filter_grid": {
  "condition_grids": [
    {"column": "rclust_cluster", "operator": ">=", "values": [null, 1, 2], "directions": 6, "else_directions": 0}
  ]
}
```

---

## Dateien

| Datei | Änderung |
|-------|----------|
| `src/fwbg/pipeline/base.py` | `depends_on: List[str] = []` zu BasePlugin |
| `src/fwbg/pipeline/runner.py` | Topologische Sortierung + Dependency-Validierung in `_initialize()` |
| `packages/fwbg-premium/.../indicators/regime_cluster/__init__.py` | NEU: Plugin |
| `packages/fwbg-premium/.../indicators/regime_cluster/manifest.json` | NEU: Manifest |
| `packages/fwbg-premium/manifest.json` | `regime_cluster` zu Plugin-Liste |
| `tests/test_regime_cluster.py` | NEU: Plugin-Tests |
| `tests/test_dependency_sort.py` | NEU: Dependency-Tests |
| `strategies/exploration*.json` | `regime_cluster` zu Indicators |
| `README.md` + `strategies/README.md` | Dokumentation |

## Tests

| Test | Prüft |
|------|-------|
| `test_score_computation` | 5 Inputs → Score = Durchschnitt der z-scored Werte |
| `test_cluster_labels` | Labels 0/1/2 korrekt via Quantile |
| `test_missing_optional_input` | Fehlender `regime_risk_composite` → Score aus 5 Inputs |
| `test_shift_features` | Alle Outputs um 1 Bar geshiftet |
| `test_n_inputs_diagnostic` | `rclust_n_inputs` korrekt |
| `test_dependency_validation` | Pipeline ohne `regime` → `ValueError` |
| `test_topological_sort` | Falsche JSON-Reihenfolge → korrekte Ausführung |
| `test_circular_dependency` | Zyklus → `ValueError` |
| `test_no_depends_on_unchanged` | Plugins ohne depends_on unverändert |
