# Strategy Configuration Reference

Diese Dokumentation beschreibt alle verfügbaren Parameter für Strategy-Konfigurationsdateien.

## Grundstruktur

```json
{
  "name": "Strategy Name",
  "description": "Beschreibung der Strategie",
  "category": "trading_style",
  "tags": ["tag1", "tag2"],
  "hypothesis": "Was wir erwarten",
  "expected_outcome": "Erwartetes Ergebnis",

  "model": { ... },
  "features": { ... },
  "simulation": { ... },
  "validation": { ... },
  "grids": { ... },
  "filters": { ... },
  "assets": { ... },
  "resources": { ... }
}
```

---

## Parameter-Referenz

### Basis-Informationen

| Parameter | Typ | Beschreibung | Beispiel |
|-----------|-----|--------------|----------|
| `name` | string | Name der Strategie | `"Scalping"` |
| `description` | string | Kurzbeschreibung | `"Kurzfristige Trades"` |
| `tags` | array | Tags für `--tags X,Y` zum Filtern | `["scalping", "high_winrate"]` |
| `hypothesis` | string | Arbeitshypothese | `"Scalping sollte hohe WR haben"` |
| `expected_outcome` | string | Erwartetes Ergebnis | `"Win Rate > 70%"` |

---

### model - ML-Modell Konfiguration

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `architecture` | string | `"unified"` | Modell-Architektur (siehe unten) |
| `trade_directions` | array | `["long", "short"]` | Erlaubte Trade-Richtungen |
| `hyperparameters` | object | siehe unten | XGBoost Hyperparameter |

**architecture - Optionen:**
- `"unified"` - Ein Modell für Long und Short
- `"long_short_separate"` - Separate Modelle für Long und Short

**trade_directions - Optionen:**
- `["long", "short"]` - Beide Richtungen (Default)
- `["long"]` - Nur Long-Trades
- `["short"]` - Nur Short-Trades

**hyperparameters - Default-Werte:**
```json
{
  "n_estimators": 100,
  "max_depth": 5,
  "random_state": 42
}
```

**Beispiele:**

```json
// Standard Long/Short separate Modelle
"model": {
  "architecture": "long_short_separate"
}

// Long-only Strategie
"model": {
  "architecture": "unified",
  "trade_directions": ["long"]
}

// Schneller Test mit weniger Bäumen
"model": {
  "hyperparameters": {
    "n_estimators": 30,
    "max_depth": 3
  }
}
```

---

### features - Feature-Gruppen

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `preferred_groups` | array | alle Gruppen | Feature-Gruppen die getestet werden |

**Verfügbare Feature-Gruppen:**

| Gruppe | Beschreibung | Prefixes |
|--------|--------------|----------|
| `trend` | Trend Indikatoren - ADX, EMA, SMA, MACD, CCI, Aroon, Ichimoku | `trend_`, `ichi_` |
| `momentum` | Momentum Indikatoren - RSI, Stochastic, Williams %R, ROC, Ultimate Oscillator | `mom_` |
| `volatility` | Volatilität Indikatoren - Bollinger Bands, Keltner, Donchian, ATR | `vol_` |
| `price_action` | Price Action - Range Position, Higher Highs/Lower Lows, Body Ratio, Gaps | `pa_` |
| `time` | Zeit Features - Stunde, Wochentag, Monat, Quartal, Saisonalität | `time_`, `season_` |
| `macro` | Makro Indikatoren - VIX, Yields, DXY, Indices, Commodities, Sectors | `macro_` |
| `dynamics` | Dynamik & Lags - Indikator-Änderungen, Lags, Beschleunigung | `dyn_`, `lag_`, `accel_` |
| `mtf` | Multi-Timeframe - H4 aggregierte Features | `mtf_` |
| `cross` | Cross-Indikator - Kombinierte Signale aus mehreren Indikatoren | `cross_` |

**Kombinierte Feature-Gruppen:**

| Gruppe | Enthält | Beschreibung |
|--------|---------|--------------|
| `trend_momentum` | trend + momentum | Klassische technische Analyse Kombination |
| `macro_vol` | macro + volatility | Fundamentale + Volatilitäts-basierte Signale |
| `full_technical` | trend + momentum + volatility + price_action | Alle technischen Indikatoren ohne Makro/Zeit |

**WICHTIG - Redundanzen vermeiden:**

Die kombinierten Gruppen enthalten Basis-Gruppen. Nicht zusammen verwenden:
- `full_technical` enthält bereits `trend`, `momentum`, `volatility`, `price_action` und `trend_momentum`
- `macro_vol` enthält bereits `macro` und `volatility`
- `trend_momentum` enthält bereits `trend` und `momentum`

**Beispiele:**

```json
// Nur technische Indikatoren (EINE Gruppe reicht)
"features": {
  "preferred_groups": ["full_technical"]
}

// Makro + Dynamik (sinnvolle Kombination ohne Redundanz)
"features": {
  "preferred_groups": ["macro_vol", "dynamics"]
}

// Nur Momentum-basiert (für Mean Reversion)
"features": {
  "preferred_groups": ["momentum", "cross"]
}

// Zeit-basierte Strategie (Asian Session etc.)
"features": {
  "preferred_groups": ["time", "price_action", "volatility"]
}
```

**Hinweis:** Wenn `preferred_groups` nicht angegeben wird, werden ALLE Feature-Gruppen getestet (12 Gruppen). Das kann sinnvoll sein für explorative Runs, dauert aber entsprechend länger.

---

### simulation - Trade-Simulation

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `max_trade_bars` | int | `9999` | Maximale Trade-Dauer in Bars (1 Bar = 1 Stunde), 9999 = unbegrenzt |

**max_trade_bars - Wann setzen:**
- Default `9999` = Trades laufen bis TP oder SL erreicht wird (empfohlen)
- Nur explizit setzen wenn Timeout-Verhalten gewünscht ist:
  - `72` - 3 Tage (Scalping mit Timeout)
  - `120` - 5 Tage (Day Trading)
  - `240` - 10 Tage (Swing Trading)
  - `480` - 20 Tage (Position Trading)
- `480` - 20 Tage (Swing Trading)

**Beispiel:**

```json
// Swing Trading mit 20 Tagen max Haltezeit
"simulation": {
  "max_trade_bars": 480
}
```

---

### validation - Cross-Validation

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `folds` | int | `8` | Anzahl Walk-Forward Folds |
| `oos_size` | int | `4000` | Out-of-Sample Größe pro Fold |
| `holdout_ratio` | float | `0.20` | Holdout-Anteil für finale Validierung |

**Beispiel:**

```json
// Schneller Test mit weniger Folds
"validation": {
  "folds": 2,
  "oos_size": 500
}
```

---

### grids - TP/SL/CT Grid-Search

Definiert die zu testenden Take-Profit, Stop-Loss und Confidence-Threshold Werte pro Asset-Klasse.

**Asset-Klassen:** `FOREX`, `INDEX`, `COMMODITY`, `CRYPTO`

| Parameter | Typ | Beschreibung | Beispiel |
|-----------|-----|--------------|----------|
| `tp` | array[int] | Take-Profit in Spread-Multiples | `[20, 40, 60, 80]` |
| `sl` | array[int] | Stop-Loss in Spread-Multiples | `[20, 40, 60, 80]` |
| `ct` | array[float] | Confidence Threshold (0.50-1.00) | `[0.55, 0.60, 0.65, 0.70]` |

**Separate Long/Short CT (optional):**

| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| `long_ct` | array[float] | CT nur für Long-Trades |
| `short_ct` | array[float] | CT nur für Short-Trades |

**Beispiel - Standard Grid:**

```json
"grids": {
  "FOREX": {
    "tp": [15, 20, 25, 30, 40, 50, 60, 80],
    "sl": [15, 20, 25, 30, 40, 50, 60, 80],
    "ct": [0.50, 0.55, 0.60, 0.65, 0.70]
  },
  "INDEX": {
    "tp": [20, 30, 50, 70, 100, 150],
    "sl": [20, 30, 50, 70, 100, 150],
    "ct": [0.50, 0.55, 0.60, 0.65, 0.70]
  }
}
```

**Beispiel - Swing Trading Grid (große TP/SL):**

```json
"grids": {
  "FOREX": {
    "tp": [100, 150, 200, 300, 500, 750, 1000],
    "sl": [50, 75, 100, 150, 200, 300],
    "ct": [0.55, 0.60, 0.65, 0.70, 0.75]
  }
}
```

**Beispiel - High Confidence Grid:**

```json
"grids": {
  "FOREX": {
    "tp": [25, 35, 50, 70, 100],
    "sl": [25, 35, 50, 70, 100],
    "ct": [0.70, 0.75, 0.80, 0.85]
  }
}
```

---

### filters - Ergebnis-Filter

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `min_rrr` | float | `0.0` | Minimum Risk-Reward-Ratio |
| `min_trades` | int | `50` | Minimum Trades für Validität |

**Beispiele:**

```json
// Scalping (alle RRR erlaubt)
"filters": {
  "min_rrr": 0.0,
  "min_trades": 50
}

// Swing Trading (mindestens 1:1 RRR)
"filters": {
  "min_rrr": 1.0,
  "min_trades": 30
}

// Trend Following (mindestens 1.5:1 RRR)
"filters": {
  "min_rrr": 1.5,
  "min_trades": 30
}
```

---

### assets - Asset-Filterung (optional)

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `classes` | array | alle Klassen | Nur diese Asset-Klassen testen |

**Verfügbare Klassen:** `FOREX`, `INDEX`, `COMMODITY`, `CRYPTO`, `TEST`

**Beispiel:**

```json
// Nur Commodities testen
"assets": {
  "classes": ["COMMODITY"]
}

// Nur Forex und Indices
"assets": {
  "classes": ["FOREX", "INDEX"]
}
```

---

### resources - Ressourcen-Limits (optional)

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `max_cpu_percent` | float | `0.80` | Maximale CPU-Auslastung (0.0-1.0) |
| `min_free_ram_percent` | float | `0.25` | Minimum freier RAM (0.0-1.0) |
| `ram_per_worker_gb` | float | `3.0` | RAM pro Worker in GB |

**Beispiel:**

```json
// Ressourcen-schonender Test
"resources": {
  "max_cpu_percent": 0.5,
  "min_free_ram_percent": 0.3,
  "ram_per_worker_gb": 2.0
}
```

---

## Vollständige Beispiele

### Minimale Konfiguration (nutzt alle Defaults)

```json
{
  "name": "My Strategy",
  "description": "Meine Test-Strategie"
}
```

### Scalping Strategie

```json
{
  "name": "Scalping",
  "description": "Kurzfristige Trades mit kleinen TP/SL-Zielen",
  "category": "trading_style",
  "tags": ["scalping", "high_winrate"],
  "model": {
    "architecture": "long_short_separate"
  },
  "grids": {
    "FOREX": {
      "tp": [15, 20, 25, 30, 40, 50, 60, 80],
      "sl": [15, 20, 25, 30, 40, 50, 60, 80],
      "ct": [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]
    }
  },
  "filters": {
    "min_rrr": 0.0,
    "min_trades": 50
  }
}
```

### Swing Trading Strategie

```json
{
  "name": "Swing Trading",
  "description": "Längerfristige Trades mit großen TP/SL-Zielen",
  "category": "trading_style",
  "tags": ["swing", "trend_following"],
  "model": {
    "architecture": "long_short_separate"
  },
  "simulation": {
    "max_trade_bars": 480
  },
  "grids": {
    "FOREX": {
      "tp": [100, 150, 200, 300, 500, 750, 1000],
      "sl": [50, 75, 100, 150, 200, 300],
      "ct": [0.55, 0.60, 0.65, 0.70, 0.75]
    }
  },
  "filters": {
    "min_rrr": 1.0,
    "min_trades": 30
  }
}
```

### Mean Reversion Strategie

```json
{
  "name": "Mean Reversion",
  "description": "Handelt gegen Extrembewegungen",
  "category": "mean_reversion",
  "tags": ["mean_reversion", "contrarian"],
  "features": {
    "preferred_groups": ["momentum", "cross"]
  },
  "simulation": {
    "max_trade_bars": 72
  },
  "grids": {
    "FOREX": {
      "tp": [20, 30, 40, 50, 60],
      "sl": [30, 40, 50, 60, 80],
      "ct": [0.60, 0.65, 0.70, 0.75, 0.80]
    }
  },
  "filters": {
    "min_rrr": 0.0,
    "min_trades": 50
  }
}
```

### Long-Only Strategie

```json
{
  "name": "Long Only",
  "description": "Nur Long-Positionen",
  "model": {
    "architecture": "unified",
    "trade_directions": ["long"]
  },
  "grids": {
    "INDEX": {
      "tp": [20, 30, 50, 70, 100, 150],
      "sl": [20, 30, 50, 70, 100, 150],
      "ct": [0.50, 0.55, 0.60, 0.65, 0.70]
    }
  }
}
```

### Schneller Test

```json
{
  "name": "Quick Test",
  "category": "test",
  "model": {
    "architecture": "long_short_separate",
    "hyperparameters": {
      "n_estimators": 30,
      "max_depth": 3
    }
  },
  "validation": {
    "folds": 2,
    "oos_size": 500
  },
  "assets": {
    "classes": ["TEST"]
  },
  "resources": {
    "max_cpu_percent": 0.5,
    "ram_per_worker_gb": 2.0
  }
}
```

---

## Nicht mehr genutzte Parameter

Diese Parameter werden im Code **nicht mehr verwendet** und können ignoriert oder entfernt werden:

- `baseline_run` - nur Dokumentation
- `changes` - nur Dokumentation
- `notes` - nur Dokumentation
- `features.technical_indicators` - hat keinen Effekt
- `features.macro_indicators` - hat keinen Effekt
- `features.time_features` - hat keinen Effekt
- `features.multi_timeframe` - hat keinen Effekt
- `features.custom_features` - hat keinen Effekt
- `features.feature_selection` - hat keinen Effekt
- `simulation.tp_sl_basis` - immer "spread_multiple"
- `simulation.trailing_stop` - immer true
- `simulation.slippage_model` - immer "fixed"
- `simulation.regime_filter` - immer true
- `validation.method` - immer "walk_forward"
- `filters.min_annual_return` - hat keinen Effekt
- `model.type` - immer "xgboost"
