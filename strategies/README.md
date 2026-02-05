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

### features - Feature-Gruppen & Selection

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `preferred_groups` | array | alle Gruppen | Feature-Gruppen die getestet werden |
| `feature_selection` | string | `"boruta"` | Feature-Selection Methode |
| `max_features` | int | `0` | Max Features pro Modell (0 = kein Limit) |

**feature_selection - Optionen:**

| Methode | Beschreibung | Limit |
|---------|--------------|-------|
| `boruta` | Findet alle statistisch relevanten Features | Optional (max_features) |
| `boruta_plateau` | Boruta + Plateau-Validierung für Stabilität | Optional (max_features) |
| `importance_based` | Top-5 Features nach Importance (Legacy) | Fest: 5 |

**max_features:** Begrenzt die Anzahl der Features auf die Top-N nach Importance. Empfohlen: 20-30 um Overfitting bei großen Feature-Gruppen (macro, macro_vol) zu vermeiden.

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

### preprocessing - Daten-Preprocessing

Preprocessing wird **vor** Feature-Berechnung auf OHLC-Daten angewendet.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `preprocessing` | array | `[]` | Liste der Preprocessing-Plugins |
| `preprocessing_params` | object | `{}` | Parameter pro Plugin |

**Verfügbare Plugins:**

#### fractional_diff - Fractional Differentiation

Macht Zeitreihen stationär unter Beibehaltung von Memory (nach López de Prado).

**Parameter:**
- `auto_d` (bool): Automatische d-Optimierung via ADF-Test (default: `true`)
- `default_d` (float): Fallback d-Wert wenn `auto_d=false` (default: `0.4`)
  - d=0: Keine Transformation (original)
  - d=1: Volle Differentiation (verliert Memory)
  - d=0.3-0.5: Optimal für Trading (stationär + Memory)
- `columns` (list): Zu transformierende Spalten (default: `["O", "H", "L", "C"]`)

**Wann nützlich:**
- Bei nicht-stationären Zeitreihen (Trends, Mean-Reversion)
- Verbessert ML-Modell-Performance durch stationäre Features
- Besonders wertvoll bei längeren Lookback-Perioden

**Beispiel:**

```json
{
  "preprocessing": ["fractional_diff"],
  "preprocessing_params": {
    "fractional_diff": {
      "auto_d": true,
      "default_d": 0.4,
      "columns": ["O", "H", "L", "C"]
    }
  }
}
```

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

### exit_strategy - Exit-Strategie

Definiert wie TP/SL-Distanzen berechnet werden.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `mode` | string | `"fixed"` | Exit-Modus: `"fixed"` oder `"atr_based"` |

**Mode-Optionen:**

| Mode | Beschreibung | Grid-Werte |
|------|--------------|------------|
| `fixed` | TP/SL als Spread-Multiplikatoren (konstant) | Große Zahlen: 10-100 |
| `atr_based` | TP/SL als ATR-Multiplikatoren (dynamisch) | Kleine Zahlen: 0.5-5.0 |

**ATR-Based Parameter:**

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `atr_period` | int | `14` | ATR-Periode für Volatilitätsberechnung |
| `min_tp_pips` | int | `10` | Mindest-TP in Spread-Multiples (Spread-Schutz) |
| `min_sl_pips` | int | `15` | Mindest-SL in Spread-Multiples (Spread-Schutz) |

**Beispiel - Fixed Exit (Default):**

```json
"exit_strategy": {
  "mode": "fixed"
}
```

Grid-Werte interpretiert als Spread-Multiplikatoren:
- `tp: 40` = 40 × Spread
- `sl: 30` = 30 × Spread

**Beispiel - ATR-Based Exit:**

```json
"exit_strategy": {
  "mode": "atr_based",
  "atr_based": {
    "atr_period": 14,
    "min_tp_pips": 10,
    "min_sl_pips": 15
  }
}
```

Grid-Werte interpretiert als ATR-Multiplikatoren:
- `tp: 1.5` = 1.5 × ATR (dynamisch pro Bar)
- `sl: 1.0` = 1.0 × ATR (dynamisch pro Bar)

**Wichtig:** Bei beiden Modi werden Spread und Slippage berücksichtigt!

---

### grids - TP/SL/CT Grid-Search

Definiert die zu testenden Take-Profit, Stop-Loss und Confidence-Threshold Werte pro Asset-Klasse.

**Die Interpretation der `tp`/`sl`-Werte hängt vom `exit_strategy.mode` ab:**

| Mode | tp/sl Interpretation | Typische Werte |
|------|---------------------|----------------|
| `fixed` | Spread-Multiplikatoren | 10, 20, 30, 50, 80, 100 |
| `atr_based` | ATR-Multiplikatoren | 0.5, 1.0, 1.5, 2.0, 3.0 |

**Asset-Klassen:** `FOREX`, `INDEX`, `COMMODITY`, `CRYPTO`

| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| `tp` | array | Take-Profit Werte (Interpretation je nach exit_strategy) |
| `sl` | array | Stop-Loss Werte (Interpretation je nach exit_strategy) |
| `ct` | array[float] | Confidence Threshold (0.50-1.00) |
| `timeout_bars` | array | Trade-Timeout in Bars (`[null, 24, 48]` = kein Timeout, 24h, 48h) |

**Separate Long/Short CT (optional):**

| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| `long_ct` | array[float] | CT nur für Long-Trades |
| `short_ct` | array[float] | CT nur für Short-Trades |

**Beispiel - Fixed Exit Grid:**

```json
"exit_strategy": {
  "mode": "fixed"
},
"grids": {
  "FOREX": {
    "tp": [15, 20, 25, 30, 40, 50, 60, 80],
    "sl": [15, 20, 25, 30, 40, 50, 60, 80],
    "ct": [0.50, 0.55, 0.60, 0.65, 0.70]
  }
}
```

**Beispiel - ATR-Based Exit Grid:**

```json
"exit_strategy": {
  "mode": "atr_based",
  "atr_based": {
    "atr_period": 14,
    "min_tp_pips": 10,
    "min_sl_pips": 15
  }
},
"grids": {
  "FOREX": {
    "tp": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "sl": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "ct": [0.50, 0.55, 0.60, 0.65],
    "timeout_bars": [null, 24, 48, 96]
  }
}
```

**Beispiel - Swing Trading Grid (Fixed, große TP/SL):**

```json
"grids": {
  "FOREX": {
    "tp": [100, 150, 200, 300, 500, 750, 1000],
    "sl": [50, 75, 100, 150, 200, 300],
    "ct": [0.55, 0.60, 0.65, 0.70, 0.75]
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

