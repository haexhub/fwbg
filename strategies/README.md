# Strategy Configuration Reference

Alle verfügbaren Parameter für Strategy-Konfigurationsdateien.

## Grundstruktur

```json
{
  "name": "Strategy Name",
  "description": "Beschreibung der Strategie",
  "tags": ["tag1", "tag2"],
  "hypothesis": "Was wir erwarten",
  "expected_outcome": "Erwartetes Ergebnis",

  "pipeline": {
    "preprocessing": [...],
    "indicators": [...],
    "feature_selection": [...],
    "data_loading": [...]
  },

  "exit_strategy": "fixed",
  "exit_params": {},

  "model": { ... },
  "grids": { ... },
  "validation": { ... },
  "filters": { ... },
  "resources": { ... }
}
```

---

## pipeline - Feature-Pipeline

Die `pipeline`-Sektion konfiguriert die komplette Feature-Berechnung über das Plugin-System.

### pipeline.indicators - Indikator-Plugins

Jeder Indikator ist ein Plugin mit eigenem Namen und Parametern. Kurze Namen (`"trend"`) werden automatisch auf voll qualifizierte Namen (`"fwbg-core:trend"`) aufgelöst.

**Core Plugins (fwbg-core):**

| Plugin | Beschreibung | Prefixes |
|--------|--------------|----------|
| `trend` | ADX, EMA, SMA, MACD, CCI, Aroon, Supertrend, Efficiency Ratio | `trend_` |
| `momentum` | RSI, Stochastic, Williams %R, ROC | `mom_` |
| `volatility` | Bollinger Bands, ATR, Volatilitätsschätzer, Vol Compression, RV vs IV Spread | `vol_` |
| `price_action` | Range Position, Higher Highs/Lower Lows, Body Ratio, Gaps | `pa_` |
| `time_season` | Stunde, Wochentag, Monat, Quartal, Saisonalität | `time_`, `season_` |

**Premium Plugins (fwbg-premium):**

| Plugin | Beschreibung | Prefixes |
|--------|--------------|----------|
| `regime` | Hurst Exponent, Entropy, Variance Ratio | `regime_` |
| `structure` | FFT, Path Statistics, Convexity, Event Flow, VWAP | `struct_` |
| `risk` | Drawdown, CVaR, Volatility of Volatility, Correlations | `risk_` |
| `distribution` | Skewness, Kurtosis, Z-Score | `dist_` |
| `dynamics` | Indikator-Änderungen, Lags, Beschleunigung | `dyn_`, `lag_`, `accel_` |
| `multi_timeframe` | H4/D1/W1/Y1 Multi-Timeframe Features, Trend Alignment, Volatility Ratios | `mtf_` |
| `cross_features` | Kombinierte Signale, COT × Vol Interaction, Positioning Divergence | `cross_` |
| `ichimoku` | Ichimoku Cloud Komponenten | `ichi_` |
| `macro_surprise` | Makro-Überraschungen, Gap-Analyse | `macro_surprise_` |
| `microstructure` | Bar-Microstructure, Tick-Proxies | `micro_` |
| `market_regime` | Risk-On/Off Composite aus VIX, Credit, Equity, Treasury | `regime_vix_`, `regime_credit_`, `regime_risk_` |

**Beispiel:**

```json
"pipeline": {
  "indicators": [
    {"name": "trend", "params": {"adx_periods": [7, 14, 21], "ema_periods": [8, 21, 50, 100, 200]}},
    {"name": "momentum", "params": {"rsi_periods": [7, 14, 21]}},
    {"name": "volatility", "params": {"atr_periods": [7, 14, 21]}},
    {"name": "regime", "params": {"hurst_windows": [100, 200, 500]}},
    {"name": "price_action", "params": {}},
    {"name": "time_season", "params": {}}
  ]
}
```

Jedes Plugin akzeptiert `params: {}` für seine Default-Werte.

#### multi_timeframe - Parameter

Berechnet Features für höhere Timeframes (H4, D1, W1, Y1) aus H1-Daten via Rolling Windows.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `h4_bars` | int | `4` | Bars pro H4-Candle |
| `d1_bars` | int | `24` | Bars pro D1-Candle |
| `w1_bars` | int | `120` | Bars pro W1-Candle (5 × 24) |
| `ema_periods` | list | `[20, 50]` | EMA-Perioden für MTF-Berechnung |
| `include_yearly` | bool | `true` | Y1-Features berechnen (200d EMA, 52-Wochen Range) |

**Features nach Timeframe:**

| Gruppe | Features | Beschreibung |
|--------|----------|--------------|
| H4 | `mtf_h4_trend`, `mtf_h4_range_pos`, `mtf_h4_ema{N}_dist`, `mtf_h4_adx`, `mtf_h4_rsi`, `mtf_h4_atr_pct`, `mtf_h4_bb_pband` | 4-Stunden Trend, Range, Indikatoren |
| D1 | `mtf_d1_range_pos`, `mtf_d1_ema{N}_dist`, `mtf_d1_trend_strength` | Tages-Range, EMA-Distanzen, Trend-Stärke |
| W1 | `mtf_w1_range_pos`, `mtf_w1_ema{N}_dist`, `mtf_w1_trend_strength` | Wochen-Range, EMA-Distanzen, Trend-Stärke |
| Y1 | `mtf_y1_ema200d_dist`, `mtf_y1_52w_range_pos`, `mtf_y1_52w_high_dist`, `mtf_y1_52w_low_dist` | 200-Tage EMA, 52-Wochen Range |
| Alignment | `mtf_trend_alignment_h1h4`, `mtf_trend_alignment_h4d1`, `mtf_trend_alignment_d1w1`, `mtf_consensus`, `mtf_trend_strength` | Trend-Übereinstimmung zwischen Timeframes |
| Volatility | `mtf_vol_ratio_h1h4` | H1/H4 ATR-Verhältnis |
| Divergence | `mtf_rsi_divergence` | H1-H4 RSI-Divergenz |
| S/R | `mtf_d1_above_prev_high`, `mtf_d1_below_prev_low`, `mtf_d1_dist_to_high`, `mtf_d1_dist_to_low` | Tages-Support/Resistance |

```json
{"name": "multi_timeframe", "params": {"h4_bars": 4, "d1_bars": 24, "w1_bars": 120, "ema_periods": [20, 50], "include_yearly": true}}
```

---

### pipeline.preprocessing - Daten-Preprocessing

Preprocessing wird **vor** Feature-Berechnung auf OHLC-Daten angewendet. Indikatoren mit `benefits_from_stationary = True` werden nach dem Preprocessing berechnet, die anderen vorher (damit Targets auf Originaldaten basieren).

**Verfügbare Plugins:**

#### fractional_diff - Fractional Differentiation

Macht Zeitreihen stationär unter Beibehaltung von Memory (nach López de Prado).

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `auto_d` | bool | `true` | Automatische d-Optimierung via ADF-Test. **WARNUNG**: Verursacht Lookahead Bias! |
| `default_d` | float | `0.4` | Fixer d-Wert (0=keine Transformation, 1=volle Diff) |
| `columns` | list | `["O", "H", "L", "C"]` | Zu transformierende Spalten |

```json
"preprocessing": [
  {"name": "fractional_diff", "params": {"auto_d": false, "default_d": 0.4, "columns": ["O", "H", "L", "C"]}}
]
```

---

### pipeline.feature_selection - Feature-Selektion

Wählt die relevantesten Features pro Fold aus.

#### boruta

Findet alle statistisch relevanten Features via Shadow-Feature-Vergleich.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `max_features` | int | `0` | Max Features (0 = kein Limit) |
| `n_iter` | int | `5` | Boruta-Iterationen |
| `n_estimators` | int | `30` | Random Forest Bäume |
| `max_depth` | int | `4` | Max Baumtiefe |
| `min_z_score` | float | `0.5` | Mindest-Z-Score für Akzeptanz |

```json
"feature_selection": [
  {"name": "boruta", "params": {"max_features": 20, "n_iter": 5, "min_z_score": 0.5}}
]
```

---

### pipeline.data_loading - Externe Datenquellen

Lädt externe Daten (Makro, Zinsen, etc.) über das DataSource-System und berechnet abgeleitete Features.

**Architektur:** DataSource (I/O) → Orchestrator (Index-Alignment) → Plugin (Computation)

#### macro_data

Lädt Makro-Indikatoren (VIX, Yields, DXY, etc.) und berechnet Lookbacks, Derived Features und Zinsdifferenzen.

| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| `source` | string | Datenquelle (z.B. `"forexsb"`) |

Default-Konfiguration des Plugins umfasst 28 Makro-Indikatoren, Stunden-/Tages-Lookbacks und abgeleitete Features (Spread-Berechnungen, Ratios, Zinsdifferenzen).

```json
"data_loading": [
  {"name": "macro_data", "source": "forexsb"},
  {"name": "cot_positioning", "source": "forexsb"}
]
```

#### cot_positioning

Lädt CFTC Commitment of Traders Positioning-Daten und berechnet Z-Scores, Extreme-Flags und Wochen-Momentum.

| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| `source` | string | Datenquelle (z.B. `"forexsb"`) |

Erwartet `macro_cot_*`-Spalten im DataFrame (geladen via DataSource aus `COT_{SYMBOL}_DAY.csv`).

**Berechnete Features pro COT-Spalte:**
- `cot_{pair}_zscore` — 52-Wochen Z-Score der Netto-Position
- `cot_{pair}_extreme_long` / `cot_{pair}_extreme_short` — Extreme Positioning Flags (|z| > 2.0)
- `cot_{pair}_crowded` — Crowded Trade Flag (|z| > 1.5)
- `cot_{pair}_chg_{1,4,12,26}w` — Wochen-Momentum der Netto-Position

Alle Features werden um 1 Bar geshiftet (Lookahead Prevention).

```json
"data_loading": [
  {"name": "macro_data", "source": "forexsb"},
  {"name": "cot_positioning", "source": "forexsb"}
]
```

---

## model - ML-Modell Konfiguration

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `type` | string | `"xgboost"` | Modell-Typ |
| `architecture` | string | `"unified"` | `"unified"` oder `"long_short_separate"` |
| `trade_directions` | array | `["long", "short"]` | Erlaubte Trade-Richtungen |
| `hyperparameters` | object | siehe unten | XGBoost Hyperparameter |

**hyperparameters - Default-Werte:**
```json
{
  "n_estimators": 100,
  "max_depth": 5,
  "learning_rate": 0.1,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "random_state": 42
}
```

**Beispiel:**
```json
"model": {
  "type": "xgboost",
  "architecture": "long_short_separate",
  "trade_directions": ["long", "short"],
  "hyperparameters": {"n_estimators": 100, "max_depth": 4}
}
```

---

## exit_strategy / exit_params - Exit-Strategie

Definiert wie TP/SL-Distanzen berechnet werden.

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `exit_strategy` | string | `"fixed"` | `"fixed"` oder `"atr_based"` |
| `exit_params` | object | `{}` | Parameter für die Exit-Strategie |

### fixed (Default)

TP/SL als Spread-Multiplikatoren (konstant).

```json
"exit_strategy": "fixed",
"exit_params": {}
```

Grid-Werte: `tp: 40` = 40 × Spread, `sl: 30` = 30 × Spread

### atr_based

TP/SL als ATR-Multiplikatoren (dynamisch pro Bar).

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `atr_period` | int | `14` | ATR-Periode |
| `min_tp_pips` | int | `10` | Mindest-TP in Spread-Multiples |
| `min_sl_pips` | int | `15` | Mindest-SL in Spread-Multiples |

```json
"exit_strategy": "atr_based",
"exit_params": {"atr_period": 14, "min_tp_pips": 10, "min_sl_pips": 15}
```

Grid-Werte: `tp: 1.5` = 1.5 × ATR, `sl: 1.0` = 1.0 × ATR

---

## grids - TP/SL/CT Grid-Search

Definiert die zu testenden Take-Profit, Stop-Loss und Confidence-Threshold Werte pro Asset-Klasse.

**Asset-Klassen:** `FOREX`, `INDEX`, `COMMODITY`, `CRYPTO`

| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| `tp` | array | Take-Profit Werte (Interpretation je nach exit_strategy) |
| `sl` | array | Stop-Loss Werte (Interpretation je nach exit_strategy) |
| `ct` | array[float] | Confidence Threshold (0.50-1.00) |
| `timeout_bars` | array | Trade-Timeout in Bars (`[null, 24, 48]`) |
| `long_ct` | array[float] | Optional: CT nur für Long-Trades |
| `short_ct` | array[float] | Optional: CT nur für Short-Trades |
| `regime_filter_grid` | object | Optional: Regime-Filter Grid-Search |

### regime_filter_grid

Testet Regime-Filter-Kombinationen im Grid-Search. Jede Condition prüft eine DataFrame-Spalte gegen einen Schwellenwert und steuert per **Bitmask**, welche Trade-Richtungen erlaubt sind. `null` = kein Filter (Baseline).

#### Bitmask-Encoding

Jede Condition erzeugt pro Bar eine Bitmask (int8, 0-7), die bestimmt welche Richtungen gehandelt werden dürfen:

| Bit | Wert | Richtung |
|-----|------|----------|
| 2 | 4 | Long |
| 1 | 2 | Short |
| 0 | 1 | Sideways |

**Häufige Kombinationen:** `7` = alle, `6` = Long+Short (Standard), `4` = nur Long, `2` = nur Short, `0` = blockiert

#### Condition-Parameter

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `column` | string | — | DataFrame-Spalte |
| `operator` | string | — | Vergleichsoperator (`>=`, `<=`, `>`, `<`) |
| `values` | array | — | Schwellenwerte (`null` = kein Filter) |
| `directions` | int | `6` | Bitmask wenn Condition TRUE |
| `else_directions` | int | `0` | Bitmask wenn Condition FALSE |

#### Kombinationslogik

Mehrere Conditions werden per **Bitwise AND** kombiniert — ein Trade wird nur ausgeführt, wenn **alle** Conditions die entsprechende Richtung erlauben.

```json
"regime_filter_grid": {
  "condition_grids": [
    {"column": "trend_adx_14", "operator": ">=", "values": [null, 25], "directions": 6, "else_directions": 0},
    {"column": "macro_vix", "operator": "<=", "values": [null, 30], "directions": 6, "else_directions": 0},
    {"column": "regime_hurst_100", "operator": ">=", "values": [null, 0.45], "directions": 6, "else_directions": 0}
  ]
}
```

#### Richtungs-spezifische Filter

Mit unterschiedlichen `directions`/`else_directions`-Werten können Conditions richtungs-selektiv filtern:

```json
{"column": "trend_ema_50_dist", "operator": ">=", "values": [null, 0], "directions": 4, "else_directions": 2}
```

Hier: Preis über EMA50 → nur Longs (`4`), Preis unter EMA50 → nur Shorts (`2`).

**Beispiel - Fixed Exit Grid:**

```json
"grids": {
  "FOREX": {
    "tp": [5, 10, 15, 20, 30],
    "sl": [20, 30, 40, 50, 60],
    "ct": [0.5, 0.55, 0.6, 0.65],
    "regime_filter_grid": {
      "condition_grids": [
        {"column": "trend_adx_14", "operator": ">=", "values": [null, 25], "directions": 6, "else_directions": 0}
      ]
    }
  }
}
```

**Beispiel - ATR-Based Exit Grid:**

```json
"grids": {
  "FOREX": {
    "tp": [0.5, 1.0, 1.5, 2.0, 3.0],
    "sl": [0.5, 1.0, 1.5, 2.0],
    "ct": [0.50, 0.55, 0.60],
    "timeout_bars": [null, 24, 48, 96]
  }
}
```

---

## validation - Cross-Validation

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `method` | string | `"walk_forward"` | Validierungsmethode |
| `folds` | int | `8` | Anzahl Walk-Forward Folds |
| `oos_size` | int | `4000` | Out-of-Sample Größe pro Fold |
| `min_trades` | int | `50` | Minimum Trades für Validität |
| `n_inner_folds` | int | `3` | Innere CV-Folds für Feature-Selektion |
| `embargo_bars` | int | `100` | Embargo-Bars zwischen Train/Test (Purging) |
| `sample_weights` | bool | `true` | Trade-Duration-basierte Sample Weights |

```json
"validation": {
  "method": "walk_forward",
  "folds": 8,
  "oos_size": 4000,
  "n_inner_folds": 3,
  "embargo_bars": 100,
  "sample_weights": true
}
```

---

## filters - Ergebnis-Filter

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `min_rrr` | float | `0.0` | Minimum Risk-Reward-Ratio |
| `min_trades` | int | `50` | Minimum Trades |
| `min_annual_return` | float | `0` | Minimum annualisierte Rendite |
| `min_sharpe` | float | `0` | Minimum Sharpe Ratio |
| `max_drawdown` | float | `1.0` | Maximum Drawdown (1.0 = kein Limit) |

```json
"filters": {
  "min_rrr": 0,
  "min_trades": 30,
  "min_sharpe": 0,
  "max_drawdown": 1.0
}
```

---

## resources - Ressourcen-Limits

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `max_cpu_percent` | float | `0.80` | Maximale CPU-Auslastung (0.0-1.0) |
| `min_free_ram_percent` | float | `0.25` | Minimum freier RAM (0.0-1.0) |
| `ram_per_worker_gb` | float | `3.0` | RAM pro Worker in GB |
| `max_concurrent_assets` | int | `2` | Parallele Assets |
| `xgboost_n_jobs` | int | `0` | XGBoost Threads (0 = auto) |

```json
"resources": {
  "ram_per_worker_gb": 4.0,
  "min_free_ram_percent": 0.15,
  "max_cpu_percent": 0.95,
  "max_concurrent_assets": 2
}
```

---

## Vollständige Beispiele

### Exploration (alle Features)

```json
{
  "name": "Exploration",
  "pipeline": {
    "preprocessing": [
      {"name": "fractional_diff", "params": {"auto_d": false, "default_d": 0.4, "columns": ["O", "H", "L", "C"]}}
    ],
    "indicators": [
      {"name": "trend", "params": {"adx_periods": [7, 14, 21], "ema_periods": [8, 21, 50, 100, 200]}},
      {"name": "momentum", "params": {"rsi_periods": [7, 14, 21]}},
      {"name": "volatility", "params": {"atr_periods": [7, 14, 21]}},
      {"name": "regime", "params": {"hurst_windows": [100, 200, 500]}},
      {"name": "structure", "params": {}},
      {"name": "risk", "params": {"dd_windows": [50, 100, 200]}},
      {"name": "price_action", "params": {}},
      {"name": "time_season", "params": {}},
      {"name": "distribution", "params": {"windows": [20, 50, 100]}},
      {"name": "dynamics", "params": {}},
      {"name": "multi_timeframe", "params": {}},
      {"name": "cross_features", "params": {}},
      {"name": "ichimoku", "params": {}},
      {"name": "microstructure", "params": {}},
      {"name": "macro_surprise", "params": {}},
      {"name": "market_regime", "params": {"window": 50}}
    ],
    "feature_selection": [
      {"name": "boruta", "params": {"max_features": 20}}
    ],
    "data_loading": [
      {"name": "macro_data", "source": "forexsb"},
      {"name": "cot_positioning", "source": "forexsb"}
    ]
  },
  "exit_strategy": "fixed",
  "model": {"architecture": "long_short_separate"},
  "grids": {
    "FOREX": {
      "tp": [5, 10, 15, 20, 30],
      "sl": [20, 30, 40, 50, 60],
      "ct": [0.5, 0.55, 0.6, 0.65],
      "regime_filter_grid": {
        "condition_grids": [
          {"column": "trend_adx_14", "operator": ">=", "values": [null, 25], "directions": 6, "else_directions": 0},
          {"column": "macro_vix", "operator": "<=", "values": [null, 30], "directions": 6, "else_directions": 0}
        ]
      }
    }
  },
  "validation": {"folds": 8, "oos_size": 4000, "embargo_bars": 100, "sample_weights": true}
}
```

### Scalping (kurze Perioden, hohe Confidence)

```json
{
  "name": "Scalping",
  "pipeline": {
    "indicators": [
      {"name": "trend", "params": {"adx_periods": [5, 7, 14], "ema_periods": [5, 8, 13, 21, 50]}},
      {"name": "momentum", "params": {"rsi_periods": [5, 7, 14], "stoch_periods": [5, 9, 14]}},
      {"name": "volatility", "params": {"atr_periods": [5, 7, 14]}},
      {"name": "dynamics", "params": {"lookbacks": [2, 4, 8]}},
      {"name": "microstructure", "params": {"atr_period": 7, "rolling_window": 3}},
      {"name": "market_regime", "params": {"window": 50}}
    ],
    "feature_selection": [
      {"name": "boruta", "params": {"max_features": 25}}
    ],
    "data_loading": [
      {"name": "macro_data", "source": "forexsb"},
      {"name": "cot_positioning", "source": "forexsb"}
    ]
  },
  "exit_strategy": "fixed",
  "grids": {
    "FOREX": {
      "tp": [5, 8, 10, 15],
      "sl": [8, 10, 15, 20],
      "ct": [0.6, 0.65, 0.7]
    }
  },
  "filters": {"min_rrr": 0.3, "min_trades": 50}
}
```

### Swing Trading (lange Perioden, Trend-Fokus)

```json
{
  "name": "Swing Trading",
  "pipeline": {
    "indicators": [
      {"name": "trend", "params": {"adx_periods": [14, 21, 30], "ema_periods": [21, 50, 100, 200]}},
      {"name": "regime", "params": {"hurst_windows": [100, 200, 500]}},
      {"name": "multi_timeframe", "params": {"h4_bars": 4, "d1_bars": 24, "ema_periods": [20, 50, 100]}},
      {"name": "structure", "params": {}},
      {"name": "risk", "params": {"dd_windows": [100, 200, 500]}},
      {"name": "market_regime", "params": {"window": 50}}
    ],
    "feature_selection": [
      {"name": "boruta", "params": {"max_features": 30}}
    ],
    "data_loading": [
      {"name": "macro_data", "source": "forexsb"},
      {"name": "cot_positioning", "source": "forexsb"}
    ]
  },
  "exit_strategy": "fixed",
  "grids": {
    "FOREX": {
      "tp": [20, 30, 50, 80],
      "sl": [20, 30, 40, 60],
      "ct": [0.5, 0.55, 0.6]
    }
  },
  "filters": {"min_rrr": 0.5, "min_trades": 30}
}
```

### ATR-Based Exit Strategie

```json
{
  "name": "ATR Exploration",
  "pipeline": {
    "indicators": [
      {"name": "trend", "params": {}},
      {"name": "momentum", "params": {}},
      {"name": "volatility", "params": {}},
      {"name": "market_regime", "params": {"window": 50}}
    ],
    "data_loading": [
      {"name": "macro_data", "source": "forexsb"},
      {"name": "cot_positioning", "source": "forexsb"}
    ]
  },
  "exit_strategy": "atr_based",
  "exit_params": {"atr_period": 14, "min_tp_pips": 10, "min_sl_pips": 15},
  "grids": {
    "FOREX": {
      "tp": [0.5, 1.0, 1.5, 2.0, 3.0],
      "sl": [0.5, 1.0, 1.5, 2.0],
      "ct": [0.50, 0.55, 0.60],
      "timeout_bars": [null, 24, 48, 96]
    }
  }
}
```
