# Support & Resistance Indicator Plugin Design

**Datum**: 2026-02-17
**Plugin-Name**: `support_resistance`
**Package**: `fwbg-premium`
**Inspiration**: Rayner Teo — "Price Action Trading Secrets" (MAEE-Framework)

## Konzept

Trend-Continuation + Range-Trading Strategie basierend auf S/R-Zonen:

- **Uptrend** → Long an Support (Pullback-Entry)
- **Downtrend** → Short an Resistance (Rally-Entry)
- **Sideways** → Long an Support UND Short an Resistance (Range-Trading)

Multi-Timeframe: S/R-Zonen werden auf H1 UND D1 berechnet. Daily-Zonen sind auf dem Stundenchart sichtbar, obwohl das H1-Lookback nur ~8 Tage abdeckt.

Das Plugin berechnet Features — der XGBoost lernt die Kombinationen selbst.

## Architektur

Indikator-Plugin (`BaseIndicator`), registriert als `support_resistance`.

```
packages/fwbg-premium/src/fwbg_premium/plugins/fwbg-premium/indicators/support_resistance/
├── __init__.py
└── manifest.json
```

## Multi-Timeframe-Ansatz

S/R-Zonen auf zwei Ebenen, berechnet über Rolling-Window-Aggregation (wie multi_timeframe-Plugin):

| Timeframe | Swing-Scaling | Lookback | Abdeckung |
|-----------|-------------|----------|-----------|
| H1 | `swing_period * 1` | 200 Bars | ~8 Tage (Intraday-Zonen) |
| D1 | `swing_period * 24` | 200 * 24 = 4800 Bars | ~200 Tage (Daily-Zonen) |

H1-Features erhalten `sr_`-Prefix, D1-Features `sr_d1_`-Prefix.

D1-Aggregation der OHLC-Daten:
```python
d1_high = df["H"].rolling(24).max()
d1_low = df["L"].rolling(24).min()
```

## Feature-Design

### Tier 1 — S/R-Zonen-Identifikation

**Swing Detection**: Swing High = Bar dessen High höher ist als die N Bars links UND rechts. Bestätigung erst `period` Bars später → kein Lookahead.

**Clustering**: Swing-Levels innerhalb `cluster_threshold * ATR` → eine Zone.

Zone-Eigenschaften:
- `center`: Mittelpunkt der geclusterten Levels
- `touches`: Anzahl Swing-Punkte in der Zone
- `type`: support / resistance / both (Flip-Zone)

**H1 Features (7):**

| Feature | Typ | Beschreibung |
|---------|-----|-------------|
| `sr_dist_nearest_support` | float | Distanz zur nächsten H1-Support-Zone, ATR-normalisiert |
| `sr_dist_nearest_resistance` | float | Distanz zur nächsten H1-Resistance-Zone, ATR-normalisiert |
| `sr_support_strength` | int | Touches der nächsten H1-Support-Zone |
| `sr_resistance_strength` | int | Touches der nächsten H1-Resistance-Zone |
| `sr_in_support_zone` | binary | Preis innerhalb einer H1-Support-Zone |
| `sr_in_resistance_zone` | binary | Preis innerhalb einer H1-Resistance-Zone |
| `sr_nearest_is_flip_zone` | binary | Nächste H1-Zone ist Flip-Zone |

**D1 Features (7):**

| Feature | Typ | Beschreibung |
|---------|-----|-------------|
| `sr_d1_dist_nearest_support` | float | Distanz zur nächsten D1-Support-Zone |
| `sr_d1_dist_nearest_resistance` | float | Distanz zur nächsten D1-Resistance-Zone |
| `sr_d1_support_strength` | int | Touches der nächsten D1-Support-Zone |
| `sr_d1_resistance_strength` | int | Touches der nächsten D1-Resistance-Zone |
| `sr_d1_in_support_zone` | binary | Preis in D1-Support-Zone |
| `sr_d1_in_resistance_zone` | binary | Preis in D1-Resistance-Zone |
| `sr_d1_nearest_is_flip_zone` | binary | Nächste D1-Zone ist Flip-Zone |

### Tier 2 — Trend-Kontext nach Rayner (6 Features)

Trend-Klassifizierung basierend auf Pullback-Tiefe und MA-Alignment:

| Trend-Typ | Preis-Verhalten | MA-Alignment | Trading-Logik |
|-----------|----------------|-------------|---------------|
| Starker Uptrend (+3) | Pullbacks halten über 20 MA | 20 > 50 > 200 | Long an Support |
| Gesunder Uptrend (+2) | Pullbacks zum 50 MA | 50 > 200 | Long an Support |
| Schwacher Uptrend (+1) | Pullbacks zum 200 MA | Preis nahe 200 | Long an Support |
| **Sideways (0)** | **Preis pendelt in Range** | **MAs flach/verschränkt** | **Long an S + Short an R** |
| Schwacher Downtrend (-1) | Rallies zum 200 MA | Preis nahe 200 | Short an Resistance |
| Gesunder Downtrend (-2) | Rallies zum 50 MA | 50 < 200 | Short an Resistance |
| Starker Downtrend (-3) | Rallies halten unter 20 MA | 20 < 50 < 200 | Short an Resistance |

| Feature | Typ | Beschreibung |
|---------|-----|-------------|
| `sr_trend_class` | int (-3..+3) | Trend-Kategorie nach Rayner |
| `sr_pullback_depth` | float | Distanz vom letzten Swing-Extreme, ATR-normalisiert |
| `sr_ma_alignment` | float (-1..+1) | Grad der MA-Ausrichtung (20/50/200) |
| `sr_price_vs_ma20` | float | (Close - MA20) / ATR |
| `sr_price_vs_ma50` | float | (Close - MA50) / ATR |
| `sr_price_vs_ma200` | float | (Close - MA200) / ATR |

### Tier 3 — Interaktion S/R + Trend (8 Features)

| Feature | Typ | Beschreibung |
|---------|-----|-------------|
| `sr_at_support_in_uptrend` | binary | Preis nahe Support UND Trend > 0 |
| `sr_at_resistance_in_downtrend` | binary | Preis nahe Resistance UND Trend < 0 |
| `sr_at_support_in_range` | binary | Preis nahe Support UND Trend == 0 |
| `sr_at_resistance_in_range` | binary | Preis nahe Resistance UND Trend == 0 |
| `sr_range_width` | float | Breite der Range (obere R minus untere S), ATR-normalisiert |
| `sr_range_position` | float (0..1) | Position in der Range (0=Support, 1=Resistance) |
| `sr_breakout_up` | binary | Close durchbricht Resistance-Zone |
| `sr_breakout_down` | binary | Close durchbricht Support-Zone |

**Gesamt: ~28 Features** (7 H1 + 7 D1 + 6 Trend + 8 Interaktion).

## Parameter

```json
{
  "name": "support_resistance",
  "params": {
    "swing_periods": [5, 10, 20],
    "lookback": 200,
    "cluster_threshold": 1.5,
    "atr_period": 14,
    "ma_periods": [20, 50, 200],
    "zone_proximity_atr_mult": 0.5,
    "d1_bars": 24
  }
}
```

| Parameter | Default | Beschreibung |
|-----------|---------|-------------|
| `swing_periods` | [5, 10, 20] | Perioden für Swing-Detection (mehrere → robustere Zonen) |
| `lookback` | 200 | Rolling Window für Zone-Gültigkeit (H1-Bars) |
| `cluster_threshold` | 1.5 | ATR-Multiplikator für Zone-Clustering |
| `atr_period` | 14 | ATR-Periode für Normalisierung |
| `ma_periods` | [20, 50, 200] | MAs für Trend-Klassifizierung |
| `zone_proximity_atr_mult` | 0.5 | Wann Preis als "in Zone" gilt |
| `d1_bars` | 24 | Bars pro D1-Kerze (24h für Stundenchart) |

## Algorithmen

### Swing Detection (Lookahead-safe)

```python
def _detect_swings(highs, lows, period):
    """Swing-Erkennung. Bestätigung erst period Bars nach dem Extremum."""
    n = len(highs)
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)
    for i in range(period * 2, n):
        window_h = highs[i - 2*period : i + 1]
        window_l = lows[i - 2*period : i + 1]
        if window_h[period] == np.max(window_h):
            swing_highs[i] = highs[i - period]
        if window_l[period] == np.min(window_l):
            swing_lows[i] = lows[i - period]
    return swing_highs, swing_lows
```

### Zone Clustering

```python
def _cluster_levels(levels, atr_value, threshold=1.5):
    """Gruppiert nahe Levels zu Zonen."""
    sorted_levels = np.sort(levels[~np.isnan(levels)])
    zones = []
    current = [sorted_levels[0]]
    for level in sorted_levels[1:]:
        if level - np.mean(current) < threshold * atr_value:
            current.append(level)
        else:
            zones.append({"center": np.mean(current), "touches": len(current)})
            current = [level]
    zones.append({"center": np.mean(current), "touches": len(current)})
    return zones
```

### Trend-Klassifizierung

```python
def _classify_trend(close, ma20, ma50, ma200):
    """Rayner-Trend-Klassifizierung (-3..+3)."""
    if ma20 > ma50 > ma200:
        if close > ma20: return 3
        if close > ma50: return 2
        return 1
    if ma20 < ma50 < ma200:
        if close < ma20: return -3
        if close < ma50: return -2
        return -1
    return 0
```

## Abhängigkeiten

- OHLC-Daten (O, H, L, C)
- ATR (berechnet selbst oder nutzt `vol_atr`)
- Keine externen Libraries (numpy, pandas)

## Test-Strategie

1. **Swing Detection**: Bekannte Muster → korrekte Erkennung
2. **Lookahead-Bias**: Features nur aus vergangenen Daten
3. **Zone Clustering**: Nahe Levels korrekt gruppiert
4. **D1-Zonen**: Auf Stundenchart sichtbar, plausible Distanzen
5. **Trend-Klassifizierung**: MA-Konfigurationen → korrekte Kategorie
6. **Range-Features**: `sr_range_position` im Bereich [0, 1]
7. **Integration**: Plugin in Strategy-JSON, compute_indicator_pool
8. **Smoke**: Keine NaN/0-Features bei realen Daten
