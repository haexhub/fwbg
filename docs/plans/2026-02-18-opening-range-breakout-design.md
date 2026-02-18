# Opening Range Breakout (ORB) Indicator Plugin

## Ziel

ML-Feature-Generator basierend auf dem Opening Range Breakout Konzept.
Berechnet Features für jede Stundenwechsel-Grenze und konfigurierbare Sessions,
damit das ML-Modell selbst lernt, welche Stunden einen Edge liefern.

## Feature-Gruppen

### 1. Rolling ORB (`orb_*`)
Relativ zur jeweils letzten vollen Stunde. Kompakt, immer aktuell.

| Feature | Typ | Beschreibung |
|---------|-----|-------------|
| `orb_range` | float | Range-Größe (H-L der ersten N Bars), normalisiert durch Close |
| `orb_position` | float | Preis-Position in der Range (0=Low, 1=High, >1 oben raus) |
| `orb_breakout_up` | signal | Close > Opening Range High |
| `orb_breakout_down` | signal | Close < Opening Range Low |
| `orb_range_vs_atr` | float | Opening Range / ATR — schmale Range = Breakout-Potenzial |

### 2. Session ORB (`orb_s{HH}_*`)
Pro konfigurierter Stunde dieselben Features, aber fix auf diese Stunde bezogen.
Bleibt aktiv bis nächster Session-Start.

Default-Sessions: `[0, 8, 13, 14]` (Asia, London, NY Pre-Market, NY Open)

Features pro Session: `orb_s{HH}_range`, `orb_s{HH}_position`, `orb_s{HH}_breakout_up`,
`orb_s{HH}_breakout_down`, `orb_s{HH}_range_vs_atr`

### 3. Statistik-Features (`orb_stat_*`)

| Feature | Typ | Beschreibung |
|---------|-----|-------------|
| `orb_stat_avg_range` | float | Durchschnittliche ORB-Range der letzten N Stunden |
| `orb_stat_breakout_rate` | float | Anteil der Stunden mit Breakout (rolling) |
| `orb_stat_continuation_rate` | float | Wie oft läuft Breakout-Richtung weiter? |

## Parameter

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|-------------|
| `range_bars` | int | 1 | Bars die die Opening Range definieren |
| `atr_period` | int | 14 | ATR-Periode für Normalisierung |
| `sessions` | list[int] | [0,8,13,14] | UTC-Stunden für Session-Features |
| `stat_window` | int | 20 | Rolling Window für Statistik-Features |
| `enable_rolling` | bool | True | Rolling ORB Features an/aus |
| `enable_session` | bool | True | Session ORB Features an/aus |
| `enable_stats` | bool | True | Statistik-Features an/aus |

## Plugin-Metadaten

- **Name:** `opening_range`
- **Package:** `fwbg-core`
- **Prefix:** `orb_`
- **benefits_from_stationary:** `False` (absolute Preis-Ranges, keine Stationarität nötig)
- **Timeframe-Kompatibilität:** Intraday (M1-H4). Auf DAY → nur NaN.

## Algorithmus

### Rolling ORB (vektorisiert)
```python
hour_group = df.index.floor('h')
bar_in_hour = df.groupby(hour_group).cumcount()
or_mask = bar_in_hour < range_bars
or_high = df['H'].where(or_mask).groupby(hour_group).transform('max')
or_low = df['L'].where(or_mask).groupby(hour_group).transform('min')
valid = bar_in_hour >= range_bars
```

### Session ORB
```python
# Für jede session_hour: Finde Bars wo index.hour == session_hour
# Berechne Range der ersten range_bars Bars
# Forward-fill bis nächste Session
```
