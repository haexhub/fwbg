# Feature & Indikator Dokumentation

Dieses Dokument beschreibt alle Features und Indikatoren, die dem ML-Modell zur Verfügung stehen.

**Gesamt: ~220-300+ Features** (abhängig von verfügbaren Makro-Daten)

---

## Inhaltsverzeichnis

1. [Trend Indikatoren](#1-trend-indikatoren-trend_-ichi_)
2. [Momentum Indikatoren](#2-momentum-indikatoren-mom_)
3. [Volatilität Indikatoren](#3-volatilität-indikatoren-vol_)
4. [Distribution Features](#4-distribution-features-dist_)
5. [FFT Features](#5-fft-features-fft_)
6. [Price Action Features](#6-price-action-features-pa_)
7. [Zeit Features](#7-zeit-features-time_)
8. [Saisonalität Features](#8-saisonalität-features-season_)
9. [Dynamik Features](#9-dynamik-features-dyn_)
10. [Lag Features](#10-lag-features-lag_)
11. [Beschleunigung Features](#11-beschleunigung-features-accel_)
12. [Cross-Indikator Features](#12-cross-indikator-features-cross_)
13. [Multi-Timeframe Features](#13-multi-timeframe-features-mtf_)
14. [Regime Features](#14-regime-features-regime_)
15. [Event Features](#15-event-features-event_)
16. [Struktur Features](#16-struktur-features-path_-fractal_-convex_-structure_)
17. [Korrelations Features](#17-korrelations-features-corr_-lead_lag_)
18. [Risk/Tail-Risk Features](#18-risktail-risk-features-risk_)
19. [Microstructure Features](#19-microstructure-features-micro_)
20. [Macro Surprise Features](#20-macro-surprise-features-macsurp_)
21. [Makro Features](#21-makro-features-macro_)
22. [Fair Value Gap Features](#22-fair-value-gap-features-fvg_)
23. [Support/Resistance Features](#23-supportresistance-features-sr_)
24. [Feature-Gruppen / Indicator Plugins](#feature-gruppen--indicator-plugins)
25. [Early Termination & Grid-Optimierung](#early-termination--grid-optimierung)

---

## 1. Trend Indikatoren (`trend_`, `ichi_`)

### ADX (Average Directional Index)
Misst die Stärke eines Trends, unabhängig von der Richtung.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `trend_adx_7` | ADX mit 7-Perioden | Kurzfristiger Trend |
| `trend_adx_14` | ADX mit 14-Perioden | Standard Trend-Stärke |
| `trend_adx_21` | ADX mit 21-Perioden | Längerfristiger Trend |

**Werte:** 0-100. >25 = Trend vorhanden, >50 = starker Trend, <20 = kein Trend

### EMA Distance (Exponential Moving Average)
Relativer Abstand des Preises zum EMA.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `trend_ema_dist_8` | Abstand zum EMA-8 (%) | Sehr kurzfristig |
| `trend_ema_dist_21` | Abstand zum EMA-21 (%) | Kurzfristig |
| `trend_ema_dist_50` | Abstand zum EMA-50 (%) | Mittelfristig |
| `trend_ema_dist_100` | Abstand zum EMA-100 (%) | Längerfristig |
| `trend_ema_dist_200` | Abstand zum EMA-200 (%) | Langfristig |

**Werte:** Positiv = Preis über EMA (bullish), Negativ = Preis unter EMA (bearish)

### SMA Distance (Simple Moving Average)
Relativer Abstand des Preises zum SMA.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `trend_sma_dist_20` | Abstand zum SMA-20 (%) | Kurzfristig |
| `trend_sma_dist_50` | Abstand zum SMA-50 (%) | Mittelfristig |
| `trend_sma_dist_200` | Abstand zum SMA-200 (%) | Langfristig |

### MACD (Moving Average Convergence Divergence)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `trend_macd` | MACD Differenz (normalisiert) | Momentum des Trends |
| `trend_macd_signal` | MACD Signal Line (normalisiert) | Geglättetes Signal |

### CCI (Commodity Channel Index)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `trend_cci_14` | CCI mit 14-Perioden | Standard |
| `trend_cci_20` | CCI mit 20-Perioden | Geglättet |

**Werte:** >100 = überkauft, <-100 = überverkauft

### Aroon Indikator

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `trend_aroon_up` | Aroon Up | Zeit seit letztem High |
| `trend_aroon_down` | Aroon Down | Zeit seit letztem Low |

**Werte:** 0-100. Aroon Up > Aroon Down = Aufwärtstrend

### Kaufman's Efficiency Ratio
Misst die Effizienz einer Preisbewegung (Signal vs. Noise).

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `trend_er_10` | ER über 10 Perioden | Kurzfristige Effizienz |
| `trend_er_20` | ER über 20 Perioden | Mittelfristige Effizienz |
| `trend_er_50` | ER über 50 Perioden | Längerfristige Effizienz |
| `trend_er_10_chg` | ER-10 Änderung | Momentum der Effizienz |
| `trend_er_20_chg` | ER-20 Änderung | Momentum der Effizienz |

**Werte:** 0-1. Nahe 1 = starker, effizienter Trend. Nahe 0 = Seitwärtsbewegung/Noise

### Ichimoku Cloud

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `ichi_tenkan` | Conversion Line (9-Perioden) | Schnelle Linie |
| `ichi_kijun` | Base Line (26-Perioden) | Langsame Linie |
| `ichi_senkou_a` | Leading Span A | Cloud-Grenze |
| `ichi_senkou_b` | Leading Span B | Cloud-Grenze |
| `ichi_cloud_pos` | Position relativ zur Cloud | 0-1, über/unter Cloud |
| `ichi_cloud_thick` | Cloud-Dicke (normalisiert) | Stärke Support/Resistance |
| `ichi_tk_cross` | Tenkan-Kijun Differenz | Cross-Signal |
| `ichi_price_kijun` | Preis-Kijun Abstand | Trend-Stärke |

---

## 2. Momentum Indikatoren (`mom_`)

### RSI (Relative Strength Index)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `mom_rsi_7` | RSI mit 7-Perioden | Sehr sensitiv |
| `mom_rsi_14` | RSI mit 14-Perioden | Standard |
| `mom_rsi_21` | RSI mit 21-Perioden | Geglättet |

**Werte:** 0-100. >70 = überkauft, <30 = überverkauft

### Stochastic Oscillator

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `mom_stoch_k_14` | %K mit 14-Perioden | Schnelle Linie |
| `mom_stoch_d_14` | %D mit 14-Perioden | Signal-Linie |
| `mom_stoch_k_21` | %K mit 21-Perioden | Geglättet |
| `mom_stoch_d_21` | %D mit 21-Perioden | Geglättete Signal-Linie |

**Werte:** 0-100. >80 = überkauft, <20 = überverkauft

### Williams %R

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `mom_williams_14` | Williams %R, 14-Perioden | Standard |
| `mom_williams_21` | Williams %R, 21-Perioden | Geglättet |

**Werte:** -100 bis 0. >-20 = überkauft, <-80 = überverkauft

### Ultimate Oscillator

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `mom_uo` | Ultimate Oscillator | Multi-Timeframe Momentum |

**Werte:** 0-100. Kombiniert 7, 14, 28 Perioden

### Rate of Change (ROC)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `mom_roc_5` | ROC über 5 Perioden | Kurzfristig |
| `mom_roc_10` | ROC über 10 Perioden | Mittelfristig |
| `mom_roc_20` | ROC über 20 Perioden | Längerfristig |

**Werte:** Prozentuale Preisänderung

---

## 3. Volatilität Indikatoren (`vol_`)

### Bollinger Bands

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `vol_bb_pband_20` | Position in den Bands | 0 = unteres Band, 1 = oberes Band |
| `vol_bb_wband_20` | Bandbreite (normalisiert) | Volatilität |

### Keltner Channel

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `vol_kc_pband` | Position im Channel | 0-1 |
| `vol_kc_wband` | Channel-Breite | Volatilität |

### Donchian Channel

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `vol_dc_pband` | Position im Channel | 0-1 |
| `vol_dc_wband` | Channel-Breite | Volatilität |

### ATR (Average True Range)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `vol_atr_pct_7` | ATR/Preis (7-Perioden) | Kurzfristige Volatilität |
| `vol_atr_pct_14` | ATR/Preis (14-Perioden) | Standard Volatilität |
| `vol_atr_pct_21` | ATR/Preis (21-Perioden) | Geglättete Volatilität |
| `vol_atr` | Absoluter ATR | Für interne Berechnungen |

### Volume-basiert (falls verfügbar)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `vol_obv_change` | OBV Änderung (5 Perioden) | Volume-Momentum |
| `vol_mfi` | Money Flow Index | Volume-gewichteter RSI |

---

## 4. Distribution Features (`dist_`)

Analysieren die statistische Verteilung der Returns.

### Skewness (Schiefe)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `dist_skew_20` | Rolling Skewness, 20 Perioden | Kurzfristig |
| `dist_skew_50` | Rolling Skewness, 50 Perioden | Mittelfristig |
| `dist_skew_100` | Rolling Skewness, 100 Perioden | Längerfristig |
| `dist_skew_20_z` | Z-Score der Skewness | Relativ zur Historie |
| `dist_skew_50_z` | Z-Score der Skewness | Relativ zur Historie |

**Werte:** Positiv = mehr positive Ausreißer, Negativ = mehr negative Ausreißer

### Kurtosis

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `dist_kurt_20` | Rolling Kurtosis, 20 Perioden | Kurzfristig |
| `dist_kurt_50` | Rolling Kurtosis, 50 Perioden | Mittelfristig |
| `dist_kurt_100` | Rolling Kurtosis, 100 Perioden | Längerfristig |
| `dist_kurt_20_z` | Z-Score der Kurtosis | Relativ zur Historie |
| `dist_kurt_50_z` | Z-Score der Kurtosis | Relativ zur Historie |

**Werte:** Hoch = Fat Tails (mehr Extremereignisse), Niedrig = dünne Tails

---

## 5. FFT Features (`fft_`)

Fourier-Transformation zur Zykluserkennung.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `fft_dom_freq_64` | Dominante Frequenz (64 Bars) | Hauptzyklus kurzfristig |
| `fft_dom_freq_128` | Dominante Frequenz (128 Bars) | Hauptzyklus mittelfristig |
| `fft_dom_freq_256` | Dominante Frequenz (256 Bars) | Hauptzyklus langfristig |
| `fft_dom_power_*` | Power der dominanten Frequenz | Stärke des Zyklus |
| `fft_energy_*` | Spektrale Energie | Gesamtstärke aller Zyklen |
| `fft_entropy_*` | Spektrale Entropie | Verteilung der Energie |
| `fft_lowfreq_*` | Low-Frequency Ratio | Anteil langfristiger Trends |

**Interpretation:**
- Hohe Entropie = Rauschen (gleichmäßig verteilte Energie)
- Niedrige Entropie = Klare Zyklen (konzentrierte Energie)
- Hohe Low-Freq Ratio = Starke langfristige Trends

---

## 6. Price Action Features (`pa_`)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `pa_range_pos` | Position in High-Low Range | 0 = Low, 1 = High |
| `pa_hh` | Higher Highs (letzte 5 Bars) | Aufwärtsdruck |
| `pa_ll` | Lower Lows (letzte 5 Bars) | Abwärtsdruck |
| `pa_body_ratio` | Body/Range Ratio | Kerzen-Stärke |
| `pa_gap` | Gap vom Vortag (%) | Overnight-Bewegung |

---

## 7. Zeit Features (`time_`)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `time_hour` | Stunde (0-23) | Session-Timing |
| `time_day` | Wochentag (0=Mo, 6=So) | Wochentags-Effekte |
| `time_hour_sin` | Zyklische Stunde (sin) | Für ML-Modelle |
| `time_hour_cos` | Zyklische Stunde (cos) | Für ML-Modelle |

---

## 8. Saisonalität Features (`season_`)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `season_month` | Monat (1-12) | Monats-Effekte |
| `season_month_sin/cos` | Zyklischer Monat | Für ML-Modelle |
| `season_quarter` | Quartal (1-4) | Quartals-Effekte |
| `season_quarter_sin/cos` | Zyklisches Quartal | Für ML-Modelle |
| `season_week` | Kalenderwoche (1-52) | Wochen-Effekte |
| `season_week_sin/cos` | Zyklische Woche | Für ML-Modelle |
| `season_dayofmonth` | Tag im Monat (1-31) | Monatsanfang/-ende |
| `season_dayofmonth_sin/cos` | Zyklischer Tag | Für ML-Modelle |

---

## 9. Dynamik Features (`dyn_`)

Messen Änderungen von Indikatoren über Zeit.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `dyn_rsi14_chg_4h/8h/24h` | RSI Änderung | Momentum-Shift |
| `dyn_rsi14_pct_4h/8h/24h` | RSI % Änderung | Relatives Momentum |
| `dyn_atr_chg_4h/8h/24h` | ATR Änderung | Volatilitäts-Shift |
| `dyn_bbwidth_chg_4h/8h/24h` | BB-Breite Änderung | Squeeze/Expansion |
| `dyn_adx_chg_4h/8h/24h` | ADX Änderung | Trend-Shift |
| `dyn_macd_chg_4h/8h` | MACD Änderung | Signal-Shift |
| `dyn_stoch_chg_4h/8h` | Stochastic Änderung | Momentum-Shift |

---

## 10. Lag Features (`lag_`)

Verzögerte Indikator-Werte.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `lag_rsi14_4h/8h/24h` | RSI vor X Stunden | Historischer Zustand |
| `lag_atr_4h/8h/24h` | ATR vor X Stunden | Historische Volatilität |
| `lag_adx_4h/8h` | ADX vor X Stunden | Historischer Trend |
| `lag_price_chg_4h/8h/24h/48h` | Preisänderung seit X Stunden | Performance |

---

## 11. Beschleunigung Features (`accel_`)

Zweite Ableitung - Änderung der Änderung.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `accel_rsi` | RSI Beschleunigung | Momentum des Momentums |
| `accel_atr` | ATR Beschleunigung | Volatilität des Volatilitäts-Shifts |

---

## 12. Cross-Indikator Features (`cross_`)

Kombinierte Signale aus mehreren Indikatoren.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `cross_rsi_high_rising` | RSI>70 UND steigend | Überkauft + Momentum |
| `cross_rsi_low_falling` | RSI<30 UND fallend | Überverkauft + Momentum |
| `cross_vol_trend` | ATR-Änderung × ADX | Volatilität im Trend |

---

## 13. Multi-Timeframe Features (`mtf_`)

Aggregierte Features über höhere Zeitrahmen.

### H4 (4-Stunden)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `mtf_h4_trend` | H4 Trend-Richtung | Übergeordneter Trend |
| `mtf_h4_range_pos` | Position in H4 Range | 0-1 |
| `mtf_h4_ema20_dist` | Abstand zum H4 EMA-20 | Trend-Stärke |
| `mtf_h4_ema50_dist` | Abstand zum H4 EMA-50 | Längerfristiger Trend |
| `mtf_h4_adx` | H4 ADX | Trend-Stärke |
| `mtf_h4_rsi` | H4 RSI | Momentum |
| `mtf_h4_atr_pct` | H4 Volatilität | Höherer TF Volatilität |
| `mtf_vol_ratio` | H1/H4 Volatilitäts-Ratio | Relative Volatilität |
| `mtf_trend_alignment` | H1/H4 Trend-Übereinstimmung | Confluence |

### D1 (Daily)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `mtf_d1_range_pos` | Position in D1 Range | 0-1 |
| `mtf_d1_ema20_dist` | Abstand zum D1 EMA-20 | Daily Trend |
| `mtf_consensus` | H1/H4/D1 Trend-Konsens | Multi-TF Confluence |

---

## 14. Regime Features (`regime_`)

Markt-Charakter Indikatoren basierend auf dem Hurst-Exponenten.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `regime_hurst_100` | Hurst, 100 Bars | Kurzfristiger Charakter |
| `regime_hurst_200` | Hurst, 200 Bars | Mittelfristiger Charakter |
| `regime_hurst_500` | Hurst, 500 Bars | Langfristiger Charakter |
| `regime_hurst_100_chg` | Hurst Änderung (24h) | Regime-Shift |
| `regime_hurst_200_chg` | Hurst Änderung (48h) | Regime-Shift |
| `regime_hurst_divergence` | Hurst Short vs Long | Multi-Scale Divergenz |

**Hurst-Exponent Interpretation:**
- H > 0.5: Trending/Persistent (Trend-Following funktioniert)
- H = 0.5: Random Walk (schwierig zu traden)
- H < 0.5: Mean-Reverting (Mean-Reversion funktioniert)

---

## 15. Event Features (`event_`)

Time-Since-Event Features messen, wie viele Bars seit wichtigen Ereignissen vergangen sind.

**Kernidee:** Ein Ausbruch nach 100 Bars Konsolidierung ist oft stärker als einer nach 5 Bars.

### Bars seit High/Low

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `event_bars_since_high_20` | Bars seit 20-Perioden High | Konsolidierungsdauer |
| `event_bars_since_high_50` | Bars seit 50-Perioden High | Längere Konsolidierung |
| `event_bars_since_low_20` | Bars seit 20-Perioden Low | Konsolidierungsdauer |
| `event_bars_since_low_50` | Bars seit 50-Perioden Low | Längere Konsolidierung |
| `event_bars_since_*_log` | Log-transformiert | Bessere Skalierung |

### Bars seit Signalen

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `event_bars_since_ema_cross` | Bars seit EMA8/EMA21 Cross | Trend-Alter |
| `event_bars_since_rsi_extreme` | Bars seit RSI >70 oder <30 | Zeit seit Überkauft/Überverkauft |
| `event_bars_since_vol_spike` | Bars seit ATR > 2x Durchschnitt | Zeit seit Volatilitäts-Event |

**Anwendung:** Hohe Werte = Lange Konsolidierung = Potenziell starker Ausbruch

---

## 16. Struktur Features (`path_`, `fractal_`, `convex_`, `structure_`)

Analysieren die mathematische Struktur der Preisbewegung.

### Path Efficiency (Fractal Dimension Proxy)

Misst wie "gerade" eine Preisbewegung ist.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `path_efficiency_10` | PE über 10 Bars | Kurzfristige Effizienz |
| `path_efficiency_20` | PE über 20 Bars | Mittelfristige Effizienz |
| `path_efficiency_50` | PE über 50 Bars | Längerfristige Effizienz |
| `path_efficiency_100` | PE über 100 Bars | Langfristige Effizienz |
| `path_efficiency_*_chg` | PE Änderung | Regime-Shift Detektion |

**Formel:** PE = |Netto-Bewegung| / Summe(|Einzelbewegungen|)

**Werte:**
- PE ≈ 1.0: Perfekt gerade Linie (starker Trend)
- PE ≈ 0.0: Viel Hin und Her (Range/Noise)

### Fractal Dimension

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `fractal_dim_10/20/50/100` | Fraktale Dimension | Komplexität der Bewegung |

**Formel:** D = 1 + (1 - PE)

**Werte:**
- D ≈ 1.0: Trending (einfache Struktur)
- D ≈ 2.0: Range/Random (komplexe Struktur)

### Convexity (2. Ableitung des EMA)

Frühwarnsystem für parabolische Tops/Bottoms.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `convex_ema_21` | Convexity des EMA-21 | Kurzfristige Beschleunigung |
| `convex_ema_50` | Convexity des EMA-50 | Längerfristige Beschleunigung |
| `convex_ema_*_smooth` | Geglättete Version | Weniger Noise |
| `convex_divergence` | EMA21 vs EMA50 Convexity | Multi-Scale Divergenz |
| `convex_zscore` | Z-Score der Convexity | Extremwerte-Detektion |

**Interpretation:**
- Positiv (konvex): Trend beschleunigt sich (parabolisch) → Vorsicht bei Tops
- Negativ (konkav): Trend verlangsamt sich → Mögliche Trendwende
- Nahe 0: Linearer, stabiler Trend

### VWAP Features

VWAP-ähnliche Referenzpunkte (ohne echtes Volume, Typical Price als Proxy).

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `structure_vwap_dist_20` | Abstand zum VWAP-Proxy (20 Bars) | Kurzfristige Position |
| `structure_vwap_dist_50` | Abstand zum VWAP-Proxy (50 Bars) | Mittelfristige Position |
| `structure_vwap_dist_100` | Abstand zum VWAP-Proxy (100 Bars) | Langfristige Position |
| `structure_vwap_time_above` | Zeit über VWAP (Rolling Ratio) | Acceptance vs Rejection |
| `structure_bars_since_vwap_cross` | Bars seit VWAP-Cross | Cross-Alter |

**Formel:** VWAP-Proxy = Rolling Mean des Typical Price (H+L+C)/3

**Anwendung:** Institutionelle Trader nutzen VWAP als Referenzpunkt. Preis über VWAP = bullish Bias.

---

## 17. Korrelations Features (`corr_`, `lead_lag_`)

Analysieren Beziehungen zu Benchmark-Assets (SPX, VIX).

### Correlation Stability

Decoupling (plötzlicher Korrelationsbruch) ist oft Vorbote für massive Volatilität.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `corr_spx_20` | Rolling Korrelation zu SPX (20 Bars) | Kurzfristige Korrelation |
| `corr_spx_50` | Rolling Korrelation zu SPX (50 Bars) | Mittelfristige Korrelation |
| `corr_spx_100` | Rolling Korrelation zu SPX (100 Bars) | Langfristige Korrelation |
| `corr_spx_stability` | Änderung der Korrelation | Stabilität |
| `corr_spx_decoupling` | Absolute Korrelationsänderung | Decoupling-Stärke |

**Anwendung:** Hoher `corr_spx_decoupling` = Fundamentale Änderung = Erhöhte Volatilität erwartet

### VIX Korrelation

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `corr_vix_20` | Rolling Korrelation zu VIX (20 Bars) | Kurzfristig |
| `corr_vix_50` | Rolling Korrelation zu VIX (50 Bars) | Mittelfristig |

### Lead-Lag Momentum

VIX und Benchmarks führen oft vor dem Asset.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `lead_lag_spx` | Asset-Momentum minus SPX-Momentum | Relative Stärke |
| `lead_lag_vix` | VIX-Änderung plus Asset-Änderung | Divergenz-Indikator |
| `vix_lead_signal` | VIX-Änderung vor 5 Bars | Führungsindikator |

**Anwendung:**
- `lead_lag_spx` > 0: Asset outperformt SPX
- `vix_lead_signal` stark negativ: VIX ist vor 5 Bars gefallen → bullishes Signal

---

## 18. Risk/Tail-Risk Features (`risk_`)

Features zur Messung von Tail-Risk und Crash-Wahrscheinlichkeit.

### Drawdown State Features

Messen den aktuellen Drawdown-Zustand.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `risk_dd_pct_50` | Aktueller Drawdown vom 50-Bar Peak | % unter dem Peak |
| `risk_dd_pct_100` | Aktueller Drawdown vom 100-Bar Peak | % unter dem Peak |
| `risk_dd_pct_200` | Aktueller Drawdown vom 200-Bar Peak | % unter dem Peak |
| `risk_dd_ratio_50/100/200` | DD relativ zum Max-DD des Fensters | 0=kein DD, 1=Max DD |
| `risk_bars_since_peak` | Bars seit dem letzten 200-Bar High | Drawdown-Dauer |
| `risk_bars_since_peak_log` | Log-transformiert | Bessere Skalierung |
| `risk_recovery_ratio` | Erholung vom letzten Tief | 0-1, wie viel erholt |

**Anwendung:** Hoher Drawdown + lange Dauer = erhöhtes Risiko für weitere Verluste

### CVaR (Conditional Value at Risk)

Expected Shortfall - durchschnittlicher Verlust im Tail.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `risk_var_5_50` | 95% VaR, 50-Bar Window | 5% schlimmste Returns |
| `risk_var_5_100` | 95% VaR, 100-Bar Window | Längerfristiger VaR |
| `risk_var_1_50` | 99% VaR, 50-Bar Window | Extremere Events |
| `risk_var_1_100` | 99% VaR, 100-Bar Window | Längerfristiger extremer VaR |
| `risk_cvar_5_50/100` | 95% CVaR | Durchschnitt unter VaR |
| `risk_cvar_1_50/100` | 99% CVaR | Durchschnitt unter extremem VaR |
| `risk_cvar_tail_ratio` | CVaR1 / CVaR5 | Wie viel schlimmer sind Extremevents? |
| `risk_cvar_5_change` | Änderung des CVaR | Verschlechtert sich Tail-Risk? |

**CVaR > VaR:** CVaR berücksichtigt die Größe der Tail-Verluste, nicht nur den Threshold

### Vol-of-Vol (Volatility of Volatility)

Stabilität der Volatilität selbst.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `risk_vol_of_vol_20` | Vol-of-Vol, 20-Bar | Kurzfristige Instabilität |
| `risk_vol_of_vol_50` | Vol-of-Vol, 50-Bar | Mittelfristige Instabilität |
| `risk_vol_of_vol_100` | Vol-of-Vol, 100-Bar | Langfristige Instabilität |
| `risk_vol_of_vol_zscore` | Z-Score des Vol-of-Vol | Relativ zur Historie |
| `risk_vol_of_vol_trend` | Änderung des Vol-of-Vol | Steigt Instabilität? |

**Anwendung:** Hohe Vol-of-Vol = Regime-Unsicherheit = Vorsicht geboten

### Crash Probability Proxy

Kombiniert mehrere Warnsignale zu einem Score.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `risk_crash_probability` | Aggregierter Crash-Score | 0-1, Crash-Wahrscheinlichkeit |
| `risk_crash_prob_change` | Änderung des Scores | Steigt das Risiko? |
| `risk_crash_regime` | Binäres Signal | 1 wenn Score > 0.6 |

**Komponenten:**
- Hohe Kurtosis (Fat Tails)
- Steigende Vol-of-Vol
- Correlation Decoupling
- Extreme CVaR

**Anwendung:** Hoher Score = erhöhte Vorsicht, kleinere Positionen oder keine Trades

---

## 19. Microstructure Features (`micro_`)
Analysieren die Intrabar-Dynamik und Orderflow-Muster.

**Plugin:** `microstructure`

### Wick Imbalance

Misst das Ungleichgewicht zwischen oberem und unterem Docht.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `micro_wick_imbalance` | (Oberer Docht - Unterer Docht) / Range | -1 bis 1, positiv = Verkaufsdruck oben |
| `micro_wick_imbalance_sum_N` | Rolling Sum über N Bars | Kumulierter Druck |

**Werte:**
- Positiv: Mehr Verkaufsdruck (langer oberer Docht)
- Negativ: Mehr Kaufdruck (langer unterer Docht)
- Nahe 0: Ausgewogene Kerze

### Intrabar Bias

Zeigt, wo der Close relativ zur Bar-Range liegt.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `micro_intrabar_bias` | (Close - Low) / Range | 0-1, 1 = Close am High |
| `micro_intrabar_bias_sum_N` | Rolling Sum über N Bars | Kumulierter Bias |

### Body Ratio

Verhältnis von Kerzenkörper zur Gesamtrange.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `micro_body_ratio` | |Open - Close| / Range | 0-1, 1 = keine Dochte |
| `micro_body_ratio_avg_N` | Rolling Average | Durchschnittliche Kerzenqualität |

**Interpretation:**
- Hoher Body Ratio: Starke, entschiedene Bewegung
- Niedriger Body Ratio: Unentschlossen, viel Hin und Her (Doji-ähnlich)

### Range over ATR

Normalisierte Bar-Range.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `micro_range_over_atr` | (High - Low) / ATR | Relative Größe der Bar |
| `micro_range_over_atr_max_N` | Rolling Maximum | Größte Bar im Fenster |

**Werte:**
- > 1: Überdurchschnittlich große Bar (Breakout?)
- < 1: Unterdurchschnittlich kleine Bar (Konsolidierung)

### Pressure Score

Kombinierter Orderflow-Score.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `micro_pressure_score` | wick_imbalance × body_ratio | -1 bis 1 |
| `micro_pressure_score_sum_N` | Rolling Sum | Kumulierter Druck |

**Formel:** pressure = wick_imbalance × body_ratio

**Interpretation:**
- Stark positiv: Starker Verkaufsdruck mit klarer Richtung
- Stark negativ: Starker Kaufdruck mit klarer Richtung
- Nahe 0: Kein klarer Druck

### Volume-gewichtete Features (falls Volume verfügbar)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `micro_vol_wick_imbalance` | Volume × Wick Imbalance | Volume-bestätigter Druck |
| `micro_vol_pressure` | Volume × Pressure Score | Volume-bestätigter Score |

### Konfigurierbare Parameter

```json
{
  "indicator_params": {
    "microstructure": {
      "atr_period": 14,
      "rolling_window": 5
    }
  }
}
```

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `atr_period` | 14 | ATR-Periode für Normalisierung |
| `rolling_window` | 5 | Fenster für Rolling-Aggregationen |

---

## 20. Macro Surprise Features (`macsurp_`)
Analysieren Gap-Verhalten und Informationsflüsse zwischen Sessions.

**Plugin:** `macro_surprise`

### Gap Analysis

Misst Overnight-Gaps und deren Bedeutung.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `macsurp_gap` | (Open - Previous Close) / ATR | Normalisierte Gap-Größe |
| `macsurp_gap_abs` | Absolute Gap (ohne Vorzeichen) | Gap-Magnitude |
| `macsurp_gap_ma` | Moving Average der Gaps | Durchschnittliche Gap-Aktivität |
| `macsurp_gap_std` | Standardabweichung der Gaps | Gap-Volatilität |

**Interpretation:**
- Große positive Gaps: Bullische Overnight-News
- Große negative Gaps: Bearische Overnight-News
- Hohe Gap-Volatilität: Unruhige Märkte, News-getrieben

### Return Decomposition

Zerlegt Returns in Gap und Intraday-Komponenten.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `macsurp_intraday_return` | (Close - Open) / Open | Intraday-Performance |
| `macsurp_total_return` | (Close - Prev Close) / Prev Close | Gesamt-Performance |
| `macsurp_gap_ratio` | Gap / Total Return | Anteil des Gaps am Return |

**Anwendung:**
- `gap_ratio` nahe 1: Return komplett durch Gap bestimmt
- `gap_ratio` nahe 0: Return durch Intraday-Trading bestimmt
- Negative Ratio: Gap und Intraday gegenläufig

### Surprise Detection

Erkennt unerwartete Marktbewegungen.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `macsurp_surprise` | |Gap| > threshold × rolling_std | Binär: 1 = Überraschung |
| `macsurp_surprise_direction` | Gap-Richtung bei Surprise | 1 = bullish, -1 = bearish |
| `macsurp_bars_since_surprise` | Bars seit letzter Surprise | Marktberuhigung |

**threshold** wird konfiguriert (Default: 2.0 = 2 Standardabweichungen)

### Volatility Breaks

Erkennt ungewöhnliche Intraday-Volatilität.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `macsurp_vol_break` | Range > 2 × rolling_avg_range | Binär: Volatility Break |
| `macsurp_vol_break_strength` | Range / rolling_avg_range | Stärke des Breaks |

### Konfigurierbare Parameter

```json
{
  "indicator_params": {
    "macro_surprise": {
      "vol_lookback": 20,
      "surprise_threshold": 2.0,
      "gap_ma_period": 10
    }
  }
}
```

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `vol_lookback` | 20 | Lookback für Rolling-Statistiken |
| `surprise_threshold` | 2.0 | Anzahl Std-Abweichungen für Surprise |
| `gap_ma_period` | 10 | MA-Periode für Gap-Glättung |

---

## 21. Makro Features (`macro_`)

Fundamentale Marktdaten von externen Quellen. Jeder Indikator generiert mehrere Features mit verschiedenen Lookback-Perioden.

### Volatilitäts-Indizes

| Indikator | Features | Beschreibung |
|-----------|----------|--------------|
| VIX | `macro_vix`, `macro_vix_chg_*` | S&P 500 Volatilität |
| VVIX | `macro_vvix`, `macro_vvix_chg_*` | Volatilität des VIX |
| SKEW | `macro_skew`, `macro_skew_chg_*` | Tail-Risk Indikator |
| VXN | `macro_vxn`, `macro_vxn_chg_*` | NASDAQ Volatilität |

### Zinsen (Yields)

| Indikator | Features | Beschreibung |
|-----------|----------|--------------|
| TNX | `macro_tnx`, `macro_tnx_chg_*` | 10-Year Treasury |
| TYX | `macro_tyx`, `macro_tyx_chg_*` | 30-Year Treasury |
| FVX | `macro_fvx`, `macro_fvx_chg_*` | 5-Year Treasury |
| IRX | `macro_irx`, `macro_irx_chg_*` | 13-Week Treasury |

### Währungen

| Indikator | Features | Beschreibung |
|-----------|----------|--------------|
| DXY | `macro_dxy`, `macro_dxy_chg_*` | US Dollar Index |

### Aktienindizes

| Indikator | Features | Beschreibung |
|-----------|----------|--------------|
| SPX | `macro_spx`, `macro_spx_chg_*` | S&P 500 |
| NASDAQ | `macro_nasdaq`, `macro_nasdaq_chg_*` | NASDAQ Composite |
| DOW | `macro_dow`, `macro_dow_chg_*` | Dow Jones |
| RUSSELL | `macro_russell`, `macro_russell_chg_*` | Russell 2000 |
| NIKKEI | `macro_nikkei`, `macro_nikkei_chg_*` | Nikkei 225 |
| HANGSENG | `macro_hangseng`, `macro_hangseng_chg_*` | Hang Seng |
| FTSE | `macro_ftse_idx`, `macro_ftse_idx_chg_*` | FTSE 100 |
| DAX | `macro_dax_idx`, `macro_dax_idx_chg_*` | DAX 40 |

### Sektoren (ETFs)

| Indikator | Features | Beschreibung |
|-----------|----------|--------------|
| XLF | `macro_xlf`, `macro_xlf_chg_*` | Financials |
| XLE | `macro_xle`, `macro_xle_chg_*` | Energy |
| XLK | `macro_xlk`, `macro_xlk_chg_*` | Technology |
| XLU | `macro_xlu`, `macro_xlu_chg_*` | Utilities |
| XLP | `macro_xlp`, `macro_xlp_chg_*` | Consumer Staples |

### Bonds

| Indikator | Features | Beschreibung |
|-----------|----------|--------------|
| TLT | `macro_tlt`, `macro_tlt_chg_*` | 20+ Year Treasury |
| HYG | `macro_hyg`, `macro_hyg_chg_*` | High Yield Corporate |
| LQD | `macro_lqd`, `macro_lqd_chg_*` | Investment Grade Corporate |

### Rohstoffe

| Indikator | Features | Beschreibung |
|-----------|----------|--------------|
| GOLD_FUT | `macro_gold_fut`, `macro_gold_fut_chg_*` | Gold Futures |
| OIL_FUT | `macro_oil`, `macro_oil_chg_*` | Crude Oil Futures |
| SILVER_FUT | `macro_silver_fut`, `macro_silver_fut_chg_*` | Silver Futures |

### Lookback-Perioden

Für jeden Makro-Indikator werden Change-Features mit folgenden Lookbacks generiert:

**Stunden:** 1h, 2h, 4h, 8h, 12h, 24h
**Tage:** 2d, 5d, 10d, 20d, 60d

---

## 22. Fair Value Gap Features (`fvg_`)

Fair Value Gaps (FVG) sind Preislücken zwischen drei aufeinanderfolgenden Bars, die auf aggressives Kaufen/Verkaufen hindeuten.

| Feature | Beschreibung |
|---------|-------------|
| `fvg_bull_active` | Bullish FVG aktiv (aktuelle Bar liegt über dem Gap) |
| `fvg_bear_active` | Bearish FVG aktiv (aktuelle Bar liegt unter dem Gap) |
| `fvg_bull_dist` | Normalisierte Distanz zum nächsten Bull-FVG |
| `fvg_bear_dist` | Normalisierte Distanz zum nächsten Bear-FVG |
| `fvg_bull_size` | Größe des nächsten Bull-FVG (normalisiert) |
| `fvg_bear_size` | Größe des nächsten Bear-FVG (normalisiert) |
| `fvg_in_gap` | Preis befindet sich aktuell in einem FVG |
| `fvg_count` | Anzahl aktiver FVGs |

**Plugin:** `fwbg-core:fair_value_gap` | **Prefix:** `fvg_` | **8 Features**

---

## 23. Support/Resistance Features (`sr_`)

Support/Resistance-Zonen basierend auf Swing-Highs/Lows mit Clustering und Trend-Klassifikation.

### Basis S/R (Hourly)

| Feature | Beschreibung |
|---------|-------------|
| `sr_dist_nearest_support` | Normalisierte Distanz zur nächsten Support-Zone |
| `sr_dist_nearest_resistance` | Normalisierte Distanz zur nächsten Resistance-Zone |
| `sr_support_strength` | Stärke der Support-Zone (Touch-Count) |
| `sr_resistance_strength` | Stärke der Resistance-Zone (Touch-Count) |
| `sr_in_support_zone` | Preis in Support-Zone (0/1) |
| `sr_in_resistance_zone` | Preis in Resistance-Zone (0/1) |
| `sr_nearest_is_flip_zone` | Nächste Zone ist eine Flip-Zone (S→R oder R→S) |

### Daily S/R (`sr_d1_*`)

Gleiche 7 Features, berechnet auf Daily-Timeframe für stärkere Zonen.

### Trend-Klassifikation

| Feature | Beschreibung |
|---------|-------------|
| `sr_trend_class` | Trend: 1=Uptrend, -1=Downtrend, 0=Range |
| `sr_pullback_depth` | Tiefe des Pullbacks relativ zum Trend (0-1) |
| `sr_ma_alignment` | MA-Alignment Score (MA20/50/200 Stacking) |
| `sr_price_vs_ma20` | Preis relativ zu MA20 (normalisiert) |
| `sr_price_vs_ma50` | Preis relativ zu MA50 (normalisiert) |
| `sr_price_vs_ma200` | Preis relativ zu MA200 (normalisiert) |
| `sr_trend_break` | Trendlinie gebrochen (0/1) |

### Confluence (Trend + S/R)

| Feature | Beschreibung |
|---------|-------------|
| `sr_at_support_in_uptrend` | Am Support im Uptrend (Long-Setup) |
| `sr_at_resistance_in_downtrend` | An Resistance im Downtrend (Short-Setup) |
| `sr_at_support_in_range` | Am Support in Range |
| `sr_at_resistance_in_range` | An Resistance in Range |

### Range & Breakout

| Feature | Beschreibung |
|---------|-------------|
| `sr_range_width` | Breite der aktuellen Range (normalisiert) |
| `sr_range_position` | Position in der Range (0=Bottom, 1=Top) |
| `sr_breakout_up` | Ausbruch über Resistance (0/1) |
| `sr_breakout_down` | Ausbruch unter Support (0/1) |

### Flip Zones

| Feature | Beschreibung |
|---------|-------------|
| `sr_at_flipped_support` | An Zone die von Resistance zu Support gewechselt hat |
| `sr_at_flipped_resistance` | An Zone die von Support zu Resistance gewechselt hat |

**Plugin:** `fwbg-premium:support_resistance` | **Prefix:** `sr_` | **31 Features**

---

## Feature-Gruppen / Indicator Plugins

Das Plugin-System ermöglicht modulare Konfiguration von Indikatoren. Jeder Indikator ist ein separates Plugin mit eigenen konfigurierbaren Parametern.

### Verfügbare Indicator Plugins

| Plugin Name | Gruppe | Anzahl Features | Beschreibung |
|-------------|--------|-----------------|--------------|
| `trend` | trend | ~34 | ADX, EMA, SMA, MACD, CCI, Aroon, ER |
| `momentum` | momentum | ~16 | RSI, Stochastic, Williams %R, ROC, UO |
| `volatility` | volatility | ~14 | Bollinger, Keltner, Donchian, ATR |
| `ichimoku` | trend | ~8 | Ichimoku Cloud System |
| `price_action` | price_action | ~5 | Range, Higher Highs/Lows, Gaps |
| `time_season` | time | ~14 | Zeit und Saisonalität |
| `dynamics` | dynamics | ~27 | Änderungen, Lags, Beschleunigung |
| `multi_timeframe` | mtf | ~12 | H4/D1 Aggregation |
| `cross_features` | cross | ~3 | Kombinierte Signale |
| `distribution` | distribution | ~10 | Skewness, Kurtosis |
| `structure` | structure | ~20 | FFT, Path Efficiency, VWAP |
| `regime` | regime | ~6 | Hurst-Exponent |
| `risk` | risk | ~25 | Drawdown, CVaR, Vol-of-Vol |
| `microstructure` | microstructure | ~15 | Intrabar-Analyse, Orderflow |
| `macro_surprise` | macro_surprise | ~12 | Gap-Analyse, Surprises |
| `fair_value_gap` | fair_value_gap | 8 | Bull/Bear FVG, Distance, Size, Count |
| `support_resistance` | support_resistance | 31 | S/R Zones, Trend, Pullbacks, Breakouts |

### Plugin-Konfiguration in Strategy JSON

```json
{
  "indicators": [
    "trend",
    "momentum",
    "volatility",
    "microstructure",
    "macro_surprise"
  ],

  "indicator_params": {
    "trend": {
      "adx_periods": [14, 21],
      "ema_periods": [21, 50, 100]
    },
    "microstructure": {
      "atr_period": 14,
      "rolling_window": 5
    },
    "macro_surprise": {
      "vol_lookback": 20,
      "surprise_threshold": 2.0
    }
  }
}
```

### Legacy Feature-Gruppen (Backward Compatibility)

| Gruppe | Prefixes | Beschreibung |
|--------|----------|--------------|
| `trend_momentum` | `trend_`, `ichi_`, `mom_` | Klassische technische Analyse |
| `macro_vol` | `macro_`, `vol_` | Fundamentale + Volatilitäts-Signale |
| `full_technical` | `trend_`, `ichi_`, `mom_`, `vol_`, `pa_` | Alle technischen Indikatoren |

---

## Preprocessing-Optionen

Daten-Preprocessing wird **vor** Feature-Berechnung auf OHLC-Daten angewendet.

### Fractional Differentiation
Macht Zeitreihen stationär unter Beibehaltung von Memory (nach López de Prado).

**Plugin-Name:** `fractional_diff`

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

**Parameter:**
- `auto_d` (bool): Automatische d-Optimierung via ADF-Test (default: `true`)
  - ⚠️ **WARNUNG**: `auto_d=true` verursacht Lookahead Bias! Der ADF-Test läuft auf dem GESAMTEN Datensatz (inkl. Future-Daten). Für valide Backtests: `auto_d=false` verwenden.
- `default_d` (float): Fixer d-Wert (default: `0.4`)
  - d=0: Keine Transformation (original)
  - d=1: Volle Differentiation (verliert Memory)
  - d=0.3-0.5: Optimal für Trading (stationär + Memory)
- `columns` (list): Zu transformierende Spalten (default: `["O", "H", "L", "C"]`)

**Wann nützlich:**
- Bei nicht-stationären Zeitreihen (Trends, Mean-Reversion)
- Verbessert ML-Modell-Performance durch stationäre Features
- Besonders wertvoll bei längeren Lookback-Perioden

---

## Feature Selection

Das System unterstützt verschiedene Methoden zur automatischen Feature-Auswahl.

### Boruta (Default)
Boruta ist ein "All-Relevant" Feature Selection Algorithmus:

1. Erstellt **Shadow-Features** (permutierte Kopien aller Features)
2. Trainiert XGBoost auf Original + Shadow Features
3. Berechnet **Z-Score** jedes Features vs. bestes Shadow-Feature
4. Features mit Z-Score signifikant über Shadow = **bestätigt relevant**
5. **Kein hartes Feature-Limit** - alle relevanten Features werden genutzt

**Vorteile:**
- Findet alle statistisch relevanten Features
- Robust gegen Noise (Zufalls-Features werden erkannt und verworfen)
- Keine willkürliche Limitierung auf Top-N

```json
{
  "features": {
    "feature_selection": "boruta"
  }
}
```

### Boruta + Plateau (Kombination)

Kombiniert die Stärken beider Methoden:

1. **Boruta** findet alle relevanten Features
2. **Plateau-Validierung** filtert instabile Lookback-Perioden

Ein Feature gilt als stabil wenn benachbarte Lookback-Perioden (z.B. RSI_12, RSI_14, RSI_16) ähnliche Importance haben.

**Vorteile:**
- Boruta findet echte Signale
- Plateau verhindert Overfitting auf spezifische Perioden

```json
{
  "features": {
    "feature_selection": "boruta_plateau"
  }
}
```

### Importance-Based (Legacy)

Das alte Verhalten mit festem top_n=5 Limit pro Feature-Gruppe:

1. Trainiert XGBoost-Modell
2. Berechnet Feature Importances
3. Plateau-Score für Stabilität
4. Wählt **Top-5 Features** pro Gruppe

```json
{
  "features": {
    "feature_selection": "importance_based"
  }
}
```

### Correlation Filter
Greedy Korrelationsfilter: Entfernt redundante Features die hoch miteinander korrelieren. Designed als Nachbearbeitung nach importance-basierter Selektion (z.B. Stability Boruta).

**Problem:** Boruta/Stability selektiert oft viele Makro-Indikatoren die dasselbe messen (VIX, VVIX, SKEW, VXN, ...) — hoch korreliert, aber als separate Features gezählt.

**Algorithmus:**
1. Iteriert Features in Input-Reihenfolge (wichtigste zuerst)
2. Behält Feature nur wenn |corr| < `max_correlation` mit allen bereits behaltenen Features
3. Optional: Hard Cap via `max_features`

**Plugin-Name:** `correlation_filter`

```json
{
  "pipeline": {
    "feature_selection": [
      {"name": "stability", "params": {
        "inner_selector": "boruta",
        "inner_params": {"n_iter": 5, "n_estimators": 30, "max_depth": 4, "min_z_score": 0.5},
        "n_bootstrap": 7, "threshold": 0.6, "bootstrap_ratio": 0.8
      }},
      {"name": "correlation_filter", "params": {"max_correlation": 0.7, "max_features": 20}}
    ]
  }
}
```

**Empfohlene Pipeline:** Stability Boruta (ohne `max_features`) selektiert alle robusten Features, dann entfernt correlation_filter redundante und setzt den Hard Cap.

### Vergleich der Methoden

| Methode | Feature-Limit | Redundanz-Filter | Stabilität | Anwendungsfall |
|---------|---------------|-----------------|------------|----------------|
| `stability` + `correlation_filter` | Via correlation_filter | Ja | Sehr hoch | **Empfohlen** — robuste, orthogonale Features |
| `boruta` | Kein Limit | Nein | Mittel | Standard, maximale Information |
| `boruta_plateau` | Kein Limit | Nein | Hoch | Wenn Overfitting ein Problem ist |
| `importance_based` | Top-5 | Nein | Hoch | Legacy, schneller |

**Referenz:** Kursa & Rudnicki (2010) "Feature Selection with the Boruta Package"

---

## Early Termination & Grid-Optimierung

Das Optimizer-System verfügt über intelligente Early-Termination-Mechanismen, um hoffnungslose Kandidaten frühzeitig abzubrechen und Rechenzeit zu sparen.

### Early Termination (Fold-Stability)

Bricht die Evaluation eines Kandidaten ab, wenn mathematisch nicht mehr genug profitable Folds erreicht werden können.

**Funktionsweise:**
1. Für jeden Grid-Kandidaten werden N Inner-Folds evaluiert
2. Ein Fold gilt als "profitabel" wenn PnL > 0
3. `min_fold_stability` definiert den Mindestanteil profitabler Folds (Default: 50%)
4. Nach jedem Fold wird geprüft: Können wir noch genug profitable Folds sammeln?
5. Wenn mathematisch unmöglich → Abbruch

**Beispiel mit 5 Folds und min_fold_stability=0.5:**
- Benötigt: ceil(5 × 0.5) = 3 profitable Folds
- Nach 3 Verlusten in Folge: max. 2 Folds noch möglich → Abbruch

**Parameter in SimulationContext:**

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `min_fold_stability` | 0.5 | Mindestanteil profitabler Folds (50%) |
| `early_termination` | true | Early Termination aktivieren |

### First-Fold Sanity Check
Ein zusätzlicher Sicherheitsmechanismus, der nach dem ersten Fold prüft, ob das Ergebnis **katastrophal** ist.

**Wichtig:** Dieser Check ist bewusst sehr konservativ - nur extreme Fälle werden abgebrochen. Normal schlechte Folds werden durchgelassen, da sich spätere Folds erholen können.

**Ein Kandidat wird nur abgebrochen wenn ALLE drei Bedingungen erfüllt sind:**

1. **Win-Rate < 25%** - Weniger als jeder 4. Trade ist profitabel
2. **PnL < -10** - Deutlich negatives Ergebnis (stark im Minus)
3. **>= 5 Trades** - Genug Trades für statistisch sinnvolle Aussage

**Rationale:**
- Ein einziger schlechter Fold sollte nicht alles torpedieren
- Aber wenn der erste Fold katastrophal ist, sind weitere Folds Zeitverschwendung
- Die Schwellenwerte sind großzügig gewählt um False Positives zu vermeiden

**Parameter in SimulationContext:**

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `first_fold_sanity_check` | true | Sanity Check aktivieren |
| `first_fold_min_win_rate` | 0.25 | Minimum Win-Rate (25%) |
| `first_fold_min_pnl` | -10.0 | Minimum PnL (sehr großzügig) |
| `first_fold_min_trades` | 5 | Minimum Trades für Aussagekraft |

### Logging

Bei aktiviertem Debug-Logging (OPTIMIZER_LOG >= 3) werden Abbrüche geloggt:

```
    Early terminated (3 failed folds)
    First-fold sanity check failed
```

### Zeitersparnis

Bei einem Grid mit 1000+ Kombinationen und 5 Folds kann Early Termination die Laufzeit um 30-50% reduzieren, da hoffnungslose Kandidaten nach 2-3 Folds abgebrochen werden statt alle 5 durchzulaufen.

### Deaktivierung

Falls gewünscht, können beide Mechanismen deaktiviert werden:

```python
# In einer benutzerdefinierten SimulationContext-Erstellung:
ctx.early_termination = False
ctx.first_fold_sanity_check = False
```

**Empfehlung:** Beide Mechanismen aktiviert lassen für optimale Performance.
