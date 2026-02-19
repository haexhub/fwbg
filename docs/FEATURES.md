# Feature & Indikator Dokumentation

Dieses Dokument beschreibt alle Features und Indikatoren, die dem ML-Modell zur Verfügung stehen.

**Gesamt: ~400+ Features** aus 28 Indicator Plugins (abhängig von Konfiguration und verfügbaren Makro-Daten)

---

## Inhaltsverzeichnis

### Klassische Technische Analyse
1. [Trend Indikatoren](#1-trend-indikatoren-trend_) — ADX, EMA, MACD, Supertrend
2. [Momentum Indikatoren](#2-momentum-indikatoren-mom_) — RSI, Stochastic, ROC
3. [Volatilität Indikatoren](#3-volatilität-indikatoren-vol_) — ATR, Bollinger, GK/PK/YZ Vol
4. [Price Action Features](#6-price-action-features-pa_) — Kerzenstruktur, Gaps, Volume

### Statistische & Frequenz-Analyse
5. [Distribution Features](#4-distribution-features-dist_) — Skewness, Kurtosis
6. [FFT Features](#5-fft-features-fft_) — Fourier-Zyklen
7. [Struktur Features](#16-struktur-features-path_-fractal_-convex_-structure_) — Path Efficiency, Convexity, VWAP

### Zeit & Saisonalität
8. [Zeit Features](#7-zeit-features-time_) — Intraday-Sessions
9. [Saisonalität Features](#8-saisonalität-features-season_) — Kalender-Effekte
10. [Calendar Event Features](#25-calendar-event-features-cal_) — Turn-of-Month, OpEx, FOMC

### Dynamik & Meta-Features
11. [Dynamik Features](#9-dynamik-features-dyn_) — Indikator-Änderungen
12. [Lag Features](#10-lag-features-lag_) — Historische Zustände
13. [Beschleunigung Features](#11-beschleunigung-features-accel_) — Zweite Ableitung
14. [Cross-Indikator Features](#12-cross-indikator-features-cross_) — Confluence

### Multi-Timeframe & Session
15. [Multi-Timeframe Features](#13-multi-timeframe-features-mtf_) — H4/D1 Aggregation
16. [Opening Range Features](#31-opening-range-features-orb_) — ORB, Session-Breakouts

### Regime & Marktstruktur
17. [Regime Features](#14-regime-features-regime_) — Hurst-Exponent
18. [Market Regime Features](#33-market-regime-features-regime_risk_) — Risk-On/Off Composite
19. [Regime Cluster Features](#34-regime-cluster-features-regime_cluster_) — 3-State Clustering

### Event & Level Detection
20. [Event Features](#15-event-features-event_) — Bars-Since-Event
21. [CUSUM Event Features](#24-cusum-event-features-cusum_) — Structural Breaks
22. [Fair Value Gap Features](#22-fair-value-gap-features-fvg_) — Institutionelle Gaps
23. [Support/Resistance Features](#23-supportresistance-features-sr_) — S/R Zones, Flip-Zones

### Risk & Korrelation
24. [Risk/Tail-Risk Features](#18-risktail-risk-features-risk_) — CVaR, Crash-Prob
25. [Korrelations Features](#17-korrelations-features-corr_-lead_lag_) — SPX/VIX Korrelation
26. [Microstructure Features](#19-microstructure-features-micro_) — Orderflow, Pressure

### Makro & Externe Daten
27. [Macro Surprise Features](#20-macro-surprise-features-macsurp_) — Gap-Analyse, Surprises
28. [Makro Features](#21-makro-features-macro_) — VIX, Zinsen, Indices, Rohstoffe

### Fortgeschrittene Methoden
29. [Ichimoku Cloud Features](#32-ichimoku-cloud-features-ichi_) — Vollständiges Trading-System
30. [Fractal Dimension Features](#26-fractal-dimension-features-fd_) — Higuchi FD
31. [Wavelet Features](#27-wavelet-features-wt_) — DWT Dekomposition
32. [Autoencoder / PCA Features](#28-autoencoder--pca-features-ae_) — Latent Features
33. [Topological Data Analysis](#29-topological-data-analysis-features-tda_) — Persistent Homology
34. [Adversarial Validation](#30-adversarial-validation-features-adv_) — Distribution Shift

### Referenz
35. [Feature-Gruppen / Indicator Plugins](#feature-gruppen--indicator-plugins) — Plugin-Übersicht
36. [Early Termination & Grid-Optimierung](#early-termination--grid-optimierung)

---

## 1. Trend Indikatoren (`trend_`)

**Plugin:** `fwbg-core:trend` | **~34 Features**

Trend-Indikatoren beantworten drei Kernfragen: *Gibt es einen Trend?* (ADX), *Wie stark ist er?* (EMA/SMA-Distanz, Efficiency Ratio), und *Wohin geht er?* (MACD, Aroon, Supertrend). Die Kombination mehrerer Methoden reduziert Fehlsignale: ADX misst Trendstärke richtungsunabhängig, MACD erfasst Momentum-Divergenzen, und der Supertrend liefert klare Ein-/Ausstiegssignale über ATR-basierte Bänder.

**Trading-Relevanz:** Trend-Following ist die profitabelste Strategie-Klasse — aber nur in trendenden Märkten. ADX > 25 signalisiert, dass Breakout-Strategien funktionieren; ADX < 20 warnt vor Mean-Reversion-Phasen. EMA-Distanzen zeigen Überdehnung (Mean-Reversion-Setup) oder Unterstützung (Trend-Continuation). Die Efficiency Ratio unterscheidet saubere Trends (ER → 1) von choppy Seitwärtsbewegungen (ER → 0).

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

### Supertrend
ATR-basierter Trend-Filter mit dynamischen Bändern. Flipped nur in Trendrichtung — reduziert Whipsaws in Seitwärtsmärkten.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `trend_supertrend` | Supertrend-Band (normalisiert) | Dynamischer Support/Resistance |
| `trend_supertrend_dir` | Trend-Richtung | 1 = bullish, -1 = bearish |
| `trend_supertrend_dist` | Abstand zum Band (%) | Trend-Stärke |
| `trend_supertrend_flip` | Band-Flip erkannt | 1 = Trendwechsel |

**Parameter:** `supertrend_period` (default: 14), `supertrend_multiplier` (default: 3.0)

---

## 2. Momentum Indikatoren (`mom_`)

**Plugin:** `fwbg-core:momentum` | **~16 Features**

Momentum-Indikatoren messen die *Geschwindigkeit* von Preisbewegungen und identifizieren überkaufte/überverkaufte Zustände. RSI und Stochastic normalisieren Momentum auf feste Skalen (0-100), was ML-Modellen erlaubt, Extremwerte konsistent über verschiedene Assets und Zeiträume zu erkennen. Rate of Change (ROC) misst dagegen die reine prozentuale Veränderung ohne Normalisierung.

**Trading-Relevanz:** Momentum-Divergenzen (RSI fällt, Preis steigt) sind klassische Warnsignale für Trenderschöpfung. In Range-Märkten funktionieren Overbought/Oversold-Signale gut; in starken Trends kann RSI wochen­lang >70 bleiben — daher immer in Kombination mit Trend-Indikatoren nutzen. Der Ultimate Oscillator kombiniert drei Zeitebenen und reduziert so False Signals einzelner Perioden.

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

**Plugin:** `fwbg-core:volatility` | **~14 Features**

Volatilitäts-Indikatoren quantifizieren die Schwankungsbreite des Marktes und identifizieren Phasen der Kompression (Setup für große Bewegungen) und Expansion. Neben dem klassischen ATR verwendet das Plugin fortgeschrittene Volatilitätsschätzer: **Garman-Klass** nutzt alle vier OHLC-Preise für effizientere Schätzung, **Parkinson** fokussiert auf die High-Low-Range, und **Yang-Zhang** ist der robusteste Schätzer, der Overnight-Gaps, Close-to-Close-Moves und Rogers-Satchell-Varianz kombiniert.

**Trading-Relevanz:** ATR bestimmt direkt die Positionsgröße und Stop-Loss-Distanz. Volatilitätskompression (ATR + BB-Breite gleichzeitig unter 20. Perzentil) signalisiert bevorstehende Breakouts — klassisches Squeeze-Setup. Verschiedene Vol-Schätzer reagieren unterschiedlich auf Marktphasen: Garman-Klass ist effizient bei Normal­bedingungen, Yang-Zhang robust bei Overnight-Gaps. Realized Vol vs. Implied Vol (VIX) Spread zeigt Fehlbepreisung im Options­markt.

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

### Volatilitäts-Schätzer

Fortgeschrittene OHLC-basierte Volatilitätsschätzer, die effizienter als Close-to-Close-Varianz sind.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `vol_gk_20/50` | Garman-Klass Volatilität | σ² = 0.5·ln(H/L)² − (2ln2−1)·ln(C/O)² |
| `vol_pk_20/50` | Parkinson Volatilität | σ² = ln(H/L)² / (4·ln2) |
| `vol_yz_20/50` | Yang-Zhang Volatilität | Kombiniert Overnight + Rogers-Satchell + Close-to-Close |
| `vol_gk_20_rank` | Garman-Klass Perzentil-Rang | 0-1, Position in der Verteilung |

### Volatilitäts-Kompression & Regime

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `vol_compression` | ATR14 + BB-Breite unter 20. Perzentil | 1 = Squeeze-Setup, Breakout erwartet |
| `vol_realized_20` | Annualisierte Realized Vol (20 Bars) | Log-Return basierte Volatilität |
| `vol_rv_iv_spread` | Realized Vol minus VIX (falls verfügbar) | Positiv = IV zu niedrig, Negativ = IV zu hoch |
| `vol_rv_iv_ratio` | Realized Vol / VIX | <1 = IV-Prämie, >1 = RV dominiert |

### Volume-basiert (falls verfügbar)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `vol_obv_change` | OBV Änderung (5 Perioden) | Volume-Momentum |
| `vol_mfi` | Money Flow Index | Volume-gewichteter RSI |

---

## 4. Distribution Features (`dist_`)

**Plugin:** `fwbg-core:distribution` | **~10 Features**

Distribution-Features analysieren die statistische Verteilung der Returns jenseits von Mittelwert und Varianz. Finanzmärkte sind nicht normalverteilt — sie zeigen Fat Tails (Kurtosis > 3) und asymmetrische Schiefen, die für Risikomodellierung entscheidend sind.

**Trading-Relevanz:** Negative Schiefe warnt vor asymmetrischem Downside-Risiko (häufige kleine Gewinne, seltene große Verluste). Hohe Kurtosis signalisiert Fat-Tail-Regime — hier sind größere Stops nötig, da Extrembewegungen wahrscheinlicher sind. Z-Score-normalisierte Versionen zeigen, ob die aktuelle Verteilung *relativ zur jüngsten Geschichte* extrem ist, nicht absolut.

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

**Enthalten in Plugin:** `fwbg-core:structure`

Fourier-Transformation zur Zykluserkennung. Die FFT zerlegt Preisbewegungen in Frequenzkomponenten und zeigt, welche Zykluslängen aktuell dominieren. Im Gegensatz zu Wavelets (→ Sektion 27) liefert FFT *globale* Frequenzinformation ohne Zeitlokalisierung.

**Trading-Relevanz:** Märkte durchlaufen zyklische Phasen — nicht perfekt periodisch, aber mit dominanten Frequenzen, die über Rolling-Fenster stabil sein können. Niedrige spektrale Entropie zeigt klare, dominante Zyklen (gut für zyklische Strategien); hohe Entropie bedeutet Rauschen (kein zyklischer Edge). Die Low-Frequency Ratio identifiziert trendende Phasen, in denen langfristige Zyklen dominieren.

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

**Plugin:** `fwbg-core:price_action` | **~17 Features**

Price Action Features kodieren die Kerzenstruktur und Bar-zu-Bar-Beziehungen ohne gleitende Durchschnitte oder Lookback-Perioden. Sie messen *wer die Bar gewonnen hat* (Range Position, Body Ratio), *wie die Struktur aussieht* (Higher Highs/Lows, Trend Structure Score), und *ob Volume die Bewegung bestätigt* (OBV, MFI, Relative Volume).

**Trading-Relevanz:** Price Action ist die direkteste Form von Markt-Feedback — keine Lag durch Averaging. Body Ratio nahe 1 zeigt Überzeugung (starker Kerzenkörper, keine Dochte); nahe 0 zeigt Unentschlossenheit (Doji). Konsekutive Higher Highs + Higher Lows bestätigen Trendqualität. Gaps zeigen Overnight-Informationsschocks. Volume-Features (falls verfügbar) unterscheiden echte Breakouts von Fehlausbrüchen.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `pa_range_pos` | Position in High-Low Range | 0 = Low, 1 = High |
| `pa_hh` | Higher Highs (letzte 5 Bars) | Aufwärtsdruck |
| `pa_ll` | Lower Lows (letzte 5 Bars) | Abwärtsdruck |
| `pa_body_ratio` | Body/Range Ratio | Kerzen-Stärke |
| `pa_gap` | Gap vom Vortag (%) | Overnight-Bewegung |
| `pa_body_dir` | Body-Richtung | +1 bullish, -1 bearish, 0 Doji |
| `pa_upper_shadow` | Oberer Docht / Range | Verkaufsdruck oben |
| `pa_lower_shadow` | Unterer Docht / Range | Kaufdruck unten |
| `pa_hl` | Higher Lows (letzte 5 Bars) | Aufwärtsstruktur-Qualität |
| `pa_lh` | Lower Highs (letzte 5 Bars) | Abwärtsstruktur-Qualität |
| `pa_trend_structure` | (HH+HL) - (LL+LH) | Positiv = bullish, Negativ = bearish |
| `pa_gap_dir` | Gap-Richtung | 1 = up, -1 = down, 0 = kein Gap |
| `pa_consec_bull` | Konsekutive bullishe Bars | Bullisher Streak |
| `pa_consec_bear` | Konsekutive bearishe Bars | Bearisher Streak |
| `pa_range_expansion` | Aktuelle Range / 20-Bar Avg | >1 = Expansion, <1 = Kontraktion |
| `pa_inside_bar` | Inside Bar erkannt | Volatilitätskontraktion |

---

## 7. Zeit Features (`time_`)

**Plugin:** `fwbg-core:time_season` | **~14 Features (zusammen mit Saisonalität)**

Zeit-Features kodieren Intraday-Muster und Trading-Sessions. Finanzmärkte zeigen ausgeprägte Tageszeit-Effekte: US-Open (14:30 UTC) bringt die höchste Volatilität, der Asien-Europa-Overlap (8-12 UTC) erhöhte Liquidität. Sin/Cos-Encoding ist kritisch für ML-Modelle — ohne zyklische Kodierung sehen Baummodelle Stunde 23 und 0 als maximal verschieden, obwohl sie benachbart sind.

**Trading-Relevanz:** Session-Timing bestimmt Spread, Liquidität und Volatilität. Breakout-Strategien funktionieren besser am Session-Open, Mean-Reversion besser in ruhigen Phasen. Wochentags-Effekte (Montag-Reversals, Freitag-Profit-Taking) sind empirisch gut dokumentiert.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `time_hour` | Stunde (0-23) | Session-Timing |
| `time_day` | Wochentag (0=Mo, 6=So) | Wochentags-Effekte |
| `time_hour_sin` | Zyklische Stunde (sin) | Für ML-Modelle |
| `time_hour_cos` | Zyklische Stunde (cos) | Für ML-Modelle |

---

## 8. Saisonalität Features (`season_`)

**Plugin:** `fwbg-core:time_season` (gleicher Plugin wie Zeit-Features)

Saisonale Features erfassen kalendrische Muster über Tage, Wochen, Monate und Quartale hinweg. Der **January Effect** (Small Caps outperformen im Januar), **Turn-of-Month** (~65% der monatlichen Returns fallen auf erste/letzte Tage), und **Quarter-End Rebalancing** (institutionelle Portfolio-Umschichtungen) sind akademisch gut dokumentierte Anomalien.

**Trading-Relevanz:** Institutionelle Flows folgen festen Kalendern — Month-End Window Dressing, Quarter-End Rebalancing, Year-End Tax-Loss Selling. ML-Modelle lernen diese Muster nur mit passenden Features. Sin/Cos-Kodierung ermöglicht, dass Dezember und Januar als benachbart erkannt werden.

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

**Plugin:** `fwbg-core:dynamics` | **~27 Features**

Dynamik-Features messen die *Veränderung* von Indikatoren über verschiedene Zeithorizonte. Während ein RSI von 65 allein wenig aussagt, zeigt ein RSI-Anstieg von 45→65 in 4 Stunden (dyn_rsi14_chg_4h) klares Momentum. Die Features umfassen Deltas (absolute Änderung), Lags (historische Werte) und Beschleunigungen (zweite Ableitung).

**Trading-Relevanz:** Die Änderungsrate eines Indikators ist oft prädiktiver als sein Absolutwert. Ein steigender ADX zeigt *entstehenden* Trend (frühes Signal), ein hoher aber fallender ADX zeigt *endenden* Trend. Beschleunigung (zweite Ableitung) erkennt Wendepunkte: wenn das Momentum des Momentums kippt, steht eine Trendumkehr bevor.

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

**Enthalten in Plugin:** `fwbg-core:dynamics`

Verzögerte Indikator-Werte. Lags geben dem ML-Modell Zugang zu historischen Zuständen — z.B. ob RSI vor 24 Stunden bereits überkauft war und der Markt seitdem korrigiert hat.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `lag_rsi14_4h/8h/24h` | RSI vor X Stunden | Historischer Zustand |
| `lag_atr_4h/8h/24h` | ATR vor X Stunden | Historische Volatilität |
| `lag_adx_4h/8h` | ADX vor X Stunden | Historischer Trend |
| `lag_price_chg_4h/8h/24h/48h` | Preisänderung seit X Stunden | Performance |

---

## 11. Beschleunigung Features (`accel_`)

**Enthalten in Plugin:** `fwbg-core:dynamics`

Zweite Ableitung — Änderung der Änderung. Erkennt Wendepunkte bevor sie in den Primär-Indikatoren sichtbar werden.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `accel_rsi` | RSI Beschleunigung | Momentum des Momentums |
| `accel_atr` | ATR Beschleunigung | Volatilität des Volatilitäts-Shifts |

---

## 12. Cross-Indikator Features (`cross_`)

**Plugin:** `fwbg-core:cross_features` | **~3 Features**

Cross-Features kombinieren Signale aus verschiedenen Indikator-Kategorien zu Confluence-Signalen. Anstatt isolierte Extremwerte zu betrachten, erkennen sie *gleichzeitige* Muster: RSI überkauft UND steigend, oder Volatilitätsanstieg IM Trend.

**Trading-Relevanz:** Einzelne Signale haben hohe False-Positive-Raten. Confluence (mehrere unabhängige Bestätigungen) reduziert diese drastisch. `cross_rsi_high_rising` filtert z.B. Situationen wo RSI >70 UND weiter steigt — ein stärkeres Signal als RSI >70 allein.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `cross_rsi_high_rising` | RSI>70 UND steigend | Überkauft + Momentum |
| `cross_rsi_low_falling` | RSI<30 UND fallend | Überverkauft + Momentum |
| `cross_vol_trend` | ATR-Änderung × ADX | Volatilität im Trend |

---

## 13. Multi-Timeframe Features (`mtf_`)

**Plugin:** `fwbg-premium:multi_timeframe` | **~25 Features**

Multi-Timeframe Features aggregieren H1-Daten zu höheren Zeitrahmen (H4, D1, W1) und messen die Übereinstimmung der Trends über Zeitebenen. Der **Trend-Consensus** Score (0-3) zählt wie viele Timeframes in die gleiche Richtung zeigen — volle Übereinstimmung ist das stärkste Confluence-Signal.

**Trading-Relevanz:** "Trade with the higher timeframe" ist eine Grundregel technischer Analyse. Ein H1-Long-Signal hat deutlich höhere Trefferquote, wenn D1 und H4 ebenfalls bullish sind. Volatilitäts-Ratios (H1/H4) zeigen, ob Intraday-Moves relativ zur übergeordneten Dynamik überdurchschnittlich sind (potentielle Übertreibung) oder unterdurchschnittlich (Range-Markt).

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

**Plugin:** `fwbg-core:regime` | **~6 Features**

Regime-Features klassifizieren den Marktcharakter über den Hurst-Exponenten — einen nicht-parametrischen Indikator der *Persistenz* von Preisbewegungen. Im Gegensatz zu Trend-Indikatoren, die die Richtung messen, misst Hurst ob der Markt überhaupt *trending-fähig* ist.

**Trading-Relevanz:** Die Wahl der Trading-Strategie hängt fundamental vom Regime ab. H > 0.5 (persistent) bevorzugt Trend-Following; H < 0.5 (anti-persistent) bevorzugt Mean-Reversion; H ≈ 0.5 (Random Walk) erschwert jede systematische Strategie. Die Hurst-Divergenz (kurzfristiger vs. langfristiger Hurst) erkennt Regime-Wechsel: wenn der kurzfristige Hurst plötzlich abweicht, steht eine strukturelle Veränderung bevor.

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

**Plugin:** `fwbg-premium:events` | **~12 Features**

Time-Since-Event Features messen, wie viele Bars seit wichtigen Ereignissen vergangen sind. Die Kernidee: *Kontext ist alles*. Ein Ausbruch nach 100 Bars Konsolidierung hat mehr aufgestaute Energie als einer nach 5 Bars.

**Trading-Relevanz:** Märkte "vergessen" Events nicht sofort — die Zeit seit dem letzten Extremereignis beeinflusst die Wahrscheinlichkeit des nächsten. Lange Konsolidierung (hohes `bars_since_high`) vor Breakout korreliert mit stärkeren, nachhaltigeren Bewegungen. Log-transformierte Versionen verbessern die ML-Performance, da die Verteilung rechtsschief ist.

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

**Plugin:** `fwbg-core:structure` | **~20 Features**

Struktur-Features analysieren die *mathematische Form* der Preisbewegung — nicht Richtung oder Stärke, sondern Effizienz, Komplexität und Krümmung. Path Efficiency misst ob sich der Preis "gerade" bewegt (Trend) oder "zickzackt" (Range). Convexity erkennt parabolische Beschleunigung. VWAP-Proxy liefert institutionelle Referenzpunkte.

**Trading-Relevanz:** Path Efficiency nahe 1 = sauberer Trend (Trend-Following profitabel). PE nahe 0 = choppy Markt (Mean-Reversion oder Abstinenz). Convexity-Divergenz (EMA21 vs EMA50 Convexity) warnt vor parabolischen Tops: wenn die kurzfristige Beschleunigung die langfristige deutlich übersteigt, ist Vorsicht geboten.

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

**Enthalten in Plugin:** `fwbg-premium:risk` (berechnet wenn Makro-Daten verfügbar)

Korrelations-Features analysieren die dynamische Beziehung eines Assets zu Benchmark-Märkten (SPX, VIX). Stabile Korrelationen zeigen "normales" Marktverhalten; plötzliche Entkopplung (Decoupling) ist ein Frühwarnsignal für fundamentale Veränderungen.

**Trading-Relevanz:** Correlations "break" in Krisen — genau wenn Diversifikation am wichtigsten wäre. Ein hohes `corr_spx_decoupling` warnt vor erhöhter Volatilität und möglichen Regime-Wechseln. Lead-Lag-Features erkennen, ob VIX oder SPX dem Asset vorauslaufen — nützlich für Early Positioning.

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

**Plugin:** `fwbg-premium:risk` | **~25 Features**

Risk-Features quantifizieren Tail-Risiko, Drawdown-Zustand und Marktstress-Level. Sie gehen weit über einfache Volatilität hinaus: **CVaR** (Conditional Value at Risk) misst den *erwarteten* Verlust im extremen Tail, **Vol-of-Vol** zeigt Regime-Unsicherheit, und der **Crash Probability Score** aggregiert multiple Stress-Signale zu einem einzigen Warnsignal.

**Trading-Relevanz:** Risiko-Management ist kein Overlay, sondern sollte *ins Modell integriert* sein. Ein ML-Modell mit Zugang zu CVaR-Features lernt automatisch, in Hochrisiko-Phasen konservativere Trades zu nehmen. Die Crash-Probability kombiniert Kurtosis, Vol-of-Vol, Correlation Decoupling und extreme CVaR — eine Art "Fear Gauge" die dem VIX ähnelt, aber asset-spezifisch ist.

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

**Plugin:** `fwbg-premium:microstructure` | **~17 Features**

Microstructure-Features extrahieren Orderflow-Signale aus OHLC-Bars, die in traditionellen Indikatoren verloren gehen. Durch Analyse von Dochten, Body-Ratio und deren Zusammenspiel mit Volume rekonstruieren sie das Kräfteverhältnis zwischen Käufern und Verkäufern *innerhalb* jeder Bar.

**Trading-Relevanz:** Institutionelle Trader hinterlassen Spuren in der Bar-Struktur: aggressive Sells erzeugen lange obere Dochte (Rejection), Accumulation zeigt sich in langen unteren Dochten mit Close nahe High. Der Pressure Score (wick_imbalance × body_ratio) kombiniert Richtung und Überzeugung zu einem einzigen Orderflow-Signal. CLV-basierte Accumulation/Distribution und Chaikin Money Flow (falls Volume verfügbar) bestätigen institutionelle Positionierung.

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

**Plugin:** `fwbg-premium:macro_surprise` | **~21 Features**

Macro Surprise Features detektieren Informationsschocks durch Analyse von Overnight-Gaps, Session-Dekomposition (Overnight vs. Intraday Returns) und unerwarteten Volatilitäts-Ausbrüchen. Sie messen nicht *was* passiert ist, sondern *ob es überraschend* war — relativ zur jüngsten Erwartung.

**Trading-Relevanz:** Märkte bewegen sich auf Überraschungen, nicht auf erwartete News. Ein großer Gap der *persistiert* zeigt starke Überzeugung (Follow-Through wahrscheinlich); ein schnell gefüllter Gap zeigt Fading-Momentum. Die Gap-Ratio (Gap / Total Return) zeigt ob der Tag von Overnight-News oder Intraday-Trading dominiert wird — wichtig für die Wahl zwischen Gap-Strategien und Session-Strategien.

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

**Plugin:** `fwbg-premium:macro_data` (Data Loader) | **~200+ Features**

Makro-Features sind fundamentale Marktdaten (VIX, Zinsen, Rohstoffe, Aktienindizes, Sektor-ETFs), die als externe Kontextvariablen geladen werden. Jeder Makro-Indikator wird in Change-Features über 11 Lookback-Perioden (1h bis 60d) transformiert, sodass das ML-Modell sowohl den aktuellen Stand als auch die Dynamik sieht.

**Trading-Relevanz:** Finanzmärkte sind vernetzt — EURUSD reagiert auf Treasury Yields, Gold auf Real Yields, Tech-Aktien auf VIX. Makro-Features geben dem Modell Zugang zu diesen Cross-Asset-Dynamiken. Besonders wertvoll: VIX-Änderungen als Leading Indicator (Fear precedes Price), Yield-Curve-Spreads für Rezessionswarnung, und Sektor-Rotation (XLU/XLP-Stärke = Risk-Off).

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

**Plugin:** `fwbg-core:fair_value_gap` | **8 Features**

Fair Value Gaps (FVG) sind Preislücken zwischen drei aufeinanderfolgenden Bars, die auf aggressives institutionelles Kaufen/Verkaufen hindeuten. Ein Bullish FVG entsteht, wenn das Low von Bar 3 über dem High von Bar 1 liegt — der Markt hat so schnell gekauft, dass die mittlere Bar keine Überlappung hat.

**Trading-Relevanz:** Institutionelle Algorithmen füllen FVGs oft nachträglich, da diese Zonen unvollständige Orderausführung repräsentieren. Der Preis kehrt häufig zum FVG zurück, bevor er in Trendrichtung weiterläuft — ein klassisches Pullback-Setup. Aktive FVG-Counts zeigen Marktdynamik: viele offene Gaps = schneller, impulsiver Markt.

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

**Plugin:** `fwbg-premium:support_resistance` | **31 Features**

Support/Resistance-Zonen basierend auf Swing-Highs/Lows mit DBSCAN-Clustering und Trend-Klassifikation. Erkennt automatisch S/R-Zonen, Flip-Zones (ehemaliger Support wird Resistance), Trendbrüche und Confluence-Setups.

**Trading-Relevanz:** S/R-Zonen sind die universellste Form technischer Analyse — sie funktionieren weil genug Trader danach handeln (Self-Fulfilling Prophecy). Flip-Zones sind besonders stark: wenn ehemaliger Support zu Resistance wird, zeigt das ein "Sentiment Flip". Die Confluence-Features (`sr_at_support_in_uptrend`) kombinieren Trend und S/R für High-Probability Setups.

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

## 24. CUSUM Event Features (`cusum_`)

CUSUM (Cumulative Sum) Structural Break Detection nach de Prado (AFML Ch. 2). Erkennt strukturelle Brüche im Preisprozess — Momente, in denen die kumulative Abweichung der Returns von ihrem Erwartungswert einen Schwellenwert überschreitet.

| Feature | Beschreibung |
|---------|-------------|
| `cusum_pos_event` | Positiver struktureller Bruch erkannt (0/1) |
| `cusum_neg_event` | Negativer struktureller Bruch erkannt (0/1) |
| `cusum_pos_value` | Aktuelle positive kumulative Summe (normalisiert, 0-1) |
| `cusum_neg_value` | Aktuelle negative kumulative Summe (normalisiert, 0-1) |
| `cusum_intensity` | Stärke des Events (Overshoot über Threshold, >= 1.0 bei Event) |
| `cusum_bars_since` | Bars seit letztem Event (normalisiert durch Lookback) |

**Algorithmus:**
```
S_pos[t] = max(0, S_pos[t-1] + (r[t] - E[r]))
S_neg[t] = min(0, S_neg[t-1] + (r[t] - E[r]))
Event wenn S_pos > h·σ oder S_neg < -h·σ (dann Reset)
```

**Parameter:**
- `threshold` (default: 1.5) — Multiplikator für Rolling-Standardabweichung
- `lookback` (default: 100) — Fenster für E[r] und σ Berechnung

**Plugin:** `fwbg-core:cusum_events` | **Prefix:** `cusum_` | **6 Features**

---

## 25. Calendar Event Features (`cal_`)

Kalender-Anomalien die auf gut dokumentierten systematischen Markteffekten basieren. Alle Features werden deterministisch aus dem DatetimeIndex berechnet — keine Preisdaten nötig.

### Binäre Event-Features

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `cal_turn_of_month` | Erste 3 / letzte 2 Tage des Monats | Turn-of-Month Effekt (~65% der monatlichen Returns) |
| `cal_quarter_end` | Letzte 5 Tage von Mar/Jun/Sep/Dec | Portfolio-Rebalancing, Window Dressing |
| `cal_triple_witching` | ±2 Tage um 3. Freitag von Mar/Jun/Sep/Dec | Triple Witching (Futures + Options Verfall) |
| `cal_monthly_opex` | ±2 Tage um 3. Freitag jeden Monats | Monatlicher Options-Verfall |
| `cal_nfp_week` | Erste 5 Tage jeden Monats | Non-Farm Payrolls Woche |
| `cal_year_boundary` | Letzte 5 Tage Dec / Erste 5 Tage Jan | January Effect, Year-End Rebalancing |

### Proximity-Features

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `cal_days_to_month_end` | Tage bis Monatsende (normalisiert 0..1) | Nähe zum Monatsende |
| `cal_fomc_proximity` | Zyklische Sinus-Approximation (~46-Tage FOMC Zyklus) | Nähe zum nächsten FOMC Meeting |
| `cal_week_of_month` | Woche im Monat (normalisiert 0..1) | Position innerhalb des Monats |

**Parameter:**
- `include_binary` (default: true) — Binäre Event-Features einschließen
- `include_proximity` (default: true) — Kontinuierliche Proximity-Features einschließen

**Plugin:** `fwbg-core:calendar_events` | **Prefix:** `cal_` | **9 Features**

---

## 26. Fractal Dimension Features (`fd_`)

Higuchi Fractal Dimension (HFD) — misst Komplexität und Rauheit der Preisreihe. Ergänzt den Hurst-Exponenten: Hurst misst Persistenz, FD misst Komplexität.

**Algorithmus:** Für ein Fenster der Länge N mit k_max Intervallen:
1. Konstruiere k Sub-Serien für jedes k in 1..k_max
2. Berechne normalisierte Länge L(k) jeder Sub-Serie
3. D = Steigung von log(L(k)) vs log(1/k)

### Features pro Window

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `fd_higuchi_{w}` | Rohe Fraktaldimension | 1.0 = glatt/trendend, 1.5 = Random Walk, 2.0 = komplex/rauschend |
| `fd_higuchi_change_{w}` | Änderung der FD über Fenster | Regime-Transition Erkennung |
| `fd_complexity_ratio_{w}` | Abstand von 1.5: `abs(fd-1.5)*2` | 0 = zufällig, 1 = strukturiert (Trend oder Mean-Reversion) |
| `fd_regime_{w}` | Diskretisiert: -1/0/1 | -1 = trendend (FD<1.4), 0 = zufällig, 1 = mean-reverting (FD>1.6) |

**Werte:** FD ∈ [1.0, 2.0]. Interpretation spiegelt Hurst: FD ≈ 2 - H (approximativ).

**Parameter:**
- `windows` (default: [50, 100, 200]) — Rolling-Fenster für FD-Berechnung
- `k_max` (default: 10) — Max. Intervall im Higuchi-Algorithmus

**Plugin:** `fwbg-core:fractal_dimension` | **Prefix:** `fd_` | **4 Features × N Windows = 12 Features (default)**

---

## 27. Wavelet Features (`wt_`)

Discrete Wavelet Transform (DWT) Dekomposition von Log-Returns. Im Gegensatz zu FFT liefern Wavelets **Zeit-und-Frequenz-Lokalisierung** — entscheidend für nicht-stationäre Finanzsignale.

**Algorithmus:** Mehrstufige DWT-Zerlegung via `pywt.wavedec()` (Default: Daubechies-4). Jede Stufe halbiert die Frequenz:
- **Detail 1** = höchste Frequenz (Noise, kurzfristige Schwankungen)
- **Detail 2** = mittlere Frequenz
- **Detail 3** = niedrige Frequenz (mittelfristige Zyklen)
- **Approximation** = Trend-Komponente

### Energy Features (pro Level × Window)

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `wt_detail_{lvl}_energy_{w}` | Rolling-Energie (Mittel der Quadrate) | Aktivität auf dieser Frequenz |
| `wt_detail_{lvl}_mean_{w}` | Rolling-Mittelwert der Koeffizienten | Richtungsbias auf dieser Frequenz |
| `wt_approx_energy_{w}` | Energie der Approximation (Trend) | Trend-Stärke |

### Ratio Features

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `wt_detail_ratio_{lvl}` | Anteil der Detail-Energie an Gesamt-Energie | Welche Frequenz dominiert? |
| `wt_high_freq_ratio_{w}` | Detail_1 / Detail_N Energie-Verhältnis | Hoch = choppy/noisy, Niedrig = trendend |

**Parameter:**
- `wavelet` (default: "db4") — Wavelet-Familie (db4, haar, sym5, etc.)
- `levels` (default: 3) — Dekompositions-Stufen
- `windows` (default: [10, 20, 50]) — Rolling-Fenster für Energie-Berechnung

**Plugin:** `fwbg-core:wavelets` | **Prefix:** `wt_` | **27 Features (default: 3 Levels × 3 Windows)**

---

## 28. Autoencoder / PCA Features (`ae_`)

Latent Feature Extraction via PCA (Principal Component Analysis) — komprimiert alle vorhandenen Indicator-Features in niedrig-dimensionale Repräsentationen, die nicht-lineare Zusammenhänge zwischen Indikatoren erfassen.

**Funktionsweise:**
1. Sammelt alle numerischen Feature-Spalten aus dem DataFrame (exkl. OHLCV und `ae_`-Prefix)
2. NaN-Imputation via Spalten-Median (robust gegen Ausreißer)
3. Standardisierung (Zero Mean, Unit Variance)
4. PCA-Transformation in `n_components` latente Dimensionen
5. Reconstruction Error als Anomalie-Signal

### Features

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `ae_latent_0..N` | PCA-Komponenten | Hauptvariationsmodi über alle Indikatoren |
| `ae_reconstruction_error` | L2 Rekonstruktionsfehler pro Zeile | Hoch = ungewöhnlicher Marktzustand (Anomalie-Detektion) |
| `ae_explained_variance` | Kumulative erklärte Varianz | Wie viel Information die Latent-Features erfassen |

**Besonderheit:** `ae_reconstruction_error` ist ein kraftvolles Anomalie-Signal — wenn der aktuelle Marktzustand schlecht durch die Hauptkomponenten erklärt wird, ist er "ungewöhnlich" relativ zur jüngsten Geschichte.

**Parameter:**
- `n_components` (default: 8) — Anzahl latenter Dimensionen
- `exclude_prefixes` (default: ["ae_"]) — Feature-Prefixes die ausgeschlossen werden

**Plugin:** `fwbg-core:autoencoder_features` | **Prefix:** `ae_` | **10 Features (default: 8 Latent + Error + Variance)**

---

## 29. Topological Data Analysis Features (`tda_`)

Erkennt topologische Strukturen in der Preisdynamik mittels Persistent Homology. Konvertiert Preisreturns über Takens Time-Delay Embedding in eine Punktwolke und berechnet daraus topologische Invarianten.

**Algorithmus:**
1. Log-Returns berechnen
2. Takens Embedding: 1D-Zeitreihe → Punktwolke im höherdimensionalen Raum
3. Persistent Homology (ripser): H0 (Connected Components), H1 (Loops/Zyklen)
4. Feature-Extraktion aus Persistence Diagrams

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `tda_h0_count_{w}` | Anzahl H0 Features (Connected Components) | Fragmentierung der Preisstruktur |
| `tda_h1_count_{w}` | Anzahl H1 Features (Loops/Zyklen) | Zyklische Muster, Mean-Reversion |
| `tda_h0_max_pers_{w}` | Max Persistence in H0 | Stärke der dominanten Struktur |
| `tda_h1_max_pers_{w}` | Max Persistence in H1 | Stärke des dominanten Zyklus |
| `tda_h0_mean_pers_{w}` | Mittlere Persistence H0 | Durchschnittliche Strukturstärke |
| `tda_h1_mean_pers_{w}` | Mittlere Persistence H1 | Durchschnittliche Zyklusstärke |
| `tda_persistence_entropy_{w}` | Shannon-Entropie des Persistence Diagrams | Komplexität der Topologie |
| `tda_wasserstein_amp_{w}` | L2-Norm des Persistence Diagrams | Gesamtamplitude topologischer Features |
| `tda_h1_ratio_{w}` | H1 / H0 Count (safe_divide) | Relative Zyklizität |
| `tda_max_loop_persistence_{w}` | Max H1 / Max H0 Persistence | Relative Loop-Stärke |

**Intuition:** Trending-Märkte haben wenige H1-Features (Loops), Mean-Reverting-Märkte viele. Die Persistence Entropy ist niedrig bei einfacher Topologie (Trend) und hoch bei komplexer Struktur (Chop).

**Parameter:**
- `windows` (default: [50, 100]) — Rolling-Window-Größen
- `embedding_dim` (default: 3) — Dimension des Takens Embedding
- `time_delay` (default: 1) — Zeitverzögerung im Embedding
- `maxdim` (default: 1) — Maximale Homologie-Dimension (0=H0, 1=H0+H1)

**Plugin:** `fwbg-core:topological_features` | **Prefix:** `tda_` | **20 Features (10 pro Window × 2 Windows)**

---

## 30. Adversarial Validation Features (`adv_`)

Erkennt Distribution Shift zwischen älteren und neueren Marktdaten innerhalb eines Sliding Window. Ein Classifier (LogisticRegression) versucht, "alt" von "neu" zu unterscheiden — hohe AUC = Regime-Wechsel.

**Algorithmus:**
1. Numeric Feature-Spalten selektieren (exkl. OHLCV + adv_)
2. Pro Position: Window in "alt" und "neu" Hälfte teilen
3. LogisticRegression fitten, AUC berechnen
4. AUC ~0.5 = stabile Distribution, AUC ~1.0 = starker Shift

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `adv_auc_{w}` | AUC des Alt-vs-Neu Classifiers | 0.5 = identisch, 1.0 = komplett verschieden |
| `adv_drift_score_{w}` | Normalisierter Drift: 2×(AUC−0.5) | 0 = kein Drift, 1 = maximaler Drift |
| `adv_stability_{w}` | 1 − drift_score | Regime-Stabilität |
| `adv_max_feature_importance_{w}` | Max. abs. Koeffizient der LogReg | Welches Feature driftet am stärksten |
| `adv_drift_acceleration_{w}` | Änderung des drift_score | Beschleunigung/Verlangsamung des Shifts |

**Besonderheit:** Meta-Feature über die Stationarität aller anderen Features. Besonders wertvoll in Kombination mit Walk-Forward — signalisiert dem Modell, wenn es sich auf eine veränderte Marktumgebung einstellen muss.

**Parameter:**
- `windows` (default: [100, 200]) — Window-Größen für Alt/Neu-Vergleich
- `step` (default: 10) — Berechnungsschrittweite (Forward-Fill dazwischen)
- `max_features` (default: 30) — Max. Feature-Anzahl pro Berechnung (Subsampling)
- `exclude_prefixes` (default: ["adv_"]) — Auszuschließende Feature-Prefixes

**Plugin:** `fwbg-core:adversarial_validation` | **Prefix:** `adv_` | **10 Features (5 pro Window × 2 Windows)**

---

## 31. Opening Range Features (`orb_`)

**Plugin:** `fwbg-core:opening_range` | **~23 Features**

Opening Range Breakout (ORB) Features erfassen die Intraday-Eröffnungsdynamik — sowohl als Rolling-ORB (stündliche Berechnung) als auch als Session-spezifische ORB für konfigurierbare Handelsstunden (Asia Open, London Open, NY Open). Zusätzlich werden Rolling-Statistiken (Breakout-Rate, Continuation-Rate) berechnet, die zeigen ob ORB-Breakouts aktuell *funktionieren*.

**Trading-Relevanz:** Die Opening Range ist eines der ältesten und robustesten Intraday-Konzepte. Die erste Bar einer Session etabliert Support/Resistance — ein Breakout darüber signalisiert Richtung für den Rest der Session. Session-spezifische ORBs sind besonders wertvoll: der NY Open (14:00 UTC) bringt institutionelles Volumen, London Open (08:00 UTC) definiert die europäische Richtung. Die Range-Breite relativ zum ATR unterscheidet Tight Ranges (Breakout-Setup) von Wide Ranges (Mean-Reversion-Setup).

### Rolling ORB

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `orb_range_atr` | ORB-Range normalisiert durch ATR | <1 = enge Range (Breakout), >1 = weite Range |
| `orb_position` | Preis-Position innerhalb der ORB | 0 = Low, 1 = High |
| `orb_breakout_up` | Breakout über ORB-High | 1 = bullisher Breakout |
| `orb_breakout_down` | Breakout unter ORB-Low | 1 = bearisher Breakout |
| `orb_time_since_open` | Bars seit ORB-Etablierung | Alter des Setups |

### Session-spezifische ORB (`orb_sXX_*`)

Pro konfigurierter Session (z.B. s00, s08, s13, s14) werden 5 Features berechnet:

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `orb_sXX_range_atr` | Session-ORB Range / ATR | Range-Qualität für diese Session |
| `orb_sXX_position` | Position in Session-ORB | Bullish/Bearish Bias |
| `orb_sXX_breakout_up` | Session-Breakout nach oben | Session-Richtungssignal |
| `orb_sXX_breakout_down` | Session-Breakout nach unten | Session-Richtungssignal |
| `orb_sXX_active` | Session-ORB aktiv (selbe Stunde) | Nur während Session-Open |

### Rolling Statistiken

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `orb_breakout_rate` | Anteil der Stunden mit Breakout | Markt-Volatilität |
| `orb_continuation_rate` | Anteil der Breakouts mit Follow-Through | ORB-Qualität aktuell |
| `orb_avg_range_atr` | Durchschnittliche ORB-Range / ATR | Typische Range-Breite |

**Parameter:**
- `range_bars` (default: 1) — Bars für die Opening Range (1 = erste Bar der Stunde)
- `atr_period` (default: 14) — ATR-Normalisierungsperiode
- `sessions` (default: [0, 8, 13, 14]) — UTC-Stunden für Session-ORB
- `stat_window` (default: 20) — Rolling-Fenster für Statistiken
- `enable_rolling/session/stats` (default: true) — Feature-Gruppen aktivieren/deaktivieren

**Hinweis:** Überspringe Berechnung für Daily-Daten (Intraday-Feature).

---

## 32. Ichimoku Cloud Features (`ichi_`)

**Plugin:** `fwbg-premium:ichimoku` | **~20 Features**

Ichimoku Kinko Hyo ("Gleichgewicht auf einen Blick") ist ein vollständiges Trading-System aus fünf Komponenten, das Trend, Momentum, Support/Resistance und Signale in einem einzigen Framework vereint. Im Gegensatz zu typischen Indikatoren, die jeweils einen Aspekt messen, liefert Ichimoku ein *ganzheitliches* Marktbild.

**Trading-Relevanz:** Ichimoku ist besonders wertvoll als Regime-Filter: Preis über der Cloud = starker Bullish Bias (nur Longs), unter der Cloud = Bearish Bias, in der Cloud = Range/Unsicherheit. Die Cloud-Dicke zeigt die Stärke des Support/Resistance — dicke Clouds absorbieren Breakout-Versuche, dünne Clouds brechen leicht. TK-Crosses (Tenkan über Kijun) sind klassische Entry-Signale; Kumo-Twists (Cloud-Farbwechsel) signalisieren Regime-Changes.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `ichi_tenkan` | Conversion Line (9-Perioden Midpoint) | Schnelle Gleichgewichtslinie |
| `ichi_kijun` | Base Line (26-Perioden Midpoint) | Langsame Gleichgewichtslinie |
| `ichi_senkou_a` | Leading Span A (Tenkan+Kijun)/2 | Obere/untere Cloud-Grenze |
| `ichi_senkou_b` | Leading Span B (52-Perioden Midpoint) | Gegenüberliegende Cloud-Grenze |
| `ichi_cloud_pos` | Position relativ zur Cloud | 0-1, über/unter Cloud |
| `ichi_cloud_thick` | Cloud-Dicke (normalisiert) | Stärke Support/Resistance |
| `ichi_cloud_color` | Cloud-Farbe | 1 = bullish (A>B), -1 = bearish |
| `ichi_tk_cross` | Tenkan-Kijun Differenz | Cross-Signal |
| `ichi_price_kijun` | Preis-Kijun Abstand | Trend-Stärke |
| `ichi_kumo_twist` | Cloud-Farbwechsel erkannt | Regime-Change Signal |
| `ichi_distance_cloud` | Abstand zur Cloud | Überdehnung vom Equilibrium |
| `ichi_composite_bull` | Bullishes Kompositum | Preis > Cloud, TK-Cross bullish, Cloud bullish |
| `ichi_composite_bear` | Bearishes Kompositum | Preis < Cloud, TK-Cross bearish, Cloud bearish |
| `ichi_composite_neutral` | Neutrales Kompositum | Gemischte Signale |

**Parameter:** `tenkan_period` (default: 9), `kijun_period` (default: 26), `senkou_b_period` (default: 52)

---

## 33. Market Regime Features (`regime_risk_`)

**Plugin:** `fwbg-premium:market_regime` | **~7 Features**

Market Regime Features synthetisieren einen Composite Risk-Score aus vier makroökonomischen Dimensionen: Volatilität (VIX), Kreditstress (HYG/LQD Spread), Equity-Momentum (SPX) und Treasury-Flucht (TLT). Das Ergebnis ist eine binäre Risk-On/Risk-Off Klassifikation.

**Trading-Relevanz:** Das Marktregime bestimmt welche Strategien funktionieren. In Risk-On Phasen (niedriger VIX, enge Credit Spreads, steigende Aktien) funktioniert Trend-Following auf Risk-Assets. In Risk-Off (hoher VIX, weite Spreads, fallende Aktien, steigende Treasuries) dominieren Safe-Haven-Flows und Korrelationen brechen zusammen. Ein Composite-Score ist robuster als einzelne Indikatoren, da er mehrere unabhängige Signalquellen kombiniert.

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `regime_risk_composite` | Composite Risk-Score | Z-Score basiert, positiv = Risk-On |
| `regime_risk_on` | Risk-On Flag | 1 wenn Composite > Threshold |
| `regime_risk_off` | Risk-Off Flag | 1 wenn Composite < -Threshold |
| `regime_risk_vix_z` | VIX Z-Score (invertiert) | Niedriger VIX = Risk-On |
| `regime_risk_credit_z` | Credit Spread Z-Score | Enge Spreads = Risk-On |
| `regime_risk_equity_z` | SPX Momentum Z-Score | Steigende Aktien = Risk-On |
| `regime_risk_treasury_z` | TLT Flight Z-Score | Fallende Treasuries = Risk-On |

**Parameter:** `window` (default: 50 Tage, konvertiert zu Bars)

**Voraussetzung:** Erfordert Makro-Daten (macro_vix, macro_hyg, macro_lqd, macro_spx, macro_tlt). Gibt leeren DataFrame zurück wenn Daten fehlen.

---

## 34. Regime Cluster Features (`regime_cluster_`)

**Plugin:** `fwbg-premium:regime_cluster` | **~4 Features**

Regime Cluster Features produzieren eine quantil-basierte 3-Klassen-Klassifikation (0/1/2) des Marktregimes durch Kombination orthogonaler Strukturindikatoren: Hurst-Exponent (Persistenz), Entropie (Ordnung), Variance Ratio (Effizienz), Volatilitäts-Rang und Hurst-Divergenz. Optional wird der Makro-Risk-Composite einbezogen.

**Trading-Relevanz:** Regime-Cluster vereinfachen die Strategieauswahl: Cluster 0 = choppy/mean-reverting (Mean-Reversion Strategien), Cluster 1 = neutral (Vorsicht), Cluster 2 = trending/persistent (Trend-Following). Im Gegensatz zu kontinuierlichen Regime-Scores sind diskrete Cluster stabil und interpretierbar — ideal für regelbasierte Filter in der Grid-Optimierung (z.B. "nur traden in Regime 2").

| Feature | Beschreibung | Interpretation |
|---------|--------------|----------------|
| `regime_cluster_label` | Regime-Cluster (0, 1, 2) | 0 = choppy, 1 = neutral, 2 = trending |
| `regime_cluster_score` | Composite Z-Score | Kontinuierlicher Regime-Score |
| `regime_cluster_score_chg` | Score-Änderung | Regime-Transition Erkennung |
| `regime_cluster_stable` | Stabilität des Clusters | 1 wenn Cluster über Fenster konstant |

**Kern-Inputs (immer verwendet):**
- `regime_hurst_200` — Persistenz (positiv gewichtet)
- `regime_entropy_100` — Entropie (invertiert, niedrig = günstig)
- `regime_vr_200_5` — Variance Ratio (um 0 zentriert)
- `vol_atr_pct_14_rank` — Volatilitätslevel
- `regime_hurst_divergence` — Regime-Shift Signal

**Optionaler Input:** `regime_risk_composite` (Makro-Risk, gewichtet +1.0)

**Parameter:** `zscore_window` (default: 200), `quantile_window` (default: 500), `n_regimes` (default: 3)

**Voraussetzung:** Erfordert upstream Regime- und Volatilitäts-Indikatoren.

---

## Feature-Gruppen / Indicator Plugins

Das Plugin-System ermöglicht modulare Konfiguration von Indikatoren. Jeder Indikator ist ein separates Plugin mit eigenen konfigurierbaren Parametern.

### Verfügbare Indicator Plugins

#### Core Plugins (`fwbg-core`)

| Plugin Name | Gruppe | Features | Beschreibung |
|-------------|--------|----------|--------------|
| `trend` | trend | ~34 | ADX, EMA, SMA, MACD, CCI, Aroon, ER, Supertrend |
| `momentum` | momentum | ~16 | RSI, Stochastic, Williams %R, ROC, UO |
| `volatility` | volatility | ~14 | Bollinger, Keltner, Donchian, ATR, GK/PK/YZ Vol |
| `price_action` | price_action | ~17 | Range, Body, Gaps, Streaks, Inside Bars, Volume |
| `time_season` | time | ~14 | Zeit, Sessions, Saisonalität (sin/cos) |
| `dynamics` | dynamics | ~27 | Änderungen, Lags, Beschleunigung |
| `cross_features` | cross | ~3 | Confluence-Signale |
| `distribution` | distribution | ~10 | Skewness, Kurtosis |
| `structure` | structure | ~20 | FFT, Path Efficiency, Convexity, VWAP |
| `regime` | regime | ~6 | Hurst-Exponent, Persistenz |
| `opening_range` | orb | ~23 | Opening Range Breakout, Session-ORB |
| `fair_value_gap` | fvg | 8 | Bull/Bear FVG, Distance, Size |
| `cusum_events` | cusum | 6 | CUSUM Structural Breaks (AFML Ch. 2) |
| `calendar_events` | calendar | 9 | Turn-of-Month, OpEx, FOMC, NFP |
| `fractal_dimension` | fractal | 12 | Higuchi FD, Komplexität, Regime |
| `wavelets` | wavelets | 27 | DWT Energie, Frequenz-Ratios |
| `autoencoder_features` | autoencoder | 10 | PCA Latent Features, Reconstruction Error |
| `topological_features` | topology | 20 | Persistent Homology, H0/H1 |
| `adversarial_validation` | meta | 10 | Distribution Shift, Drift Score |

#### Premium Plugins (`fwbg-premium`)

| Plugin Name | Gruppe | Features | Beschreibung |
|-------------|--------|----------|--------------|
| `ichimoku` | trend | ~20 | Ichimoku Cloud, TK-Cross, Kumo-Twist |
| `multi_timeframe` | mtf | ~25 | H4/D1/W1 Aggregation, Consensus |
| `risk` | risk | ~25 | Drawdown, CVaR, Vol-of-Vol, Crash-Prob |
| `microstructure` | microstructure | ~17 | Wick Imbalance, Pressure, A/D, CMF |
| `macro_surprise` | macro_surprise | ~21 | Gap-Analyse, Surprises, Vol-Breaks |
| `events` | events | ~12 | Bars-Since-Event, Konsolidierung |
| `support_resistance` | s/r | 31 | S/R Zones, Trend, Flip-Zones, Breakouts |
| `market_regime` | regime | ~7 | Risk-On/Off Composite (VIX, Credit, SPX) |
| `regime_cluster` | regime | ~4 | Quantil-basierte 3-State Regime-Cluster |

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
