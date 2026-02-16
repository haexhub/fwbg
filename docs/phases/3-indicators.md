# Phase 3: Indicators

## Zweck

Die Indicator-Phase berechnet technische Features aus OHLCV-Daten. Jeder Indikator erzeugt neue Spalten im DataFrame, die anschließend vom ML-Modell als Input verwendet werden.

---

## BaseIndicator

Basisklasse: `src/fwbg/plugins/indicator.py`

```python
class BaseIndicator(BasePlugin, ABC):
    phase = PluginPhase.INDICATORS
    stateful = False
    cacheable = True
    group: str = "custom"                    # Feature-Gruppe
    benefits_from_stationary: bool = False    # Nach Preprocessing berechnen?

    @abstractmethod
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Berechnet Indicator-Spalten. Muss shift_features() verwenden!"""

    def get_feature_columns(self) -> List[str]:
        """Gibt Feature-Spaltennamen zurück."""
```

- Registrierung: `@register_indicator("name")`
- `group`: Kategorisierung (z.B. "trend", "momentum", "custom")
- `benefits_from_stationary`: Siehe [Architektur](../architecture.md#benefits_from_stationary-bool-nur-indikatoren-default-false)

---

## Pflicht-Helfer

Jede `compute()`-Methode **muss** zwei Helfer-Funktionen verwenden. Das ist keine Empfehlung, sondern Pflicht.

### shift_features(features, index)

```python
from fwbg.plugins.indicator import shift_features

features = {"my_rsi": rsi_series, "my_macd": macd_series}
features_df = shift_features(features, df.index)
```

**Was es tut:** Erstellt einen DataFrame aus dem Feature-Dict und shiftet **alle Spalten um 1 Bar** (`shift(1)`). Die erste Zeile wird dadurch `NaN`.

**Warum es Pflicht ist:** Ohne den Shift sieht das ML-Modell bei Bar `i` die Indikator-Werte von Bar `i` — also Informationen, die zum Zeitpunkt der Handelsentscheidung **noch nicht verfügbar wären** (die aktuelle Bar ist noch nicht abgeschlossen). Das ist **Lookahead-Bias** und macht jedes Backtesting-Ergebnis wertlos.

Mit dem 1-Bar-Shift sieht das Modell nur Features von Bar `i-1` (die letzte abgeschlossene Bar).

**Beispiel:**

```
Bar:     | 0  | 1  | 2  | 3  | 4  |
RSI(14): | 45 | 52 | 61 | 48 | 55 |  ← Originalwerte
Shifted: | NaN| 45 | 52 | 61 | 48 |  ← Was das Modell sieht
```

Bei Bar 3 sieht das Modell RSI=61 (von Bar 2), nicht RSI=48 (von Bar 3 selbst).

### safe_divide(numerator, denominator)

```python
from fwbg.plugins.indicator import safe_divide

ratio = safe_divide(df["C"] - ema, df["C"])
```

**Was es tut:** Division mit `NaN` statt Division-by-Zero. Verwendet einen Epsilon-Threshold von `1e-10` — Werte kleiner als Epsilon werden als Null behandelt.

**Warum es Pflicht ist:** Viele Indikatoren berechnen Ratios (RSI, Efficiency Ratio, Bollinger %B, etc.). Ohne safe_divide können bei kleinen Nennern `inf`-Werte entstehen, die das ML-Modell korrumpieren. `NaN` wird dagegen vom Modell sauber als fehlender Wert behandelt.

**Funktioniert mit:** `pd.Series` und `np.ndarray`.

---

## Vollständiges Beispiel

```python
import pandas as pd
import numpy as np
from fwbg.plugins.indicator import BaseIndicator, shift_features, safe_divide
from fwbg.pipeline.base import PluginPhase
from fwbg.core.registry import register_indicator


@register_indicator("my_momentum")
class MyMomentumIndicator(BaseIndicator):
    name = "my_momentum"
    version = "1.0.0"
    group = "momentum"
    benefits_from_stationary = False  # Auf Raw-Daten berechnen

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        lookback = params.get("lookback", 14)

        features = {}
        returns = df["C"].pct_change()

        # Feature 1: Durchschnittliche Returns
        features["my_avg_return"] = returns.rolling(lookback).mean()

        # Feature 2: Return-Volatilität
        features["my_return_vol"] = returns.rolling(lookback).std()

        # Feature 3: Sharpe-artiges Ratio (MUSS safe_divide verwenden!)
        features["my_sharpe"] = safe_divide(
            features["my_avg_return"],
            features["my_return_vol"]
        )

        # PFLICHT: shift_features() am Ende!
        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> list:
        return ["my_avg_return", "my_return_vol", "my_sharpe"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"lookback": 14}
```

---

## benefits_from_stationary

| Wert | Berechnung | Caching | Beispiele |
|------|------------|---------|-----------|
| `False` (Default) | Einmalig auf Raw-OHLC | Gecacht über alle Folds | `momentum`, `volatility`, `price_action` |
| `True` | Pro Fold auf preprocessed Data | Nicht gecacht | `trend` (ADX auf differenzierten Daten) |

Die Entscheidung liegt beim Plugin-Entwickler. Faustregel:
- **Trendfolge-Indikatoren** (ADX, Moving Averages): Profitieren von stationären Daten → `True`
- **Ratio-basierte Indikatoren** (RSI, Stochastic): Bereits normalisiert → `False`
- **Volatilitäts-Indikatoren** (ATR, Bollinger): Skalenunabhängig → `False`

---

## Verfügbare Indikatoren

### Core-Paket (fwbg-core)

| Plugin | Beschreibung | Feature-Prefix |
|--------|--------------|----------------|
| `trend` | ADX, EMA, SMA, MACD, CCI, Aroon, Supertrend, Efficiency Ratio | `trend_` |
| `momentum` | RSI, Stochastic, Williams %R, ROC | `mom_` |
| `volatility` | Bollinger Bands, ATR, Volatilitätsschätzer, Vol Compression, RV vs IV | `vol_` |
| `price_action` | Range Position, Higher Highs/Lower Lows, Body Ratio, Gaps | `pa_` |
| `time_season` | Stunde, Wochentag, Monat, Quartal, Saisonalität | `time_`, `season_` |

### Premium-Paket (fwbg-premium)

| Plugin | Beschreibung | Feature-Prefix |
|--------|--------------|----------------|
| `regime` | Hurst Exponent, Entropy, Variance Ratio | `regime_` |
| `structure` | FFT, Path Statistics, Convexity, Event Flow, VWAP | `struct_` |
| `risk` | Drawdown, CVaR, Volatility of Volatility, Correlations | `risk_` |
| `distribution` | Skewness, Kurtosis, Z-Score | `dist_` |
| `dynamics` | Indikator-Änderungen, Lags, Beschleunigung | `dyn_`, `lag_`, `accel_` |
| `multi_timeframe` | H4/D1/W1/Y1 Multi-Timeframe Features, Trend Alignment | `mtf_` |
| `cross_features` | Kombinierte Signale, COT × Vol Interaction | `cross_` |
| `ichimoku` | Ichimoku Cloud Komponenten | `ichi_` |
| `macro_surprise` | Makro-Überraschungen, Gap-Analyse | `macro_surprise_` |
| `microstructure` | Bar-Microstructure, Tick-Proxies | `micro_` |
| `market_regime` | Risk-On/Off Composite aus VIX, Credit, Equity, Treasury | `regime_risk_`, `regime_vix_` |
| `regime_cluster` | Composite Regime Score → K-Means Clustering | `regime_cluster_` |

**Vollständige Feature-Dokumentation:** [docs/FEATURES.md](../FEATURES.md)

---

## Strategy-JSON Konfiguration

```json
"pipeline": {
  "indicators": [
    {"name": "trend", "params": {"adx_periods": [7, 14, 21], "ema_periods": [8, 21, 50]}},
    {"name": "momentum", "params": {"rsi_periods": [7, 14]}},
    {"name": "volatility", "params": {"atr_periods": [7, 14, 21]}},
    {"name": "regime", "params": {}},
    {"name": "market_regime", "params": {"window": 50}}
  ]
}
```

Parameter überschreiben die Defaults des Plugins. Nicht angegebene Parameter verwenden den Default aus `get_default_params()`.

---

## Eigene Indikatoren erstellen

Siehe [Plugin Development Guide](../plugin-development.md) für die vollständige Anleitung.
