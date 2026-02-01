# FWBG Plugin Development Guide

Dieses Dokument erklärt, wie du eigene Plugins für FWBG entwickelst.

## Plugin-Typen

FWBG unterstützt 4 Plugin-Typen:

| Typ | Base Class | Entry Point Group |
|-----|------------|-------------------|
| Indicators | `BaseIndicator` | `fwbg.indicators` |
| Exit Strategies | `BaseExitStrategy` | `fwbg.exit_strategies` |
| Feature Selectors | `BaseFeatureSelector` | `fwbg.feature_selectors` |
| Preprocessors | `BasePreprocessor` | `fwbg.preprocessors` |

## Built-in Indicator Plugins

FWBG enthält 13 Indicator-Plugins:

| Plugin | Gruppe | Features |
|--------|--------|----------|
| `trend` | trend | ADX, EMA, SMA, MACD, CCI, Aroon, Efficiency Ratio |
| `momentum` | momentum | RSI, Stochastic, Williams %R, Ultimate Oscillator, ROC |
| `volatility` | volatility | ATR, Bollinger Bands, Keltner Channel, Donchian |
| `regime` | regime | Hurst Exponent, Regime Shift Detection |
| `structure` | structure | FFT, Path Efficiency, Convexity, Events, VWAP |
| `risk` | risk | Drawdown, CVaR, Vol-of-Vol, Crash Probability |
| `price_action` | price_action | Candle Features, HH/LL, Gaps, Volume |
| `time_season` | time | Hour, Day, Month, Quarter, Sessions |
| `distribution` | distribution | Skewness, Kurtosis, Tail Risk |
| `dynamics` | dynamics | Indicator Changes, Lags, Acceleration |
| `multi_timeframe` | multi_timeframe | H4/D1 Features, Trend Alignment |
| `cross_features` | cross | Confluences, Divergences, Composite Scores |
| `ichimoku` | ichimoku | Cloud, TK Cross, Kijun, Chikou |

## Schnellstart: Eigenes Plugin erstellen

### 1. Package-Struktur

```
fwbg-my-plugin/
├── pyproject.toml
├── src/
│   └── fwbg_my_plugin/
│       ├── __init__.py
│       └── my_indicator.py
└── tests/
    └── test_my_indicator.py
```

### 2. pyproject.toml

```toml
[project]
name = "fwbg-my-plugin"
version = "1.0.0"
dependencies = ["fwbg>=2.0.0"]

[project.entry-points."fwbg.indicators"]
my_indicator = "fwbg_my_plugin.my_indicator:MyIndicator"
```

### 3. Plugin implementieren

```python
# fwbg_my_plugin/my_indicator.py
from typing import List
import pandas as pd
from fwbg.plugins import BaseIndicator


class MyIndicator(BaseIndicator):
    """Mein Custom Indicator."""

    group = "custom"

    def compute(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        # Berechne Features...
        df["my_feature"] = df["C"].rolling(period).mean()
        return df

    def get_feature_columns(self) -> List[str]:
        return ["my_feature"]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"period": 14}
```

### 4. Installieren & Nutzen

```bash
pip install -e ./fwbg-my-plugin
```

Das Plugin wird automatisch erkannt!

---

## BaseIndicator

Indicators berechnen technische Features basierend auf OHLCV-Daten.

```python
from fwbg.plugins import BaseIndicator

class MyIndicator(BaseIndicator):
    """Docstring wird für Hilfe genutzt."""

    # Feature-Gruppe (trend, momentum, volatility, custom, etc.)
    group: str = "custom"

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        Berechnet Features und fügt sie zum DataFrame hinzu.

        Args:
            df: DataFrame mit O, H, L, C Spalten
            **params: Indicator-spezifische Parameter

        Returns:
            DataFrame mit neuen Feature-Spalten
        """
        ...

    def get_feature_columns(self) -> List[str]:
        """Liste der Feature-Spalten die dieser Indicator erzeugt."""
        ...

    @classmethod
    def get_default_params(cls) -> dict:
        """Default-Parameter."""
        ...
```

### Beispiel: RSI Indicator

```python
from fwbg.plugins import BaseIndicator
import ta

class RSIIndicator(BaseIndicator):
    group = "momentum"

    def compute(self, df, period: int = 14) -> pd.DataFrame:
        df["rsi_value"] = ta.momentum.rsi(df["C"], window=period)
        df["rsi_overbought"] = (df["rsi_value"] > 70).astype(float)
        df["rsi_oversold"] = (df["rsi_value"] < 30).astype(float)
        return df

    def get_feature_columns(self):
        return ["rsi_value", "rsi_overbought", "rsi_oversold"]
```

---

## BaseExitStrategy

Exit Strategies definieren wie TP/SL berechnet werden.

```python
from fwbg.plugins import BaseExitStrategy

class MyExitStrategy(BaseExitStrategy):

    def compute_targets(
        self, df: pd.DataFrame, ctx: "SimulationContext", **params
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Win/Loss Targets.

        Returns:
            (targets_long, targets_short) - 1.0 für Win, 0.0 für Loss
        """
        ...

    def iterate_grid(
        self, grid_config: dict, ctx: "SimulationContext"
    ) -> Iterator[dict]:
        """Iteriert über Parameter-Kombinationen."""
        ...

    def get_cache_key(self, params: dict) -> str:
        """Eindeutiger Key für Caching."""
        ...
```

### Beispiel: Trailing Stop

```python
class TrailingStopStrategy(BaseExitStrategy):

    def compute_targets(self, df, ctx, trail_pct: float = 0.01, **params):
        # Trailing Stop Simulation...
        return targets_long, targets_short

    def iterate_grid(self, grid_config, ctx):
        for trail in grid_config.get("trail_pct", [0.005, 0.01, 0.02]):
            yield {"trail_pct": trail}

    def get_cache_key(self, params):
        return f"trailing_{params['trail_pct']}"
```

---

## BaseFeatureSelector

Feature Selectors wählen die wichtigsten Features aus.

```python
from fwbg.plugins import BaseFeatureSelector

class MySelector(BaseFeatureSelector):

    def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        max_features: int = None,
        **params
    ) -> Tuple[List[str], dict]:
        """
        Wählt Features aus.

        Returns:
            (selected_features, metadata)
        """
        ...
```

### Beispiel: Mutual Information

```python
from sklearn.feature_selection import mutual_info_classif

class MutualInfoSelector(BaseFeatureSelector):

    def select_features(self, X, y, max_features=30, **params):
        mi_scores = mutual_info_classif(X, y)

        feature_scores = list(zip(X.columns, mi_scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        selected = [f for f, _ in feature_scores[:max_features]]

        metadata = {"scores": dict(feature_scores)}
        return selected, metadata
```

---

## BasePreprocessor

Preprocessors transformieren Daten vor der Feature-Berechnung.

```python
from fwbg.plugins import BasePreprocessor

class MyPreprocessor(BasePreprocessor):

    # Ausführungsreihenfolge (niedriger = früher)
    order: int = 100

    def transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Transformiert den DataFrame."""
        ...

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rücktransformation (optional)."""
        return df
```

---

## Strategy-Config Nutzung

Plugins werden in Strategy JSONs konfiguriert:

```json
{
  "name": "My Strategy",

  "indicators": ["trend", "momentum", "my_indicator"],

  "indicator_params": {
    "my_indicator": {"period": 21}
  },

  "exit_strategy": "trailing_stop",

  "exit_params": {
    "trail_pct": 0.01
  },

  "feature_selector": "boruta",

  "preprocessing": ["fractional_diff"],

  "grids": {
    "FOREX": {
      "tp": [1.0, 1.5, 2.0],
      "sl": [1.0, 1.5],
      "ct": [0.5, 0.55, 0.6]
    }
  }
}
```

---

## Best Practices

### Performance
- Nutze Numba für CPU-intensive Berechnungen
- Vermeide Python-Loops über große Arrays
- Cache teure Berechnungen

### Robustheit
- Behandle NaN/Inf-Werte
- Validiere Input-Parameter
- Gib aussagekräftige Fehlermeldungen

### Testing
- Schreibe Unit-Tests für jede Methode
- Teste Edge-Cases (leere DataFrames, NaN-Werte)
- Prüfe Performance mit realistischen Datenmengen

---

## Distribution

### Open Source (Public PyPI)

```bash
# Build
python -m build

# Upload zu PyPI
python -m twine upload dist/*
```

### Private (für Paid Plugins)

```bash
# Upload zu privatem PyPI
python -m twine upload \
  --repository-url https://pypi.yourcompany.com \
  dist/*
```

User installieren dann mit:

```bash
pip install fwbg-my-plugin \
  --extra-index-url https://pypi.yourcompany.com
```
