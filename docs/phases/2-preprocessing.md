# Phase 2: Preprocessing

## Zweck

Die Preprocessing-Phase transformiert OHLC-Daten vor der Feature-Berechnung. Der Hauptanwendungsfall ist die **Stationaritätstransformation** — Finanzzeitreihen sind typischerweise nicht-stationär, viele ML-Modelle funktionieren aber besser mit stationären Eingangsdaten.

---

## BasePreprocessor

Basisklasse: `src/fwbg/plugins/preprocessor.py`

```python
class BasePreprocessor(BasePlugin, ABC):
    phase = PluginPhase.PREPROCESSING
    name: str = "base"
    order: int = 100       # Ausführungsreihenfolge (niedriger = früher)
    fitted_: bool = False  # Ob fit() bereits aufgerufen wurde

    def fit(self, df: pd.DataFrame, **params) -> "BasePreprocessor":
        """Lernt Parameter von Train-Daten. NIEMALS auf Test/OOS-Daten!"""

    @abstractmethod
    def transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Transformiert DataFrame mit gelernten Parametern."""

    def fit_transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Kombiniert fit() und transform() für Train-Daten."""

    def inverse_transform(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Optional: Rücktransformation."""
```

- Registrierung: `@register_preprocessor("name")`
- `order` bestimmt die Reihenfolge bei mehreren Preprocessors (niedriger = früher)
- Folgt dem **sklearn fit/transform-Pattern**

---

## Lifecycle: fit/transform pro CV-Fold

Preprocessors sind **stateful** — sie lernen Parameter aus Trainingsdaten und wenden diese auf beliebige Daten an. Der Lifecycle pro Walk-Forward Fold:

```
┌─ Fold N ──────────────────────────────────────────────┐
│                                                        │
│  1. reset()                     Zustand zurücksetzen   │
│  2. fit(train_df)               Parameter lernen       │
│  3. transform(train_df)         Train transformieren   │
│  4. transform(test_df)          Test transformieren    │
│     (mit den in Schritt 2 gelernten Parametern!)       │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Warum fit() nur auf Train-Daten?

Das ist essentiell für **Lookahead-Bias-Prevention**:

- `fit()` lernt statistische Parameter (z.B. den optimalen Differenzierungsgrad d)
- Wenn `fit()` auf **allen** Daten aufgerufen wird, fließen zukünftige Informationen in die Transformation ein → das Modell "sieht" die Zukunft
- Deshalb: `fit()` **ausschließlich** auf dem Training-Split aufrufen
- `transform()` darf auf beliebige Daten angewendet werden — es nutzt nur die in `fit()` gelernten Parameter

---

## Interaktion mit Indikatoren

Preprocessing beeinflusst welche Indikatoren wann berechnet werden:

| `benefits_from_stationary` | Berechnung | Caching |
|----------------------------|------------|---------|
| `False` (Default) | **Einmalig auf Raw-OHLC** vor Preprocessing | Über alle Folds gecacht |
| `True` | **Pro Fold auf preprocessed OHLC** nach Preprocessing | Nicht gecacht |

Die Aufteilung erfolgt automatisch über `split_indicators_by_stationarity()` in `src/fwbg/pipeline/features.py`:

```
Raw Indicators (benefits_from_stationary=False):
  → Einmal auf Originaldaten berechnen
  → Gecacht, schnell

Stationary Indicators (benefits_from_stationary=True):
  → Pro Fold auf preprocessed Daten berechnen
  → Langsamer, aber korrekt für stationaritätsabhängige Features
```

---

## Verfügbare Plugins

### fractional_diff (fwbg-premium)

Fractional Differentiation nach López de Prado — macht Zeitreihen stationär **ohne den gesamten Memory zu verlieren** (im Gegensatz zu normaler Differenzierung).

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `auto_d` | `false` | Automatisch optimalen d-Wert finden (ADF-Test) |
| `default_d` | `0.4` | Fester d-Wert (0=original, 1=volle Differenzierung) |
| `columns` | `["O", "H", "L", "C"]` | Welche Spalten transformiert werden |

**Warnung:** `auto_d: true` kann Lookahead-Bias verursachen, wenn der ADF-Test auf dem gesamten Datensatz statt nur auf Train-Daten ausgeführt wird. Empfehlung: `auto_d: false` mit festem `default_d`.

---

## Strategy-JSON Konfiguration

```json
"pipeline": {
  "preprocessing": [
    {
      "name": "fractional_diff",
      "params": {
        "auto_d": false,
        "default_d": 0.4,
        "columns": ["O", "H", "L", "C"]
      }
    }
  ]
}
```

---

## Eigenes Preprocessing-Plugin erstellen

Siehe [Plugin Development Guide](../plugin-development.md) für die vollständige Anleitung.

### Kurzbeispiel

```python
from fwbg.plugins.preprocessor import BasePreprocessor
from fwbg.core.registry import register_preprocessor

@register_preprocessor("my_normalizer")
class MyNormalizer(BasePreprocessor):
    name = "my_normalizer"
    order = 50  # Vor fractional_diff (order=100)

    def fit(self, df, **params):
        self.mean_ = df["C"].mean()
        self.std_ = df["C"].std()
        return super().fit(df, **params)

    def transform(self, df, **params):
        super().transform(df, **params)  # Prüft ob fit() aufgerufen wurde
        result = df.copy()
        result["C"] = (result["C"] - self.mean_) / self.std_
        return result
```
