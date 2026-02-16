# Phase 3: Indicators

## Purpose

The indicator phase computes technical features from OHLCV data. Each indicator produces new columns in the DataFrame, which are subsequently used as input for the ML model.

---

## BaseIndicator

Base class: `src/fwbg/plugins/indicator.py`

```python
class BaseIndicator(BasePlugin, ABC):
    phase = PluginPhase.INDICATORS
    stateful = False
    cacheable = True
    group: str = "custom"                    # Feature group
    benefits_from_stationary: bool = False    # Compute after preprocessing?

    @abstractmethod
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Computes indicator columns. Must use shift_features()!"""

    def get_feature_columns(self) -> List[str]:
        """Returns feature column names."""
```

- Registration: `@register_indicator("name")`
- `group`: Categorization (e.g., "trend", "momentum", "custom")
- `benefits_from_stationary`: See [Architecture](../architecture.md#benefits_from_stationary-bool-indicators-only-default-false)

---

## Required Helpers

Every `compute()` method **must** use two helper functions. This is not a recommendation — it is mandatory.

### shift_features(features, index)

```python
from fwbg.plugins.indicator import shift_features

features = {"my_rsi": rsi_series, "my_macd": macd_series}
features_df = shift_features(features, df.index)
```

**What it does:** Creates a DataFrame from the feature dict and shifts **all columns by 1 bar** (`shift(1)`). The first row becomes `NaN`.

**Why it is mandatory:** Without the shift, the ML model sees indicator values for bar `i` at bar `i` — meaning information that **would not yet be available** at the time of the trading decision (the current bar has not yet closed). This is **lookahead bias** and renders any backtesting result worthless.

With the 1-bar shift, the model only sees features from bar `i-1` (the last completed bar).

**Example:**

```
Bar:     | 0  | 1  | 2  | 3  | 4  |
RSI(14): | 45 | 52 | 61 | 48 | 55 |  ← Original values
Shifted: | NaN| 45 | 52 | 61 | 48 |  ← What the model sees
```

At bar 3, the model sees RSI=61 (from bar 2), not RSI=48 (from bar 3 itself).

### safe_divide(numerator, denominator)

```python
from fwbg.plugins.indicator import safe_divide

ratio = safe_divide(df["C"] - ema, df["C"])
```

**What it does:** Division with `NaN` instead of division-by-zero. Uses an epsilon threshold of `1e-10` — values smaller than epsilon are treated as zero.

**Why it is mandatory:** Many indicators compute ratios (RSI, Efficiency Ratio, Bollinger %B, etc.). Without safe_divide, small denominators can produce `inf` values that corrupt the ML model. `NaN`, on the other hand, is cleanly handled by the model as a missing value.

**Works with:** `pd.Series` and `np.ndarray`.

---

## Complete Example

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
    benefits_from_stationary = False  # Compute on raw data

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        lookback = params.get("lookback", 14)

        features = {}
        returns = df["C"].pct_change()

        # Feature 1: Average returns
        features["my_avg_return"] = returns.rolling(lookback).mean()

        # Feature 2: Return volatility
        features["my_return_vol"] = returns.rolling(lookback).std()

        # Feature 3: Sharpe-like ratio (MUST use safe_divide!)
        features["my_sharpe"] = safe_divide(
            features["my_avg_return"],
            features["my_return_vol"]
        )

        # MANDATORY: shift_features() at the end!
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

| Value | Computation | Caching | Examples |
|-------|-------------|---------|----------|
| `False` (default) | Once on raw OHLC | Cached across all folds | `momentum`, `volatility`, `price_action` |
| `True` | Per fold on preprocessed data | Not cached | `trend` (ADX on differentiated data) |

The decision is up to the plugin developer. Rule of thumb:
- **Trend-following indicators** (ADX, Moving Averages): Benefit from stationary data → `True`
- **Ratio-based indicators** (RSI, Stochastic): Already normalized → `False`
- **Volatility indicators** (ATR, Bollinger): Scale-independent → `False`

---

## Available Indicators

### Core Package (fwbg-core)

| Plugin | Description | Feature Prefix |
|--------|-------------|----------------|
| `trend` | ADX, EMA, SMA, MACD, CCI, Aroon, Supertrend, Efficiency Ratio | `trend_` |
| `momentum` | RSI, Stochastic, Williams %R, ROC | `mom_` |
| `volatility` | Bollinger Bands, ATR, Volatility Estimators, Vol Compression, RV vs IV | `vol_` |
| `price_action` | Range Position, Higher Highs/Lower Lows, Body Ratio, Gaps | `pa_` |
| `time_season` | Hour, Day of Week, Month, Quarter, Seasonality | `time_`, `season_` |

### Premium Package (fwbg-premium)

| Plugin | Description | Feature Prefix |
|--------|-------------|----------------|
| `regime` | Hurst Exponent, Entropy, Variance Ratio | `regime_` |
| `structure` | FFT, Path Statistics, Convexity, Event Flow, VWAP | `struct_` |
| `risk` | Drawdown, CVaR, Volatility of Volatility, Correlations | `risk_` |
| `distribution` | Skewness, Kurtosis, Z-Score | `dist_` |
| `dynamics` | Indicator Changes, Lags, Acceleration | `dyn_`, `lag_`, `accel_` |
| `multi_timeframe` | H4/D1/W1/Y1 Multi-Timeframe Features, Trend Alignment | `mtf_` |
| `cross_features` | Combined Signals, COT × Vol Interaction | `cross_` |
| `ichimoku` | Ichimoku Cloud Components | `ichi_` |
| `macro_surprise` | Macro Surprises, Gap Analysis | `macro_surprise_` |
| `microstructure` | Bar Microstructure, Tick Proxies | `micro_` |
| `market_regime` | Risk-On/Off Composite from VIX, Credit, Equity, Treasury | `regime_risk_`, `regime_vix_` |
| `regime_cluster` | Composite Regime Score → K-Means Clustering | `regime_cluster_` |

**Complete feature documentation:** [docs/FEATURES.md](../FEATURES.md)

---

## Strategy JSON Configuration

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

Parameters override the plugin defaults. Unspecified parameters use the default from `get_default_params()`.

---

## Creating Custom Indicators

See [Plugin Development Guide](../plugin-development.md) for the complete guide.
