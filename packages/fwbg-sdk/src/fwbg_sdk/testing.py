"""Testing utilities for FWBG plugin developers."""
import numpy as np
import pandas as pd
from fwbg_sdk.contexts import AssetInfo


def create_sample_ohlcv(bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic OHLCV data for testing."""
    rng = np.random.default_rng(seed)

    # Random walk for close prices
    returns = rng.normal(0, 0.001, bars)
    close = 1.1000 + np.cumsum(returns)

    # Generate OHLC from close
    high_offset = np.abs(rng.normal(0, 0.0005, bars))
    low_offset = np.abs(rng.normal(0, 0.0005, bars))
    open_offset = rng.normal(0, 0.0003, bars)

    opens = close + open_offset
    highs = np.maximum(close, opens) + high_offset
    lows = np.minimum(close, opens) - low_offset
    volume = rng.integers(100, 10000, bars).astype(float)

    idx = pd.date_range("2020-01-01", periods=bars, freq="h")
    return pd.DataFrame(
        {"O": opens, "H": highs, "L": lows, "C": close, "V": volume},
        index=idx,
    )


def assert_features_shifted(df: pd.DataFrame, feature_cols: list) -> None:
    """Assert that feature columns are properly shifted (first row NaN)."""
    for col in feature_cols:
        assert col in df.columns, f"Column '{col}' not in DataFrame"
        assert pd.isna(df[col].iloc[0]), (
            f"Feature '{col}' first row is {df[col].iloc[0]}, expected NaN. "
            f"Did you forget shift_features()?"
        )


def assert_no_inf(df: pd.DataFrame, feature_cols: list) -> None:
    """Assert that no inf values exist in feature columns."""
    for col in feature_cols:
        assert col in df.columns, f"Column '{col}' not in DataFrame"
        inf_count = np.isinf(df[col].dropna()).sum()
        assert inf_count == 0, (
            f"Feature '{col}' has {inf_count} inf values. "
            f"Did you forget safe_divide()?"
        )


def create_sample_asset(symbol: str = "EURUSD") -> AssetInfo:
    """Create a sample AssetInfo for testing exit strategies."""
    defaults = {
        "EURUSD": ("FOREX", 0.00010, 0.00001),
        "GBPUSD": ("FOREX", 0.00015, 0.00001),
        "USDJPY": ("FOREX", 0.015, 0.001),
        "XAUUSD": ("COMMODITY", 0.30, 0.01),
        "BTCUSD": ("CRYPTO", 10.0, 0.01),
    }
    asset_class, spread, point = defaults.get(symbol, ("FOREX", 0.00010, 0.00001))
    return AssetInfo(
        symbol=symbol,
        asset_class=asset_class,
        spread=spread,
        point=point,
    )
