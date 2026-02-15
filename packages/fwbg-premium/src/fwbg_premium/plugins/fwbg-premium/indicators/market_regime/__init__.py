"""
Market Regime Indicator — Risk-On/Off Composite.

Combines VIX, credit spreads, equity momentum, and treasury flight
into a composite risk regime score. Requires macro_data columns.
"""
from typing import List

import numpy as np
import pandas as pd

from fwbg.core.registry import register_indicator
from fwbg.plugins import BaseIndicator
from fwbg.plugins.indicator import shift_features


@register_indicator("market_regime")
class MarketRegimeIndicator(BaseIndicator):
    """
    Composite risk regime indicator.

    Features:
    - regime_vix_zscore: VIX z-score (inverted, high VIX = negative)
    - regime_credit_zscore: Credit spread z-score (HYG/LQD ratio)
    - regime_equity_momentum: SPX 20-day percentage change
    - regime_treasury_flight: TLT momentum (inverted)
    - regime_risk_composite: Average of z-scored components
    - regime_risk_on: Binary flag (composite > 0.5)
    - regime_risk_off: Binary flag (composite < -0.5)
    """

    name = "market_regime"
    version = "1.0.0"
    group = "regime"

    def compute(
        self,
        df: pd.DataFrame,
        window: int = 50,
        **params,
    ) -> pd.DataFrame:
        features = {}
        bars_per_day = params.get("bars_per_day", 24)

        # Component 1: VIX Z-Score (inverted: high VIX = risk-off)
        if "macro_vix" in df.columns:
            vix = df["macro_vix"]
            vix_mean = vix.rolling(window * bars_per_day, min_periods=window).mean()
            vix_std = vix.rolling(window * bars_per_day, min_periods=window).std().clip(lower=1e-6)
            features["regime_vix_zscore"] = -(vix - vix_mean) / vix_std

        # Component 2: Credit Spread (HYG/LQD ratio — high = risk-on)
        if "macro_hyg" in df.columns and "macro_lqd" in df.columns:
            credit = df["macro_hyg"] / (df["macro_lqd"] + 1e-10)
            cr_mean = credit.rolling(window * bars_per_day, min_periods=window).mean()
            cr_std = credit.rolling(window * bars_per_day, min_periods=window).std().clip(lower=1e-6)
            features["regime_credit_zscore"] = (credit - cr_mean) / cr_std

        # Component 3: Equity Momentum (SPX 20-day change)
        if "macro_spx" in df.columns:
            features["regime_equity_momentum"] = (
                df["macro_spx"].pct_change(20 * bars_per_day) * 100
            )

        # Component 4: Treasury Flight (TLT momentum — high = risk-off)
        if "macro_tlt" in df.columns:
            features["regime_treasury_flight"] = -(
                df["macro_tlt"].pct_change(10 * bars_per_day) * 100
            )

        # Composite: Average of z-scored components
        zscore_cols = [
            v for k, v in features.items()
            if k.endswith("zscore")
        ]
        if zscore_cols:
            composite = pd.concat(zscore_cols, axis=1).mean(axis=1)
            features["regime_risk_composite"] = composite
            features["regime_risk_on"] = (composite > 0.5).astype(float)
            features["regime_risk_off"] = (composite < -0.5).astype(float)

        if not features:
            return df

        features_df = shift_features(features, df.index)
        return pd.concat([df, features_df], axis=1)

    def get_feature_columns(self) -> List[str]:
        return [
            "regime_vix_zscore",
            "regime_credit_zscore",
            "regime_equity_momentum",
            "regime_treasury_flight",
            "regime_risk_composite",
            "regime_risk_on",
            "regime_risk_off",
        ]

    @classmethod
    def get_default_params(cls) -> dict:
        return {"window": 50, "bars_per_day": 24}

    def validate(self) -> bool:
        return True
