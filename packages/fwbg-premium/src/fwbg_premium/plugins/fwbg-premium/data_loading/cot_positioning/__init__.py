"""
COT Positioning DataLoader — computes features from CFTC positioning data.

Expects COT data to be loaded as macro_cot_{symbol} base columns via DataSource.
Computes: z-scores, extremes, momentum, crowded trade flags.
"""
from fwbg_sdk import BaseDataLoader, register_data_loader


# COT indicator mapping: file stem → column prefix
DEFAULT_COT_INDICATORS = {
    "COT_EURUSD_DAY": "cot_eurusd",
    "COT_USDJPY_DAY": "cot_usdjpy",
    "COT_GBPUSD_DAY": "cot_gbpusd",
    "COT_USDCAD_DAY": "cot_usdcad",
    "COT_AUDUSD_DAY": "cot_audusd",
    "COT_USDCHF_DAY": "cot_usdchf",
    "COT_NZDUSD_DAY": "cot_nzdusd",
}

DEFAULT_LOOKBACKS_WEEKS = [1, 4, 12, 26]
ZSCORE_WINDOW_WEEKS = 52


@register_data_loader("cot_positioning")
class COTPositioningLoader(BaseDataLoader):
    """Computes COT positioning features from net position base columns."""

    name = "cot_positioning"
    version = "1.0.0"

    def execute(self, ctx, **params):
        df = ctx.df
        lookbacks = params.get("lookbacks_weeks", DEFAULT_LOOKBACKS_WEEKS)
        zscore_weeks = params.get("zscore_window_weeks", ZSCORE_WINDOW_WEEKS)

        # Find all COT base columns (loaded by DataSource as macro_cot_*)
        cot_cols = [c for c in df.columns if c.startswith("macro_cot_")]

        for col in cot_cols:
            prefix = col.replace("macro_", "")  # e.g. "cot_eurusd"
            net = df[col]

            # 52-week z-score normalization
            window = zscore_weeks * 5 * 24  # weeks → H1 bars
            roll_mean = net.rolling(window, min_periods=window // 4).mean()
            roll_std = net.rolling(window, min_periods=window // 4).std().clip(lower=1e-6)
            zscore = (net - roll_mean) / roll_std
            df[f"{prefix}_zscore"] = zscore

            # Positioning extremes
            df[f"{prefix}_extreme_long"] = (zscore > 2.0).astype(float)
            df[f"{prefix}_extreme_short"] = (zscore < -2.0).astype(float)

            # Crowded trade flag (absolute z-score > 1.5)
            df[f"{prefix}_crowded"] = (zscore.abs() > 1.5).astype(float)

            # Week-based momentum
            for lb in lookbacks:
                bars = lb * 5 * 24  # weeks → H1 bars
                df[f"{prefix}_chg_{lb}w"] = net.pct_change(bars) * 100

        # Shift all COT features by 1 bar (lookahead prevention)
        cot_feature_cols = [c for c in df.columns
                           if c.startswith("cot_") and not c.startswith("macro_cot_")]
        for col in cot_feature_cols:
            df[col] = df[col].shift(1)

        ctx.df = df
        return ctx

    def get_default_params(self):
        return {
            "indicators": DEFAULT_COT_INDICATORS,
            "lookbacks_weeks": DEFAULT_LOOKBACKS_WEEKS,
            "zscore_window_weeks": ZSCORE_WINDOW_WEEKS,
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "indicators": {
                "type": "string",
                "default": DEFAULT_COT_INDICATORS,
                "description": "Mapping of COT data file stems to column prefixes (e.g. COT_EURUSD_DAY -> cot_eurusd)",
            },
            "lookbacks_weeks": {
                "type": "list[int]",
                "default": DEFAULT_LOOKBACKS_WEEKS,
                "description": "Week-based lookback periods for computing net positioning momentum",
            },
            "zscore_window_weeks": {
                "type": "int",
                "default": ZSCORE_WINDOW_WEEKS,
                "description": "Rolling window in weeks for z-score normalization of net positions",
                "min": 1,
                "max": 520,
                "step": 1,
            },
        }

    def get_feature_columns(self, **params):
        """Dynamically compute feature column names."""
        indicators = params.get("indicators", DEFAULT_COT_INDICATORS)
        lookbacks = params.get("lookbacks_weeks", DEFAULT_LOOKBACKS_WEEKS)

        cols = []
        for prefix in indicators.values():
            cols.extend([
                f"{prefix}_zscore",
                f"{prefix}_extreme_long",
                f"{prefix}_extreme_short",
                f"{prefix}_crowded",
            ])
            for lb in lookbacks:
                cols.append(f"{prefix}_chg_{lb}w")

        return cols
