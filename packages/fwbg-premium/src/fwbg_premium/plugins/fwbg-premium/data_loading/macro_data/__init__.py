"""
MacroDataLoader — computes features from macro base columns.

Pure computation plugin (no I/O). Expects macro_* base columns
to already be present in ctx.df (loaded by orchestrator via DataSource).

Features computed:
- Hourly lookback changes: macro_{prefix}_chg_{lb}h
- Daily lookback changes: macro_{prefix}_chg_{lb}d
- Derived features (subtract/ratio): yield curves, risk ratios
- Interest rate changes and diffs
"""
from fwbg_sdk import BaseDataLoader, register_data_loader


# Default configuration (moved from data/config.py)
DEFAULT_INDICATORS = {
    "VIX_DAY": "vix", "VVIX_DAY": "vvix", "SKEW_DAY": "skew",
    "VXN_DAY": "vxn", "TNX_DAY": "tnx", "TYX_DAY": "tyx",
    "FVX_DAY": "fvx", "IRX_DAY": "irx", "DXY_DAY": "dxy",
    "GOLD_FUT_DAY": "gold_fut", "OIL_FUT_DAY": "oil",
    "SILVER_FUT_DAY": "silver_fut", "SPX_DAY": "spx",
    "NASDAQ_DAY": "nasdaq", "DOW_DAY": "dow", "RUSSELL_DAY": "russell",
    "NIKKEI_DAY": "nikkei", "HANGSENG_DAY": "hangseng",
    "FTSE_DAY": "ftse_idx", "DAX_IDX_DAY": "dax_idx",
    "XLF_DAY": "xlf", "XLE_DAY": "xle", "XLK_DAY": "xlk",
    "XLU_DAY": "xlu", "XLP_DAY": "xlp", "TLT_DAY": "tlt",
    "HYG_DAY": "hyg", "LQD_DAY": "lqd",
    # US Treasury curve (for yield curve shape)
    "US2Y_DAY": "us2y", "US5Y_DAY": "us5y", "US30Y_DAY": "us30y",
    # International bond yields (for FX yield spreads)
    "DE10Y_DAY": "de10y", "JP10Y_DAY": "jp10y",
    "GB10Y_DAY": "gb10y", "AU10Y_DAY": "au10y",
}

DEFAULT_LOOKBACKS_HOURS = [1, 2, 4, 8, 12, 24]
DEFAULT_LOOKBACKS_DAYS = [2, 5, 10, 20, 60]

DEFAULT_DERIVED_FEATURES = [
    {"name": "macro_yield_curve_10y_3m", "op": "subtract", "a": "macro_tnx", "b": "macro_irx"},
    {"name": "macro_yield_curve_10y_5y", "op": "subtract", "a": "macro_tnx", "b": "macro_fvx"},
    {"name": "macro_yield_curve_10y_2y", "op": "subtract", "a": "macro_tnx", "b": "macro_us2y"},
    {"name": "macro_yield_curve_30y_5y", "op": "subtract", "a": "macro_us30y", "b": "macro_us5y"},
    {"name": "macro_vix_vvix_ratio", "op": "ratio", "a": "macro_vix", "b": "macro_vvix"},
    {"name": "macro_risk_ratio_spx_tlt", "op": "ratio", "a": "macro_spx", "b": "macro_tlt"},
    {"name": "macro_credit_spread_proxy", "op": "ratio", "a": "macro_hyg", "b": "macro_lqd"},
    {"name": "macro_smallcap_ratio", "op": "ratio", "a": "macro_russell", "b": "macro_spx"},
    {"name": "macro_tech_defensive_ratio", "op": "ratio", "a": "macro_xlk", "b": "macro_xlu"},
    # Yield spreads (US vs international — FX carry signal)
    {"name": "macro_yield_spread_us_de", "op": "subtract", "a": "macro_tnx", "b": "macro_de10y"},
    {"name": "macro_yield_spread_us_jp", "op": "subtract", "a": "macro_tnx", "b": "macro_jp10y"},
    {"name": "macro_yield_spread_us_gb", "op": "subtract", "a": "macro_tnx", "b": "macro_gb10y"},
    {"name": "macro_yield_spread_us_au", "op": "subtract", "a": "macro_tnx", "b": "macro_au10y"},
]

DEFAULT_INTEREST_RATES = [
    {"name": "fed", "file": "FED_RATE.csv", "lookbacks_days": [30, 90, 180]},
    {"name": "ecb", "file": "ECB_RATE.csv", "lookbacks_days": [30, 90, 180]},
]

DEFAULT_INTEREST_RATE_DIFFS = [
    {"name": "macro_rate_diff_usd_eur", "a": "macro_fed_rate", "b": "macro_ecb_rate"},
]


@register_data_loader("macro_data")
class MacroDataLoader(BaseDataLoader):
    """Computes macro-derived features from base macro columns."""

    name = "macro_data"
    version = "1.0.0"

    def execute(self, ctx, **params):
        df = ctx.df
        lookbacks_hours = params.get("lookbacks_hours", DEFAULT_LOOKBACKS_HOURS)
        lookbacks_days = params.get("lookbacks_days", DEFAULT_LOOKBACKS_DAYS)
        derived = params.get("derived_features", DEFAULT_DERIVED_FEATURES)
        rate_diffs = params.get("interest_rate_diffs", DEFAULT_INTEREST_RATE_DIFFS)

        # Find all macro base columns
        macro_cols = [c for c in df.columns if c.startswith("macro_") and "_chg_" not in c]

        # Hourly lookbacks
        for col in macro_cols:
            for lb in lookbacks_hours:
                df[f"{col}_chg_{lb}h"] = df[col].pct_change(lb) * 100

        # Daily lookbacks (24 bars per day for hourly data)
        for col in macro_cols:
            for lb in lookbacks_days:
                df[f"{col}_chg_{lb}d"] = df[col].pct_change(24 * lb) * 100

        # Derived features (subtract/ratio)
        for spec in derived:
            a, b = spec["a"], spec["b"]
            if a in df.columns and b in df.columns:
                if spec["op"] == "subtract":
                    df[spec["name"]] = df[a] - df[b]
                elif spec["op"] == "ratio":
                    df[spec["name"]] = df[a] / (df[b] + 1e-10)

        # Lookbacks for derived features (spread momentum/trend)
        derived_cols = [spec["name"] for spec in derived
                        if spec["name"] in df.columns]
        for col in derived_cols:
            for lb in lookbacks_days:
                chg_key = f"{col}_chg_{lb}d"
                if chg_key not in df.columns:
                    df[chg_key] = df[col].pct_change(24 * lb) * 100

        # Interest rate diffs
        for diff in rate_diffs:
            a, b = diff["a"], diff["b"]
            if a in df.columns and b in df.columns:
                df[diff["name"]] = df[a] - df[b]

        ctx.df = df
        return ctx

    def get_default_params(self):
        return {
            "indicators": DEFAULT_INDICATORS,
            "lookbacks_hours": DEFAULT_LOOKBACKS_HOURS,
            "lookbacks_days": DEFAULT_LOOKBACKS_DAYS,
            "derived_features": DEFAULT_DERIVED_FEATURES,
            "interest_rates": DEFAULT_INTEREST_RATES,
            "interest_rate_diffs": DEFAULT_INTEREST_RATE_DIFFS,
        }

    def get_feature_columns(self, **params):
        """Dynamically compute feature column names from params."""
        indicators = params.get("indicators", DEFAULT_INDICATORS)
        lookbacks_hours = params.get("lookbacks_hours", DEFAULT_LOOKBACKS_HOURS)
        lookbacks_days = params.get("lookbacks_days", DEFAULT_LOOKBACKS_DAYS)
        derived = params.get("derived_features", DEFAULT_DERIVED_FEATURES)

        cols = []
        for prefix in indicators.values():
            base = f"macro_{prefix}"
            cols.append(base)
            for lb in lookbacks_hours:
                cols.append(f"{base}_chg_{lb}h")
            for lb in lookbacks_days:
                cols.append(f"{base}_chg_{lb}d")

        for spec in derived:
            cols.append(spec["name"])
            for lb in lookbacks_days:
                cols.append(f"{spec['name']}_chg_{lb}d")

        return cols
