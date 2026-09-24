"""Maps cached Jev provider probability columns to fwbg's composed-signal
columns, so the existing `models/signal` plugin can consume them unmodified.

Expects `{provider}_is_long_win` / `{provider}_is_short_win` base columns
already present in ctx.df (e.g. `jev_official_is_long_win`), loaded by the
orchestrator via a CSV DataSource registered over
scripts/fetch_jev_predictions.py's cache output — see
docs/plans/2026-09-24-jev-signal-provider-plan.md).
"""

import pandas as pd
from fwbg_sdk import BaseDataLoader, register_data_loader, shift_features


@register_data_loader("jev_signal")
class JevSignalLoader(BaseDataLoader):
    name = "jev_signal"
    version = "1.0.0"

    def execute(self, ctx, **params):
        provider = params.get("provider", "jev_official")
        long_col = f"{provider}_is_long_win"
        short_col = f"{provider}_is_short_win"

        long_values = (
            ctx.df[long_col] if long_col in ctx.df.columns else pd.Series(0.0, index=ctx.df.index)
        )
        short_values = (
            ctx.df[short_col] if short_col in ctx.df.columns else pd.Series(0.0, index=ctx.df.index)
        )

        features = {
            "_composed_signal_long": long_values,
            "_composed_signal_short": short_values,
        }
        shifted = shift_features(features, ctx.df.index)
        for col in shifted.columns:
            ctx.df[col] = shifted[col]
        return ctx

    def get_default_params(self):
        return {
            "provider": "jev_official",
        }

    @classmethod
    def get_param_schema(cls) -> dict:
        return {
            "provider": {
                "type": "choice",
                "default": "jev_official",
                "description": "Which cached Jev provider's probability columns to read.",
                "choices": ["jev_official", "jev_semif"],
            },
        }
