"""Batch-call a Jev provider for every bar of an OHLC(+indicator) DataFrame.

Usage:
    uv run python scripts/fetch_jev_predictions.py --provider official \
        --asset EURUSD --tp 20 --sl 20 --horizon-bars 30
"""

import argparse
from pathlib import Path

import pandas as pd

from fwbg.data.assets import get_asset
from scripts.jev.cache import PredictionCache
from scripts.jev.client import JevProvider, ask
from scripts.jev.prompt import build_questions, build_state_text

PROVIDERS = {
    "official": JevProvider(
        name="jev_official",
        base_url="https://api.example-jev.invalid/v1/decide",  # TODO: real URL, confirmed manually
        api_key_env="JEV_OFFICIAL_API_KEY",
    ),
    "semif": JevProvider(
        name="jev_semif",
        base_url="https://llms.itemis.cloud/typesafe/v1/systemone",
        api_key_env="LITELLM_API_KEY",
    ),
}


def spread_multiple_to_pips(asset, multiple: float) -> float:
    """Convert an fwbg-native spread multiplier (the unit compute_labels()/
    GridParams/FixedExitStrategy use, e.g. tp=20 means 20x the asset's spread)
    into pips (the unit build_questions()'s prompt text describes). Both must
    describe the same price distance: asset.spread * multiple == pips * asset.point.
    """
    return (asset.spread * multiple) / asset.point


def run_batch(
    df: pd.DataFrame,
    provider: JevProvider,
    tp_pips: float,
    sl_pips: float,
    horizon_bars: int,
    cache_path,
) -> None:
    """Request uncached bar predictions and persist each successful response.

    Raise the first request error after saving earlier responses for a later run.
    """
    cache = PredictionCache(cache_path)
    pending = cache.pending_timestamps(list(df.index))
    questions = build_questions(tp_pips=tp_pips, sl_pips=sl_pips, horizon_bars=horizon_bars)

    for i, timestamp in enumerate(pending):
        print(f"{i + 1}/{len(pending)} {timestamp}")
        row = df.loc[timestamp]
        state = build_state_text(row)
        try:
            result = ask(provider, state=state, questions=questions)
        except Exception as e:
            print(f"Error at {timestamp}: {e}")
            print(f"Stopping batch early; {i}/{len(pending)} bars completed this run.")
            raise
        cache.append(timestamp, result)


def main():
    """Parse batch options, convert barrier units, and run the selected provider."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=PROVIDERS.keys(), required=True)
    parser.add_argument("--asset", default="EURUSD")
    parser.add_argument(
        "--tp",
        type=float,
        required=True,
        help="Take-profit, spread multiplier (fwbg-native convention, matches compute_labels)",
    )
    parser.add_argument(
        "--sl",
        type=float,
        required=True,
        help="Stop-loss, spread multiplier (fwbg-native convention, matches compute_labels)",
    )
    parser.add_argument("--horizon-bars", type=int, required=True)
    parser.add_argument(
        "--features-csv",
        required=True,
        help="Pre-computed OHLC+indicator CSV, indexed by timestamp "
        "(see docs/plans/2026-09-24-jev-signal-provider-plan.md, Task 7)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv, index_col=0)
    provider = PROVIDERS[args.provider]
    features_id = Path(args.features_csv).stem
    cache_path = (
        f"data/jev_cache/{args.asset}_{provider.name}"
        f"_tp{args.tp:g}_sl{args.sl:g}_h{args.horizon_bars}_{features_id}.csv"
    )

    asset = get_asset(args.asset)
    tp_pips = spread_multiple_to_pips(asset, args.tp)
    sl_pips = spread_multiple_to_pips(asset, args.sl)
    run_batch(df, provider, tp_pips, sl_pips, args.horizon_bars, cache_path)


if __name__ == "__main__":
    main()
