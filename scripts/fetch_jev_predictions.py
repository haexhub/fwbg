"""Batch-call a Jev provider for every bar of an OHLC(+indicator) DataFrame.

Usage:
    uv run python scripts/fetch_jev_predictions.py --provider official \
        --asset EURUSD --tp 20 --sl 20 --horizon-bars 30
"""

import argparse

import pandas as pd

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


def run_batch(
    df: pd.DataFrame,
    provider: JevProvider,
    tp_pips: float,
    sl_pips: float,
    horizon_bars: int,
    cache_path,
) -> None:
    cache = PredictionCache(cache_path)
    pending = cache.pending_timestamps(list(df.index))
    questions = build_questions(tp_pips=tp_pips, sl_pips=sl_pips, horizon_bars=horizon_bars)

    for timestamp in pending:
        row = df.loc[timestamp]
        state = build_state_text(row)
        result = ask(provider, state=state, questions=questions)
        cache.append(timestamp, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=PROVIDERS.keys(), required=True)
    parser.add_argument("--asset", default="EURUSD")
    parser.add_argument("--tp", type=float, required=True, help="Take-profit, pips")
    parser.add_argument("--sl", type=float, required=True, help="Stop-loss, pips")
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
    cache_path = (
        f"data/jev_cache/{args.asset}_{provider.name}_tp{int(args.tp)}_sl{int(args.sl)}.csv"
    )
    run_batch(df, provider, args.tp, args.sl, args.horizon_bars, cache_path)


if __name__ == "__main__":
    main()
