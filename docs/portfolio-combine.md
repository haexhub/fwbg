# Portfolio runs over frozen sleeves (`fwbg combine`)

`fwbg combine` evaluates several strategies (sleeves) as one portfolio in a single
research-gated run, including the card's one holdout look.

```bash
python -m fwbg.cli combine --spec portfolio.json --research-id S-NNNN --note "..."
python -m fwbg.cli combine --spec portfolio.json --research-id S-NNNN --note "..." --holdout
```

The research flags are those of a backtest run (`--reason`, `--check`, `--check-json`,
`--holdout`); `--run-id` and `--description` are optional.

## Spec

```json
{
  "name": "three sleeves",
  "sleeves": [
    {"name": "ibs", "strategy_file": "ibs_nas100.json", "max_notional": 3.0},
    {"name": "gotobi", "strategy_file": "gotobi_usdjpy.json", "max_notional": 10.0},
    {"name": "rebal", "strategy_file": "rebal_spx500.json"}
  ],
  "lookback_days": 252,
  "min_days": 126,
  "target_volatility": 0.10,
  "max_scale": 5.0,
  "start_date": "2014-01-01"
}
```

- `sleeves`: at least two; `strategy_file` is a normal strategy file (relative paths are
  relative to the spec). A sleeve must not set `start_date`, `end_date` or
  `holdout_cutoff`: the spec sets the window for all sleeves, and the register clamps it
  like a backtest window.
- `lookback_days` / `min_days`: trailing window (and minimum history) of the volatility
  estimates, in business days.
- `target_volatility`: annualised volatility the portfolio is scaled to; `max_scale`
  caps that scaling.
- `max_notional` (per sleeve, optional): cap on the sleeve's notional per unit of capital.

The gate fingerprints the whole spec with every sleeve's resolved strategy metadata and
the union of the sleeves' symbols, so changing any sleeve changes the configuration.

## What a run computes

1. Each sleeve runs through the normal per-symbol pipeline (`process_symbol`:
   data checks, walk-forward, frozen unified setting) with its own datasource and
   timeframe.
2. Its trades enter the portfolio **unsized**, as net return per unit of notional
   (`pnl_raw / entry_price`: spread, slippage and financing included), booked on their
   exit date; a sleeve with several symbols weights them equally. The sleeves' own Kelly
   sizing is not used: per-trade Kelly calibration is noisy for sparse sleeves, and
   sizing belongs to the portfolio.
3. Weights are causal: each sleeve gets `target_volatility / n / vol`, with `vol` its
   trailing volatility up to the previous day; the sum is scaled to `target_volatility`
   by the portfolio's own trailing volatility (at most `max_scale`), then each sleeve's
   notional is capped. Days before every sleeve has `min_days` of history are left out.
4. Metrics of the daily portfolio returns: Sharpe (×√252), annual return (%), max
   drawdown, p value (sign permutation of monthly returns), fold stability (share of
   8 equal time slices with a positive sum), trades. Status `ok` when p < 0.05.

An iteration combines the sleeves' walk-forward trades (development window only).

## Holdout look

`--holdout` develops every sleeve on `[start, cutoff)` and evaluates its frozen setting
once on `[cutoff, end)`, **whatever the sleeve's own development status**: the portfolio,
not the sleeve, is the hypothesis, so a sleeve that is not significant alone is still
evaluated. The reported metrics come from the portfolio's returns after the cutoff alone;
the weights there use volatility estimated on the development trades before it. If a
sleeve has no frozen setting or no trades after the cutoff the run is an `error`; the look
still counts once any sleeve's data after the cutoff was evaluated.

## Output

`grid_details/PORTFOLIO/` holds `config.json` (status, metrics, per-sleeve summary,
mean notional, sleeve Sharpe and correlation, the holdout block), `unified_metrics.json`
and `daily.csv` (portfolio return and each sleeve's notional per day). The register
records the run under the symbol `PORTFOLIO`.
