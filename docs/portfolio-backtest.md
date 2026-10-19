# Monthly portfolio research

`fwbg portfolio` evaluates a long-only, unlevered portfolio using daily
total-return prices in one base currency. It has its own accounting and reporting
path; it does not run the per-trade TP/SL optimizer or reuse intraday validation
presets. Existing optimizer, API and live-trading behavior is unchanged.

## First experiment

The frozen specification is
[`examples/portfolio/trend_10m.json`](../examples/portfolio/trend_10m.json):

- Five 20% sleeves: US equities (SPY), developed equities excluding US/Canada
  (EFA), emerging-market equities (EEM), 7–10-year US Treasuries (IEF), gold (GLD).
- At each completed month end, hold a sleeve if its EUR total-return close is
  strictly above the simple average of the last ten month-end closes, including
  that month. Otherwise its allocation stays in EUR cash. Active sleeves are
  not rescaled to consume inactive sleeves' cash.
- Execute at the **next observed session's close**, after the previous positions
  have earned that day's return. This intentionally waits a full session and
  does not assume fills at the signal-generating close or next open.
- Rebalance all sleeves to their target weights monthly. Between rebalances,
  holdings drift with market prices. Fractional holdings are assumed.
- Compare against the same five assets rebalanced to 20% monthly, an SPY-only
  buy-and-hold portfolio, and EUR cash. All begin at the same time, with the same
  capital and cost assumptions. The equity benchmark is US equities, not a global
  equity index; the static portfolio is the primary matched comparator.
- Initial capital EUR 10,000, one-way trading cost 5 basis points per traded
  notional. **Uninvested cash earns exactly 0%.** Interest-rate data and cash-yield
  adjustments are not used. Trading costs are research assumptions, not quotes
  from IG or another broker.
- Evaluate 2006–2025, with a descriptive chronological split at 2016. No
  parameters are fitted and no search is performed. Report actual data dates;
  holiday calendars can move the first execution date.

## Run

From the repository root, using the project's virtual environment:

```bash
# Optional downloader dependency (already included in fwbg[ig]):
# pip install 'yfinance>=1.0'

.venv/bin/python -m fwbg.cli portfolio download-example \
  --output data/portfolio_example --start 2004-01-01 --end 2026-01-01

.venv/bin/python -m fwbg.cli portfolio run \
  --config examples/portfolio/trend_10m.json

# Same rules/data, double assumed transaction costs:
.venv/bin/python -m fwbg.cli portfolio run \
  --config examples/portfolio/trend_10m.json --cost-multiplier 2
```

An installed CLI also accepts `fwbg portfolio …`. Use `--output DIR` on `run`
for a named result directory; otherwise each run creates a unique directory
under `test_results/portfolio/`. Existing output directories are refused.
Backtests read only local inputs and do not download or silently refresh data.
The downloader is an optional fixed-universe example, not a general market-data
service. It downloads ETF prices and EUR/USD valuation rates only.

## Inputs and data limitations

The JSON paths are relative to the config file, not the shell working directory.
`prices` is a wide CSV:

```csv
date,SPY,EFA,EEM,IEF,GLD
2020-01-02,100.0,100.0,100.0,100.0,100.0
```

Every configured asset needs a positive finite daily total-return valuation on
every row. Dates must be sorted, unique, timezone-free daily labels. The engine
rejects gaps longer than seven calendar days and never forward-fills risky-asset
prices. Missing rows for individual assets must be resolved at the data source.
Use a shared trading calendar; mixed-calendar crypto/ETF portfolios are outside
this initial implementation. `price_basis: "total_return"` is a required input
declaration, not automatic verification of adjustment quality.

Uninvested capital stays at its nominal EUR amount. There is no cash-interest
parameter, no risk-free yield, and no interest-rate input. Legacy configs with
`cash_rates`, `cash_rate_spread_bps` or `cash_day_count` are rejected rather than
silently changing the experiment. Old downloaded bundles may contain unused
interest-rate files or source notes; these are not read by the backtest. Earlier
result directories with cash interest remain historical records and are
superseded by the zero-cash-return runs.

The example downloader uses:

1. [Yahoo Finance adjusted closes via yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
   These proxy reinvested distributions and splits; fund expenses are already
   reflected in the ETF price path. No additional dividend payment or management
   fee is added. These are currently available adjusted histories, not a
   point-in-time archive. Provider revisions remain possible.
2. ECB USD-per-EUR reference observations (`EXR/D.USD.EUR.SP00.A`), dividing USD
   prices by the most recent available fixing, at most seven calendar days old.
   The same-day fixing is published before US ETF closes. These are **valuation
   proxies, not simultaneous executable FX quotes**; the ECB explicitly
   [discourages transactional use](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html).
   The cost assumption must eventually be replaced by actual FX and instrument
   execution costs.

No prices are invented before ETF inception: the common history starts with the
latest first observation, followed by the ten-month signal warm-up. The five
funds are surviving historical research proxies, not a point-in-time investable
universe or a promise that EU retail accounts can purchase these US products.
Taxes, CFD funding, leverage, shorts, options, and futures rolls are not modeled.
Use this mode for fully funded holdings. Applying its results to CFDs would
require a separate funding and execution model.

## Accounting and outputs

At each valuation, existing risky holdings earn their daily total return;
uninvested cash has zero return. Rebalancing solves
`NAV_after + cost_rate * sum(abs(target_weights * NAV_after - old_values)) = NAV_before`.
Fees therefore come from existing capital; reaching 100% invested never borrows
cash to pay fees. Entry, exit and rebalancing all incur costs. End-of-test
positions remain marked to market; no liquidation is silently assumed.

Each result directory contains:

- `results.json`: effective costs, full-period and segment metrics, input SHA-256
  hashes, source metadata and timestamps; status is `historical_research_only`.
- `config.json`: the original input specification; `results.json` additionally
  records effective costs when `--cost-multiplier` is used.
- `equity.csv`, `equity.png`: daily portfolio value and drawdowns, including an
  initial pre-trade valuation so the first trading cost cannot disappear.
- `signals.csv`: decision date, later execution date, price, SMA and target.
- `trades.csv`: signed base-currency notionals and costs; these are allocation
  transactions, not independent statistical observations or closed-trade P&Ls.
- `weights.csv`: daily post-trade weights and cash fraction.
- `annual_returns.csv`: calendar-year returns; first/last years may be partial.
- `monthly_returns.csv`: non-overlapping, compounded monthly returns used for
  statistical tests (initial costs included, partial terminal months excluded).
- `robustness_folds.csv`: eight chronological evaluation windows and checks that
  truncating future data cannot alter earlier portfolio values/signals.
- `robustness.json`: native FWBG test outputs, thresholds, seed, observation
  counts, explicit applicability limits and hashes of the shared implementations.
- `report.md`: performance, statistical checks and assumptions.

CAGR uses actual elapsed calendar time. Volatility and Sharpe (reference return 0%) use
daily portfolio returns and the observed annual sampling frequency. Drawdown
uses daily marked values, including open positions, not only completed trades.
The chronological split retains holdings and includes the boundary return;
there is no restart or omitted cost at the split. It is a descriptive historical
check, **not an untouched live holdout or proof of significance**. Published
rules, universe selection and repeated research can all introduce selection
bias. Assess the matched comparator, both segments and the cost stress before
considering an implementable instrument/broker configuration.

## Robustness checks

Every `portfolio run` calls existing FWBG functions, not reimplementations:

- `fwbg.simulation.trade.monte_carlo_permutation_test`: 10,000 sign permutations,
  seed 42, on the 240 monthly returns in the example. Significance requires
  `p < 0.05`. Portfolio returns use a constant unit of capital so allocation
  transactions are not counted as independent trades. A positive result tests
  profitability above zero, not skill beyond market exposure.
- The same native sign test on non-overlapping three- and six-month compounded
  returns. This preserves within-block dependence. Incomplete trailing blocks
  are excluded and counted in the output. Blocks still need symmetric null signs
  and sufficiently independent blocks; this is a sensitivity check, not a cure
  for all time-series dependence.
- `fwbg.optimization.overfitting.compute_overfitting_metrics`: non-annualized
  monthly Sharpe, monthly skewness/kurtosis and the actual number of months.
  **One fixed rule means N=1: this is PSR, not a multiple-testing-adjusted DSR.**
  The native significance threshold is >0.95. Prior strategy-search history is
  unknown and is explicitly not treated as zero. No full deflation claim is made.
- `fwbg.optimization.robust_validation.create_walk_forward_folds`: eight
  expanding-history, non-overlapping test windows, preceded by at least 24
  months of history. For 2006–2025 this yields eight 27-month windows, 2008–2025.
  The fixed rule is not retrained or selected. Each prefix is rerun with future
  prices removed; both valuations and signals must match the original prefix.
  The same statistics are reported on the 216 concatenated test months.
- The shared `monte_carlo_equity_from_returns` is also called, but only recorded
  as a diagnostic. Permuting percentage-return order leaves final compounded
  wealth unchanged. Its identical terminal percentiles are **not a risk interval**
  and are not used as a pass/fail criterion.

`profitability_checks_passed` requires monthly MC p <0.05, single-rule PSR >0.95,
3/6-month-block p <0.05 on both the full sample and chronological test windows,
and prefix invariance. Window-by-window returns and the positive-window count
are reported separately; this does not mean every window is profitable.

`benchmark_outperformance_checks_passed` applies those checks to **paired
trend-minus-static returns**. For each longer block, first compound each
portfolio separately, then subtract their block returns. This prevents a
profitable but weaker strategy from being labeled an outperformer. Differences
are not volatility-adjusted alpha; both portfolios retain their original sizing.

**PBO is not applicable to the single-rule command:** no alternative candidate matrix was searched. Cash,
benchmarks and doubled trading costs are not fabricated parameter candidates.
A one-candidate PBO of zero would be misleading. Closed-trade count/win-rate
filters, ML feature stability and inner/outer-selection ratios likewise do not
apply to a fixed portfolio rule. `overfitting_validation_complete` remains false.

The full sample has already been inspected. These are retrospective checks, not
new untouched out-of-sample observations. No new optimizer, sizing overlay,
parameter search, cash yield or selection of the best test result is introduced.
See the original [DSR paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
and [PBO paper](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) for the
selection-history and candidate-matrix requirements.

## Predeclared search across additional rules

```bash
.venv/bin/python -m fwbg.cli portfolio search \
  --protocol examples/portfolio/search_v1.json \
  --output test_results/portfolio/search_v1
```

The protocol was written before running the additional candidates on market
data. It records every candidate, including the previously examined ten-month
rule, the sources, cost scenarios and pass criteria. This is a local research
record, not an independently registered or untouched experiment. All rules use
the same frozen 2006–2025 EUR data and 0% cash return. No new market data are
downloaded and no parameters are optimized after viewing the new results.

| Identifier | Monthly target rule |
|---|---|
| `sma_10` | Original 20% sleeves above their ten-month mean. |
| `absolute_12` | Each 20% sleeve invested only when its twelve-month return is strictly positive. |
| `relative_12_top2` | 50% in each of the two highest twelve-month returns, including negative ones. |
| `dual_12_top2` | Same ranking; a selected negative or zero-return slot remains cash. No redistribution. |
| `equity_dual_12` | 100% in the strongest of SPY/EFA/EEM if its twelve-month return is positive; otherwise 100% IEF if its return is positive; otherwise cash. |
| `reversal_1_top2` | 50% in each of the two weakest assets over the last complete month; no trend filter. An exploratory cross-asset reversal hypothesis. |
| `inverse_vol_12` | Allocate inversely to the sample standard deviation of the last twelve monthly returns, normalized to 100%. Zero-volatility assets are excluded; all-zero means cash. No leverage or return forecast. |

Momentum needs thirteen month-end closes for twelve returns. SMA needs ten
closes. All rules execute at the next session's close; ties use alphabetical
asset order. The accounting output retains the column name `trend` for the
selected strategy, even for reversal and inverse volatility. Signal logs record
the per-asset decision statistic and actual target weights.

Momentum ideas come from [Moskowitz, Ooi and Pedersen](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)
and [Antonacci](https://www.optimalmomentum.com/dual-relative-absolute-momentum/).
These ETF rules are long-only adaptations. They do not replicate futures
long/short positions, volatility targeting, risk-free hurdles or the exact
published GEM model. Reversal and inverse-volatility allocation are exploratory
comparators, not findings attributed to those momentum sources.

Each of the seven candidates runs at 5, 10 and 25 bp per traded notional, making
21 complete runs. Each run includes the original monthly and 3/6-month-block
sign tests, N=1 PSR diagnostics, eight chronological test windows and eight
prefix reruns. Benchmarks and complete monthly dates must match across candidates.

The search adds two selection checks:

- Native `probability_of_backtest_overfitting` receives a real seven-by-eight
  matrix of mean monthly returns, using the same eight chronological test windows.
  It evaluates all 70 symmetric four/four partitions at each cost level. The
  native threshold is PBO >50% for overfitting. This compares raw mean returns,
  without equalizing volatility; it is a ranking diagnostic, not a profitability
  probability. Symmetric splits can train on later periods and are not a live
  chronological trading simulation.
- Holm corrects 14 hypotheses jointly: positive returns and outperformance of
  static allocation for each of seven rules. Each raw p-value is the largest
  across full/chronological samples, monthly/3/6-month tests and all cost levels.
  This intersection criterion requires all components to pass. Holm does not
  require independent candidates, but it still depends on valid component tests.
  It does not correct unknown earlier searches or publication bias.

The combined profitability gate requires the original checks at every cost
level, Holm p <0.05, and native PBO <=50% at every cost level. Benchmark excess
performance has its own gate; passing profitability does not imply alpha.
No full DSR correction is claimed. The shared N>1 DSR routine lacks a
search-specific variance scale in its expected-maximum Sharpe term; the existing
N=1 PSR diagnostics are retained and Holm supplies this search's correction.
The original global optimizer implementation is unchanged.

Separately, the candidate with the highest mean monthly return before 2016 is
chosen once, and its 2016–2025 metrics are reported at all three cost levels.
This illustrates chronological selection; it is not a fresh holdout because
the market history was already inspected.

Artifacts include the frozen protocol and input/code hashes, all 21 individual
runs with transactions/signals/weights, candidate monthly returns, PBO matrices,
`comparison.csv`, `search_results.json`, `report.md` and `comparison.png`.
`manifest.json` starts as `running` and becomes `complete` only after all outputs
are written. A failed run's folder remains for audit and is never overwritten.

## Verification

```bash
.venv/bin/python -m pytest tests/test_portfolio.py tests/test_portfolio_robustness.py tests/test_portfolio_search.py tests/test_cli.py tests/optimization/test_overfitting.py tests/simulation/test_monte_carlo.py tests/test_robust_validation.py -q
.venv/bin/ruff check src/fwbg/portfolio tests/test_portfolio*.py
```

Regression tests cover delayed fills, future-prefix invariance, partial months,
self-financed fees, intramonth drawdowns, zero cash returns, FX direction, invalid
inputs, benchmarks, reporting and actual calls into the native FWBG statistical
functions. Network access is unnecessary for tests and local backtests.
