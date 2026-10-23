# Daily entry/exit research

The study tests the user's target of **CAGR above 15%, daily closing-equity
drawdown no worse than 40%, zero cash yield**, with moderate leverage allowed.
It runs in fwbg's portfolio research package. It uses explicit portfolio
accounting and the existing native statistical functions; it is not an ML
TP/SL-label optimizer run.

## Frozen rules and data

The initial protocol is [swing_v1.json](../examples/portfolio/swing_v1.json).
Before examining any strategy results, data inspection found that Dukascopy's
GLD, XLK, XLE and XLF daily feeds stopped on December 12, 2025. A crossed BTC
closing quote occurred on December 24, 2017. The recorded
[data amendment](../examples/portfolio/swing_v1_data_aligned.json) uses January
2018 onward for warm-up and **December 2018–November 2025** for seven complete
evaluation years. No price interpolation or correction of the crossed quote
was used. It is outside the warm-up window.

Eight markets are fixed: SPY, QQQ, GLD, XLK, XLE, XLF, BTC and ETH. Each rule is
tested on each market at 1x and 1.5x exposure: **96 candidates**, not a combined
multi-asset portfolio. Each candidate has three cost scenarios, for **288 runs**.
Changing the market is counted as another trial. All results, including failures,
are retained.

| Rule | Entry at the next session open | Exit |
|---|---|---|
| `rsi_sma` | Prior close above SMA(200), RSI(2) below 10 | Close above SMA(5), then next open |
| `rsi_level` | Same | RSI(2) above 70, then next open |
| `breakout_channel` | Close above highest high of preceding 55 sessions | Close below lowest low of preceding 20 sessions, then next open |
| `breakout_trail` | Same | Trailing stop, three ATR(14) below the highest bid high since entry |
| `pullback_ema` | EMA(50) above EMA(200), previous close below previous EMA(20), current close above EMA(20) | Close below EMA(50), then next open |
| `pullback_trail` | Same | Three-ATR trailing stop |

Both RSI rules also exit at the next open after ten held sessions. EMA uses
`adjust=False`; RSI and ATR use the installed `ta` implementations, also used
elsewhere in fwbg. Breakout windows exclude the current bar. Trailing stops
start from the entry fill minus three ATR known at the preceding close. Later
stops can only tighten, with each new high/ATR update effective on the following
bar. There is no fixed profit target for trend rules and no pyramiding.

The ideas are adaptations, not claimed replications of the complete published
systems: [Alvarez's RSI research](https://alvarezquanttrading.com/blog/rsi2-strategy-double-returns-with-a-simple-rule-change/),
[Connors on dynamic exits](https://tradingmarkets.com/recent/dynamic_exits_how_to_properly_exit_a_trade-641067),
and the [original Turtle rules](https://tradingblox.com/originalturtles/originalturtlerules.htm).

## Data and execution

`swing_data.py` uses the existing `fwbg.data.dukascopy` downloader helpers to
retrieve separate daily bid and ask bars. It does not modify the user's existing
data or global asset spread configuration. Raw responses and SHA256 hashes are
saved. ETF bars retain Dukascopy OHLC/spreads; Yahoo's dividend adjustment and
published split factors provide corporate-action adjustments. Only a split
factor supported by the action history or 1 is allowed. Independent adjusted
closing quotes must agree within 3% at the 99th percentile and 10% at the worst
observation. Missing reference sessions are rejected. Extra non-reference ETF
session dates and flat weekend placeholders removed are enumerated in the data
manifest. No missing risky prices are forward-filled.

Evaluation bars must be positive, finite, sorted, unique and internally
consistent, with nonnegative opening/closing spreads and no gap over seven
calendar days. Crypto follows the Dukascopy quotation calendar; it is not a
guarantee of continuous executable quotes on a crypto exchange.

All accounting is **USD**. It cannot be directly ranked against the preceding
EUR five-ETF experiment without accounting for currency differences. The cash
account earns **exactly zero**. Fractional positions are assumed. Unlevered
positions keep their units until an exit. Leveraged positions rebalance to 1.5x
at each available open using current portfolio equity **after transaction
costs**; this includes the additional turnover. The leverage can drift intraday.

Buys use the ask; sells and valuation use the bid. Market-order decisions use
the previous completed bar. A stop crossed by an opening gap sells at the worse
opening bid. Intrabar stops use the previously established stop, including on
the entry bar. A current-bar high cannot retroactively produce a favorable stop
fill earlier in that same bar. The daily bid/ask extrema need not be simultaneous;
this remains daily-bar simulation, not tick-level execution reconstruction.

Additional commission/slippage is 5 bp per side for ETFs and 25 bp per side for
crypto. Stress scenarios multiply these charges and observed opening spreads
by 2 and 5. These are research assumptions, not broker quotes.

Negative cash is a loan charged **8% annually**, using elapsed calendar days,
including weekends. The harshest stress uses **12%**. Positive cash receives
no interest. This models cash securities/spot with margin borrowing, **not IG
cash CFDs**, where funding can apply to the whole position rather than only
borrowed cash. It does not model perpetual funding, leveraged ETF mechanics,
taxes, margin-call rules or broker-specific borrowing availability. No claim of
IG deployability follows from a successful price-series test. Open positions
are marked at the end without a forced liquidation fee.

## Robustness and statistical power

The first 24 months precede six contiguous ten-month test windows. Each run
reruns the six historical prefixes and verifies identical equity, orders and
signals. Existing fwbg functions provide monthly sign randomization, separate
3/6-month compounded-block tests, and N=1 PSR on full and later periods. This is
not a full DSR correction for unknown prior research.

Native PBO compares 12 rule/leverage candidates **within each market**, using six
windows and all 20 symmetric half/half partitions. The score is mean monthly
return, not risk-adjusted return. PBO is not a probability of future loss.

Holm jointly corrects **192 hypotheses**: profit and excess over matched
buy-and-hold for each of the 96 candidates. Raw p-values are the largest across
monthly/3m/6m tests, full/later periods and all cost scenarios. Monte Carlo
estimates cannot be below the theoretical minimum exact sign-test probability
`2**(-number_of_blocks)` used for the selection gate.

**This deliberately retained conservative gate has insufficient power on the
available history.** Sixty later months yield only ten half-year blocks. Even
ten positive blocks have exact one-sided p >=1/1024; the first Holm comparison
across 192 hypotheses therefore has adjusted p >=0.1875. No candidate can pass
0.05. This limitation was identified before running the strategies. It would
be misleading to loosen the criterion after seeing profitable results or to
interpret its failure as proof that all strategies lack an edge.

The report separates:

- Numerical target: >15% CAGR and <=40% daily drawdown, on both full/later
  samples and under every declared cost scenario.
- Native individual checks, PBO and globally adjusted tests.
- Excess performance versus a benchmark with the same market, dates, leverage,
  costs and funding. Leveraged returns are also reported beside the 1x version.

One candidate is additionally chosen solely by mean monthly return during the
first 24 months, then held as a fixed rule for the later five years. This is
retrospective selection, not an untouched out-of-sample experiment. The research
has no separate forward sample and cannot establish a future drawdown ceiling.

## Reproduce

```bash
# Refuses an existing data folder; original downloaded snapshot stays frozen.
.venv/bin/python -m fwbg.portfolio.swing_data \
  --protocol examples/portfolio/swing_v1.json

.venv/bin/python -m fwbg.cli portfolio swing-study \
  --protocol examples/portfolio/swing_v1_data_aligned.json \
  --output test_results/portfolio/swing_v1

.venv/bin/pytest tests/test_portfolio_swing.py tests/test_portfolio.py \
  tests/test_portfolio_robustness.py tests/test_portfolio_search.py tests/test_cli.py -q
```

The run folder contains the frozen protocol and code/data hashes; 288 candidate
equity curves, trade ledgers, positions, financing and signals; matched
benchmarks; monthly returns and PBO matrices; all chronological windows;
`comparison.csv`, `results.json`, `report.md` and `risk_return.png`.
