# Data checks and trading hours

Every run checks a symbol's bars after loading (resampling, date window) and before
any indicator sees them (`fwbg.data.validation.check_bars`, called from
`load_symbol_frame`), in development and holdout alike.

## Strategy key

```json
"data_checks": "strict"
```

- `strict` (default): an error stops the symbol with status `data_invalid`; the
  report is in the symbol's result under `data_checks`.
- `warn`: errors are logged and reported, the run goes on.
- `off`: no checks, filler bars stay.

## What is checked

| Code | Severity | Meaning |
|---|---|---|
| `unordered_timestamps`, `duplicate_timestamps` | error | the index is not strictly increasing |
| `invalid_prices` | error | non-finite or non-positive O/H/L/C |
| `ohlc_inconsistent` | error | low above open/close or high below open/close |
| `outside_trading_hours` | error above 5 % of bars, else warning | bars while the asset does not trade; many of them mean wrong time stamps (time zone, bar labelling) |
| `long_gap` | warning | more than 5 days of bars missing in one gap inside the trading hours |
| `price_spikes` | warning | a close-to-close move above 20× the recent median that reverses by half on the next bar (bad tick or real shock) |
| `wide_bars` | warning | a bar range above 15× the recent median |
| `stale_prices` | warning | 5+ bars without any price change |
| `missing_bars` | info | bars missing inside the trading hours, holidays included |
| `filler_bars_dropped` | info | flat bars without volume (or, for sources without volume, flat bars outside the trading hours) were removed |

The checks never alter prices. Dropping filler bars is the only change: they record
no trade (Dukascopy writes them for weekends and holidays). An extreme bar can be a
real market event, so spikes and wide bars are reported, not smoothed.

## Trading hours

Built-in sessions per asset class and symbol group are in
`src/fwbg/data/trading_hours.json` (forex: Sunday 17:00 to Friday 17:00 New York;
US index CFDs: Globex hours plus a `cash` session 09:30-16:00 New York; metals;
crypto around the clock). A six-letter pair of ISO currency codes counts as forex.
Symbols without a definition get the price checks only.

Override or add hours per symbol in the workspace's `_asset_meta.json`:

```json
{
  "QQQ": {
    "trading_hours": {
      "tradable": {"tz": "America/New_York", "windows": [{"days": "Mon-Fri", "start": "09:30", "end": "16:00"}]}
    }
  }
}
```

A window opens on each listed day (`"Mon-Fri"`, `"Sun-Thu"`, `"Mon,Wed"`) at `start`
local time and closes at `end`, on the next day when `end` is not after `start`
(`"17:00"` to `"17:00"` is 24 hours). Holidays and early closes are not modelled.
Bar time stamps without a time zone are UTC and mark the bar's start.

## Session bars

Daily bars of one session (e.g. the US `cash` session 09:30-16:00 New York) are
built from intraday bars by fwbg, not by hand:

```
python -m fwbg.cli data session-bars --input ~/fwbg/data/dukascopy/datasource/NAS100_MINUTE_15.csv \
    --session cash --datasource dukascopy_cash
```

A source bar joins a session only when it lies entirely inside the session window
(minutes of the week, so a bar just before midnight never passes for one before
16:00). A session counts when its first bar starts at the open, its last bar ends
at the close and at least half its bars exist; early closes and holes are dropped.
Bars are labelled with the local date of the session close. The source passes the
data checks first (filler bars dropped). Next to `NAS100_DAY_1.csv` the command
writes `NAS100_DAY_1.session.json` (session, source, sessions seen and dropped,
source checks) and a datasource `config.json` when there is none.
