# Bot Risk Contract

The trading bot circuit breaker uses broker-confirmed realized PnL from closed
trades. A position becoming absent, an order rejection, or an unrealized PnL
change is not a realized trade event.

For each account-local calendar day, the risk state sums the broker's net PnL
for confirmed closed trades. The broker-reported value includes fees and
commissions. Balance transfers, deposits, and withdrawals are excluded because
they are not closed-trade events. Every event must have a stable deal or event
ID; duplicate delivery of an ID is ignored.

The state tracks the current day's realized PnL and the consecutive losing
closed trades. A winning close resets the loss streak. Rejected orders are
counted separately for diagnostics and never change realized PnL or the loss
streak. At the account-local day boundary, the daily PnL and loss streak start
fresh while previously seen event IDs remain known for replay protection.

Trading is fail-closed. Before every new order, the bot must have a successful,
recent broker observation, a valid account balance, and a parseable closed
trade event feed. Unknown, malformed, failed, or stale broker state blocks the
order. A restart may replay the broker's recent transaction history; event IDs
make that replay idempotent and produce the same daily totals.

The daily loss limit is measured as:

```text
max(0, -realized_net_pnl_for_local_day) / balance_at_local_day_start
```

The denominator is captured from the broker at the first known observation of
the local day. Transfers therefore cannot silently reset or satisfy the loss
limit.
