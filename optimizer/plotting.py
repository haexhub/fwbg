"""
Plot-Funktionen für Equity-Kurven und Trade-Analyse.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .equity import simulate_equity


def create_incremental_plot(result, plots_path):
    """
    Erstellt einen einfachen Equity-Plot für ein einzelnes Asset.
    Wird während der Verarbeitung für jedes profitable Asset aufgerufen.
    """
    sym = result.get("symbol", "?")
    config = result.get("config", {})
    trades = result.get("tr_trace", [])

    if not trades:
        return

    kelly = config.get("kelly_risk", 0.01)
    rrr = result.get("rrr", 1.0)

    # Equity simulieren
    eq_result = simulate_equity(trades, kelly, rrr)
    eq = eq_result["equity_curve"]
    drawdowns = eq_result["drawdowns"]
    max_dd = eq_result["max_drawdown"]

    # Gewinn pro Trade berechnen (aus Equity-Kurve)
    profit_per_trade = []
    for i in range(1, len(eq)):
        profit_per_trade.append(eq[i] - eq[i - 1])

    # Plot erstellen - 3 Subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), height_ratios=[3, 1, 1])

    ax1.plot(eq, color="blue", linewidth=1.5)
    ax1.fill_between(range(len(eq)), eq, alpha=0.3)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", ".")))

    ct_val = config.get("conf_thresh", 0)
    ax1.set_title(
        f"{sym} | WR: {result.get('win_rate', 0):.1%} | RRR: {rrr:.2f} | "
        f"MaxDD: {max_dd*100:.0f}% | CT: {ct_val:.2f}"
    )
    ax1.set_ylabel("Kapital (log)")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(range(len(drawdowns)), drawdowns, color="red", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(max(drawdowns) * 1.1 if drawdowns else 1, 0)
    ax2.grid(True, alpha=0.3)

    # Profit per Trade als Bar-Chart (grün = Gewinn, rot = Verlust)
    colors = ["green" if p > 0 else "red" for p in profit_per_trade]
    ax3.bar(range(len(profit_per_trade)), profit_per_trade, color=colors, alpha=0.7, width=1.0)
    ax3.axhline(y=0, color="black", linewidth=0.5)
    if profit_per_trade:
        wins = [p for p in profit_per_trade if p > 0]
        losses = [p for p in profit_per_trade if p < 0]
        if wins:
            avg_win = sum(wins) / len(wins)
            ax3.axhline(y=avg_win, color="green", linestyle="--", alpha=0.7, label=f"Ø Win: {avg_win:,.0f}")
        if losses:
            avg_loss = sum(losses) / len(losses)
            ax3.axhline(y=avg_loss, color="red", linestyle="--", alpha=0.7, label=f"Ø Loss: {avg_loss:,.0f}")
        ax3.legend(loc="upper right", fontsize=8)
    ax3.set_xlabel("Trade #")
    ax3.set_ylabel("Gewinn/Trade")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{plots_path}/{sym}.png", dpi=100)
    plt.close()


def create_elite_plot(e, plots_path, trade_directions, profit_per_trade, eq, drawdowns, max_dd, rrr):
    """
    Erstellt detaillierten Plot für ein Elite-Asset.
    """
    config = e.get("config", {})

    # Long/Short Statistiken
    n_long = sum(1 for d in trade_directions if d == "LONG")
    n_short = sum(1 for d in trade_directions if d == "SHORT")
    long_wins = sum(1 for i, d in enumerate(trade_directions) if d == "LONG" and i < len(profit_per_trade) and profit_per_trade[i] > 0)
    short_wins = sum(1 for i, d in enumerate(trade_directions) if d == "SHORT" and i < len(profit_per_trade) and profit_per_trade[i] > 0)
    long_wr = long_wins / n_long if n_long > 0 else 0
    short_wr = short_wins / n_short if n_short > 0 else 0

    # Equity Plot mit Drawdown und Profit per Trade (logarithmisch)
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 9), height_ratios=[3, 1, 1]
    )

    ax1.plot(eq, color="blue", linewidth=1.5)
    ax1.fill_between(range(len(eq)), eq, alpha=0.3)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", ".")))

    # CT-Werte für Titel (separate oder gemeinsam)
    if config.get("separate_long_short") and config.get("ct_long") != config.get("ct_short"):
        ct_str = f"CT: L={config.get('ct_long', 0):.2f}/S={config.get('ct_short', 0):.2f}"
    else:
        ct_str = f"CT: {config.get('conf_thresh', 0):.2f}"

    ax1.set_title(
        f"{e['symbol']} | WR: {e['win_rate']:.1%} | RRR: {rrr:.2f} | "
        f"Sharpe: {e.get('sharpe', 0):.2f} | MaxDD: {max_dd*100:.0f}%\n"
        f"Long: {n_long} ({long_wr:.0%}) | Short: {n_short} ({short_wr:.0%}) | {ct_str}"
    )
    ax1.set_ylabel("Kapital (log, Start=100)")
    ax1.set_xlabel("")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(range(len(drawdowns)), drawdowns, color="red", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(max(drawdowns) * 1.1 if drawdowns else 1, 0)
    ax2.set_xlabel("")
    ax2.grid(True, alpha=0.3)

    # Profit per Trade als Bar-Chart mit Long/Short Unterscheidung
    colors = []
    for i, p in enumerate(profit_per_trade):
        if i < len(trade_directions):
            is_long = trade_directions[i] == "LONG"
            if p > 0:
                colors.append("green" if is_long else "blue")
            else:
                colors.append("lightcoral" if is_long else "orange")
        else:
            colors.append("green" if p > 0 else "red")

    ax3.bar(
        range(len(profit_per_trade)),
        profit_per_trade,
        color=colors,
        alpha=0.7,
        width=1.0,
    )
    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.set_xlabel("Trade # (grün/rot=Long, blau/orange=Short)")
    ax3.set_ylabel("Gewinn/Trade")

    # Symmetrische Y-Achse
    if profit_per_trade:
        max_abs = max(abs(min(profit_per_trade)), abs(max(profit_per_trade)))
        ax3.set_ylim(-max_abs * 1.1, max_abs * 1.1)

    # Durchschnittlichen Gewinn/Verlust
    wins = [p for p in profit_per_trade if p > 0]
    losses = [p for p in profit_per_trade if p < 0]
    if wins:
        avg_win = sum(wins) / len(wins)
        ax3.axhline(y=avg_win, color="green", linestyle="--", alpha=0.7, linewidth=1)
        ax3.text(len(profit_per_trade) * 0.02, avg_win, f"Ø Win: {avg_win:.2f}",
                 fontsize=8, color="green", va="bottom")
    if losses:
        avg_loss = sum(losses) / len(losses)
        ax3.axhline(y=avg_loss, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax3.text(len(profit_per_trade) * 0.02, avg_loss, f"Ø Loss: {avg_loss:.2f}",
                 fontsize=8, color="red", va="top")

    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{plots_path}/{e['symbol']}.png", dpi=100)
    plt.close()

    return n_long, n_short, long_wr, short_wr
