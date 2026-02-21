"""
Plot-Funktionen für Equity-Kurven und Trade-Analyse.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np

from fwbg.data.config import WALK_FORWARD_FOLDS, OOS_SIZE, TIMEFRAME
from fwbg.simulation.equity import simulate_equity_from_pnl


def _format_currency(x, _):
    """Formatiert Werte für Y-Achse ohne wissenschaftliche Notation."""
    if x <= 0:
        return ""
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}k"
    return f"{x:.0f}"


def _setup_log_yaxis(ax, eq):
    """Konfiguriert logarithmische Y-Achse mit sinnvollen Ticks."""
    ax.set_yscale("log")

    # Berechne sinnvolle Tick-Positionen basierend auf Datenbereich
    min_val = max(min(eq), 1)  # Mindestens 1 für log-Skala
    max_val = max(eq)

    # Finde passende Basis für Ticks (10, 100, 1000, etc.)
    log_min = np.floor(np.log10(min_val))
    log_max = np.ceil(np.log10(max_val))

    # Erstelle manuelle Tick-Positionen
    ticks = []
    # Erweitere Bereich um eine Größenordnung in jede Richtung
    for exp in range(int(log_min) - 1, int(log_max) + 2):
        base = 10 ** exp
        # Mehr Zwischenwerte für bessere Lesbarkeit
        for mult in [1, 2, 5]:
            val = base * mult
            if min_val * 0.8 <= val <= max_val * 1.2:
                ticks.append(val)

    # Fallback: wenn keine Ticks gefunden, erzeuge gleichmäßig verteilte
    if len(ticks) < 3:
        ticks = np.logspace(np.log10(min_val), np.log10(max_val), num=5)
        ticks = [round(t, -int(np.floor(np.log10(t))) + 1) for t in ticks]  # Runden

    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_currency))
    # Deaktiviere Minor-Tick-Labels um Clutter zu vermeiden
    ax.yaxis.set_minor_formatter(NullFormatter())


def create_asset_plot(result, plots_path, trade_directions=None):
    """
    Erstellt Equity-Plot für ein Asset.

    Einheitliche Plot-Funktion für alle Assets.

    Args:
        result: Dict mit symbol, config, tr_trace, win_rate, rrr, sharpe, etc.
        plots_path: Pfad zum Speichern des Plots
        trade_directions: Optional Liste der Trade-Richtungen (LONG/SHORT) für
                          farbliche Unterscheidung
    """
    sym = result.get("symbol", "?")
    config = result.get("config") or result.get("best_config", {})
    trades = result.get("tr_trace", [])

    if not trades:
        return None

    risk = config.get("risk_per_trade", 0.01)
    if risk <= 0:
        risk = 0.01  # Minimum für Visualisierung

    # Equity mit echten PnL-Magnitudes simulieren (nicht binäres Kelly)
    eq_result = simulate_equity_from_pnl(trades, fk=risk)
    eq = eq_result["equity_curve"]
    drawdowns = eq_result["drawdowns"]
    max_dd = eq_result["max_drawdown"]

    # Gewinn pro Trade berechnen (aus Equity-Kurve)
    profit_per_trade = []
    for i in range(1, len(eq)):
        profit_per_trade.append(eq[i] - eq[i - 1])

    # Long/Short Statistiken (wenn vorhanden)
    n_long = n_short = long_wr = short_wr = 0
    if trade_directions:
        n_long = sum(1 for d in trade_directions if d == "LONG")
        n_short = sum(1 for d in trade_directions if d == "SHORT")
        long_wins = sum(1 for i, d in enumerate(trade_directions)
                        if d == "LONG" and i < len(profit_per_trade) and profit_per_trade[i] > 0)
        short_wins = sum(1 for i, d in enumerate(trade_directions)
                         if d == "SHORT" and i < len(profit_per_trade) and profit_per_trade[i] > 0)
        long_wr = long_wins / n_long if n_long > 0 else 0
        short_wr = short_wins / n_short if n_short > 0 else 0

    # Plot erstellen - 3 Subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), height_ratios=[3, 1, 1])

    # === Equity Kurve (logarithmisch) ===
    ax1.plot(eq, color="blue", linewidth=1.5)
    ax1.fill_between(range(len(eq)), eq, alpha=0.3)
    _setup_log_yaxis(ax1, eq)

    # Jahresrendite berechnen
    final_equity = eq[-1] if eq else 100.0
    bars_per_year = 24 * 250 if TIMEFRAME == "HOUR" else 96 * 250
    wf = result.get("walk_forward", {})
    fold_details = wf.get("fold_details", [])
    if fold_details:
        total_test_bars = sum(f.get("test_size", OOS_SIZE) for f in fold_details)
    else:
        total_test_bars = WALK_FORWARD_FOLDS * OOS_SIZE
    years = total_test_bars / bars_per_year if bars_per_year > 0 else 1
    if final_equity > 0 and years > 0:
        annual_return = ((final_equity / 100.0) ** (1 / years) - 1) * 100
    else:
        annual_return = -100

    # Titel zusammenbauen
    win_rate = result.get('win_rate', 0)
    sharpe = result.get('sharpe', 0)

    # CT-Werte für Titel
    if config.get("separate_long_short") and config.get("ct_long") != config.get("ct_short"):
        ct_str = f"CT: L={config.get('ct_long', 0):.2f}/S={config.get('ct_short', 0):.2f}"
    else:
        ct_str = f"CT: {config.get('conf_thresh', 0):.2f}"

    title_line1 = f"{sym} | WR: {win_rate:.1%} | RRR: {rrr:.2f} | "
    if sharpe:
        title_line1 += f"Sharpe: {sharpe:.2f} | "
    title_line1 += f"MaxDD: {max_dd*100:.0f}% | {annual_return:+.0f}%/y ({years:.1f}y)"

    if trade_directions:
        title_line2 = f"\nLong: {n_long} ({long_wr:.0%}) | Short: {n_short} ({short_wr:.0%}) | {ct_str}"
        ax1.set_title(title_line1 + title_line2)
    else:
        ax1.set_title(f"{title_line1} | {ct_str}")

    ax1.set_ylabel("Kapital (log)")
    ax1.grid(True, alpha=0.3)

    # === Drawdown ===
    ax2.fill_between(range(len(drawdowns)), drawdowns, color="red", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(max(drawdowns) * 1.1 if drawdowns else 1, 0)
    ax2.grid(True, alpha=0.3)

    # === Profit per Trade ===
    if trade_directions:
        # Mit Long/Short Unterscheidung
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
        ax3.set_xlabel("Trade # (grün/rot=Long, blau/orange=Short)")
    else:
        # Einfache Farben
        colors = ["green" if p > 0 else "red" for p in profit_per_trade]
        ax3.set_xlabel("Trade #")

    ax3.bar(range(len(profit_per_trade)), profit_per_trade, color=colors, alpha=0.7, width=1.0)
    ax3.axhline(y=0, color="black", linewidth=0.5)
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
        ax3.text(len(profit_per_trade) * 0.02, avg_win, f"Ø Win: {avg_win:.1f}",
                 fontsize=8, color="green", va="bottom")
    if losses:
        avg_loss = sum(losses) / len(losses)
        ax3.axhline(y=avg_loss, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax3.text(len(profit_per_trade) * 0.02, avg_loss, f"Ø Loss: {avg_loss:.1f}",
                 fontsize=8, color="red", va="top")

    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{plots_path}/{sym}.png", dpi=100)
    plt.close()

    if trade_directions:
        return n_long, n_short, long_wr, short_wr
    return None


# Wrapper für Abwärtskompatibilität
def create_incremental_plot(result, plots_path):
    """Wrapper für create_asset_plot (ohne Trade-Richtungen)."""
    return create_asset_plot(result, plots_path, trade_directions=None)


def create_elite_plot(e, plots_path, trade_directions, profit_per_trade, eq, drawdowns, max_dd, rrr):
    """
    Wrapper für create_asset_plot mit Trade-Richtungen.

    Hinweis: Die übergebenen profit_per_trade, eq, drawdowns, max_dd, rrr werden ignoriert
    und intern neu berechnet für Konsistenz.
    """
    return create_asset_plot(e, plots_path, trade_directions=trade_directions)
