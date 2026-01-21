import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# --- EINSTELLUNGEN ---
TRADE_LOG = "trade_history.csv"
OUTPUT_DIR = "stats_export"


def calculate_performance_metrics(df):
    """Berechnet die harten Fakten der Trading-Statistik."""
    if df.empty:
        return None

    # Basis-Metriken
    total_trades = len(df)
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]

    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_pnl = df["pnl"].sum()
    avg_trade = total_pnl / total_trades if total_trades > 0 else 0

    # Drawdown Berechnung
    cumulative = df["pnl"].cumsum()
    running_max = cumulative.cummax()
    drawdowns = cumulative - running_max
    max_drawdown = drawdowns.min()

    return {
        "summary": {
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "total_pnl_eur": round(total_pnl, 2),
            "avg_trade_pnl": round(avg_trade, 2),
            "max_drawdown_eur": round(max_drawdown, 2),
            "last_update": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "equity_curve": cumulative.tolist(),
    }


def run_analytics():
    if not os.path.exists(TRADE_LOG):
        print("❌ Keine Historie gefunden.")
        return

    # Ordner erstellen
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = pd.read_csv(TRADE_LOG)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    stats = calculate_performance_metrics(df)
    if not stats:
        return

    # --- EXPORT ALS JSON ---
    json_path = os.path.join(OUTPUT_DIR, "performance.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=4)

    # --- EXPORT ALS CSV (Zusammenfassung) ---
    pd.DataFrame([stats["summary"]]).to_csv(
        os.path.join(OUTPUT_DIR, "summary.csv"), index=False
    )

    # --- PLOTTING ---
    plt.figure(figsize=(12, 7))
    plt.style.use("dark_background")  # Sieht im Trading-Kontext oft besser aus

    plt.plot(
        df["timestamp"],
        df["pnl"].cumsum(),
        color="#00ff00",
        linewidth=2,
        label="Equity",
    )
    plt.fill_between(df["timestamp"], df["pnl"].cumsum(), color="#00ff00", alpha=0.1)

    # Text-Box mit Statistiken
    s = stats["summary"]
    stats_text = (
        f"Trades: {s['total_trades']}\n"
        f"Win-Rate: {s['win_rate_pct']}%\n"
        f"Profit Factor: {s['profit_factor']}\n"
        f"Max Drawdown: {s['max_drawdown_eur']}€"
    )

    plt.gca().text(
        0.02,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.1),
    )

    plt.title(f"Live Performance Dashboard - {pd.Timestamp.now().date()}")
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(OUTPUT_DIR, "equity_plot.png"))
    plt.show()

    print(f"✅ Statistik exportiert nach '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    run_analytics()
