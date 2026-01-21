import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_parameter_heatmap(market_name, source="stooq"):
    # Wir suchen in der History nach den letzten Läufen für diesen Markt
    if not os.path.exists("optimization_history.csv"):
        print("Keine Historie gefunden.")
        return

    df = pd.read_csv("optimization_history.csv")
    subset = df[(df["Market"] == market_name) & (df["Source"] == source)]

    if subset.empty:
        print(f"Keine Daten für {market_name} in {source} gefunden.")
        return

    # Pivot-Tabelle für die Heatmap (Thresholds vs. Profit)
    pivot = subset.pivot_table(
        index="Best_Conf", columns="Best_ADX", values="Total_Score", aggfunc="mean"
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title(f"Stabilitäts-Plateau: {market_name} ({source.upper()})")
    plt.show()


if __name__ == "__main__":
    # Teste es für EUR/USD
    plot_parameter_heatmap("EUR/USD Mini", "stooq")
