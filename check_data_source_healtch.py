import pandas as pd
import os
import yaml


def generate_health_report():
    with open("mapping.yaml", "r") as f:
        mapping = yaml.safe_load(f).get("markets", {})

    report = []

    for epic, ticker in mapping.items():
        safe_name = epic.replace(".", "_")
        row = {"Markt": ticker}

        for src in ["yahoo", "stooq"]:
            path = f"data/{src}/{safe_name}.csv"
            if os.path.exists(path):
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                row[f"{src.capitalize()} Zeilen"] = len(df)
                row[f"{src.capitalize()} Start"] = df.index[0].date()
            else:
                row[f"{src.capitalize()} Zeilen"] = 0
                row[f"{src.capitalize()} Start"] = "N/A"

        report.append(row)

    df_report = pd.DataFrame(report)
    print("\n" + "=" * 90)
    print("📊 DATEN-QUALITÄTS-BERICHT (QUELLE VERGLEICH)")
    print("=" * 90)
    print(df_report.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    generate_health_report()
