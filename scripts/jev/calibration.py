import pandas as pd


def brier_score(predicted: pd.Series, actual: pd.Series) -> float:
    predicted, actual = predicted.align(actual, join="inner")
    valid = predicted.notna() & actual.notna()
    return float(((predicted[valid] - actual[valid]) ** 2).mean())


def accuracy_vs_baseline(
    predicted: pd.Series, actual: pd.Series, threshold: float
) -> dict[str, float]:
    predicted, actual = predicted.align(actual, join="inner")
    valid = predicted.notna() & actual.notna()
    predicted, actual = predicted[valid], actual[valid]

    predicted_class = (predicted >= threshold).astype(float)
    model_accuracy = float((predicted_class == actual).mean())
    baseline_class = float(actual.mean() >= 0.5)
    baseline_accuracy = float((actual == baseline_class).mean())
    return {"model_accuracy": model_accuracy, "baseline_accuracy": baseline_accuracy}


def agreement_rate(provider_a: pd.Series, provider_b: pd.Series, threshold: float) -> float:
    provider_a, provider_b = provider_a.align(provider_b, join="inner")
    valid = provider_a.notna() & provider_b.notna()
    provider_a, provider_b = provider_a[valid], provider_b[valid]

    class_a = provider_a >= threshold
    class_b = provider_b >= threshold
    return float((class_a == class_b).mean())
