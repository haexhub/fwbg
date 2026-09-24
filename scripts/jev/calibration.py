import pandas as pd


def brier_score(predicted: pd.Series, actual: pd.Series) -> float:
    return float(((predicted - actual) ** 2).mean())


def accuracy_vs_baseline(predicted: pd.Series, actual: pd.Series, threshold: float) -> dict:
    predicted_class = (predicted >= threshold).astype(float)
    model_accuracy = float((predicted_class == actual).mean())
    baseline_class = float(actual.mean() >= 0.5)
    baseline_accuracy = float((actual == baseline_class).mean())
    return {"model_accuracy": model_accuracy, "baseline_accuracy": baseline_accuracy}


def agreement_rate(provider_a: pd.Series, provider_b: pd.Series, threshold: float) -> float:
    class_a = provider_a >= threshold
    class_b = provider_b >= threshold
    return float((class_a == class_b).mean())
