"""
Purged CV und Sample Uniqueness Weights (López de Prado, AFML Ch. 4).

Berechnet Sample Weights basierend auf Label-Uniqueness:
- Samples die zeitlich wenig mit anderen überlappen bekommen höheres Gewicht
- Gewichtet XGBoost Training, sodass einzigartige Samples mehr zählen
"""
import numpy as np
from numba import njit


@njit(cache=True)
def compute_concurrent_labels(n_samples: int, durations: np.ndarray) -> np.ndarray:
    """
    Zählt überlappende Labels an jedem Zeitpunkt.

    Für jeden Sample i mit Duration d erstreckt sich sein Label über [i, i+d).
    An jedem Zeitpunkt t zählen wir wie viele Labels aktiv sind.

    Args:
        n_samples: Anzahl Samples
        durations: Duration-Array (wie viele Bars jeder Trade dauert)

    Returns:
        concurrent: Array mit Anzahl aktiver Labels pro Zeitpunkt
    """
    concurrent = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        d = int(durations[i])
        if d > 0:
            end = min(i + d, n_samples)
            for t in range(i, end):
                concurrent[t] += 1.0
    return concurrent


@njit(cache=True)
def _compute_uniqueness(
    n_samples: int,
    durations: np.ndarray,
    concurrent: np.ndarray,
) -> np.ndarray:
    """
    Berechnet Average Uniqueness pro Sample.

    Uniqueness(i) = mean(1/concurrent[t]) für t in [i, i+duration[i])

    Hohe Uniqueness = Sample überlappt wenig mit anderen → höheres Gewicht.

    Args:
        n_samples: Anzahl Samples
        durations: Duration-Array
        concurrent: Concurrent-Label-Array

    Returns:
        weights: Uniqueness-Weight pro Sample
    """
    weights = np.ones(n_samples, dtype=np.float64)
    for i in range(n_samples):
        d = int(durations[i])
        if d > 0:
            end = min(i + d, n_samples)
            total = 0.0
            count = 0
            for t in range(i, end):
                if concurrent[t] > 0:
                    total += 1.0 / concurrent[t]
                    count += 1
            if count > 0:
                weights[i] = total / count
    return weights


def compute_sample_weights(
    durations_long: np.ndarray,
    durations_short: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Berechnet Sample Weights via Label-Uniqueness.

    Nimmt das Maximum von Long/Short Durations (konservativste Schätzung)
    und normalisiert Gewichte so dass sum(weights) = n_samples.

    Args:
        durations_long: Duration-Array für Long-Trades
        durations_short: Duration-Array für Short-Trades
        n_samples: Anzahl Samples

    Returns:
        Normalisierte Sample Weights (sum = n_samples)
    """
    durations = np.maximum(
        durations_long[:n_samples],
        durations_short[:n_samples],
    )
    concurrent = compute_concurrent_labels(n_samples, durations)
    weights = _compute_uniqueness(n_samples, durations, concurrent)

    w_sum = weights.sum()
    if w_sum > 0:
        weights = weights * (n_samples / w_sum)
    return weights
