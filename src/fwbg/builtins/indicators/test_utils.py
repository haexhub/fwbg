"""
Gemeinsame Test-Utilities für Lookahead Bias Detection.

Wird von allen Indikator-Plugins verwendet um konsistente
Lookahead-Tests durchzuführen.
"""
import numpy as np
import pandas as pd
from typing import List, Type

from fwbg.plugins import BaseIndicator


def create_test_ohlc(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Erstellt synthetische OHLC-Daten für Tests.

    Args:
        n: Anzahl Bars
        seed: Random Seed für Reproduzierbarkeit

    Returns:
        DataFrame mit O, H, L, C Spalten
    """
    np.random.seed(seed)
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.005)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.002)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.002)
    open_price = np.roll(close, 1) * (1 + np.random.randn(n) * 0.001)
    open_price[0] = 100

    # Ensure H >= max(O,C) and L <= min(O,C)
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    return pd.DataFrame({
        'O': open_price, 'H': high, 'L': low, 'C': close,
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))


def check_no_current_bar_dependency(
    indicator: BaseIndicator,
    df: pd.DataFrame,
    tolerance: float = 1e-10,
    exclude_patterns: List[str] = None,
) -> List[str]:
    """
    Prüft ob Features vom aktuellen Bar abhängen.

    Features bei Bar i dürfen sich NICHT ändern wenn nur Bar i modifiziert wird.

    Args:
        indicator: Indikator-Instanz
        df: Test-DataFrame
        tolerance: Toleranz für Float-Vergleiche
        exclude_patterns: Feature-Patterns die vom Test ausgeschlossen werden
            (z.B. Event-Features die auf Existenz von Events reagieren)

    Returns:
        Liste von Fehlermeldungen (leer wenn OK)
    """
    if exclude_patterns is None:
        # Event-Features ausschließen - diese reagieren auf Existenz von Events,
        # nicht auf den aktuellen Bar-Wert. Wenn ein extremer Bar einen neuen
        # Event-Typ triggert, ändert sich das "bars_since" für ALLE Bars.
        exclude_patterns = ['bars_since_', 'streak']

    result_orig = indicator.compute(df.copy())

    # Modifiziere nur den letzten Bar
    df_mod = df.copy()
    df_mod.loc[df_mod.index[-1], 'C'] *= 1.5
    df_mod.loc[df_mod.index[-1], 'H'] *= 1.6
    df_mod.loc[df_mod.index[-1], 'L'] *= 0.9
    df_mod.loc[df_mod.index[-1], 'O'] *= 1.2
    result_mod = indicator.compute(df_mod)

    errors = []
    feature_cols = indicator.get_feature_columns()

    for col in feature_cols:
        if col not in result_orig.columns or col not in result_mod.columns:
            continue

        # Skip excluded patterns
        if any(pattern in col for pattern in exclude_patterns):
            continue

        val_orig = result_orig[col].iloc[-1]
        val_mod = result_mod[col].iloc[-1]

        if pd.isna(val_orig) and pd.isna(val_mod):
            continue

        if pd.isna(val_orig) != pd.isna(val_mod):
            errors.append(f"'{col}': NaN mismatch (orig={val_orig}, mod={val_mod})")
        elif abs(val_orig - val_mod) > tolerance:
            errors.append(
                f"'{col}': changed with current bar "
                f"(orig={val_orig:.6f}, mod={val_mod:.6f})"
            )

    return errors


def check_no_future_leakage(
    indicator: BaseIndicator,
    df: pd.DataFrame,
    spike_position: int = 150,
    check_range: int = 50,
    tolerance: float = 1e-10,
    exclude_patterns: List[str] = None,
) -> List[str]:
    """
    Prüft ob ein zukünftiger Spike vergangene Features beeinflusst.

    Args:
        indicator: Indikator-Instanz
        df: Test-DataFrame
        spike_position: Position des Spikes
        check_range: Wie viele Bars vor dem Spike prüfen
        tolerance: Toleranz für Float-Vergleiche
        exclude_patterns: Feature-Patterns die vom Test ausgeschlossen werden

    Returns:
        Liste von Fehlermeldungen (leer wenn OK)
    """
    if exclude_patterns is None:
        # Event-Features ausschließen - diese reagieren auf Existenz von Events
        exclude_patterns = ['bars_since_', 'streak']

    # Ohne Spike
    result_normal = indicator.compute(df.copy())

    # Mit Spike
    df_spike = df.copy()
    df_spike.loc[df_spike.index[spike_position], 'C'] *= 2.0
    df_spike.loc[df_spike.index[spike_position], 'H'] *= 2.5
    df_spike.loc[df_spike.index[spike_position], 'L'] *= 0.8
    result_spike = indicator.compute(df_spike)

    errors = []
    feature_cols = indicator.get_feature_columns()

    for col in feature_cols:
        if col not in result_normal.columns or col not in result_spike.columns:
            continue

        # Skip excluded patterns
        if any(pattern in col for pattern in exclude_patterns):
            continue

        for i in range(spike_position - check_range, spike_position - 1):
            if i < 0:
                continue

            val_normal = result_normal[col].iloc[i]
            val_spike = result_spike[col].iloc[i]

            if pd.isna(val_normal) and pd.isna(val_spike):
                continue

            if pd.isna(val_normal) != pd.isna(val_spike):
                errors.append(f"'{col}' at bar {i}: NaN mismatch")
                break
            elif abs(val_normal - val_spike) > tolerance:
                errors.append(
                    f"'{col}' at bar {i}: future spike affected past "
                    f"(diff={abs(val_normal - val_spike):.2e})"
                )
                break

    return errors


def check_first_bar_nan(
    indicator: BaseIndicator,
    df: pd.DataFrame,
    min_nan_ratio: float = 0.5,
) -> List[str]:
    """
    Prüft ob der erste Bar NaN ist (wegen shift).

    Args:
        indicator: Indikator-Instanz
        df: Test-DataFrame
        min_nan_ratio: Mindestanteil der Features die NaN sein sollten

    Returns:
        Liste von Fehlermeldungen (leer wenn OK)
    """
    result = indicator.compute(df)
    feature_cols = indicator.get_feature_columns()

    non_nan_features = []
    for col in feature_cols:
        if col in result.columns:
            if not pd.isna(result[col].iloc[0]):
                non_nan_features.append(f"'{col}': value={result[col].iloc[0]}")

    nan_count = len(feature_cols) - len(non_nan_features)
    nan_ratio = nan_count / len(feature_cols) if feature_cols else 1.0

    if nan_ratio < min_nan_ratio:
        return [
            f"Only {nan_ratio:.0%} of features are NaN at first bar "
            f"(expected >= {min_nan_ratio:.0%}). Non-NaN features:\n" +
            "\n".join(non_nan_features[:10])
        ]

    return []


class LookaheadBiasTestMixin:
    """
    Mixin-Klasse für Lookahead Bias Tests.

    Verwendung:
        class TestMyIndicatorLookahead(LookaheadBiasTestMixin):
            indicator_class = MyIndicator

    Die Mixin-Klasse stellt automatisch die drei Standard-Tests bereit:
    - test_no_current_bar_dependency
    - test_no_future_leakage
    - test_first_bar_nan
    """

    indicator_class: Type[BaseIndicator] = None
    test_data_size: int = 200
    test_seed: int = 42

    def get_indicator(self) -> BaseIndicator:
        """Erstellt eine Indikator-Instanz."""
        if self.indicator_class is None:
            raise NotImplementedError("indicator_class must be set")
        return self.indicator_class()

    def get_test_df(self) -> pd.DataFrame:
        """Erstellt Test-DataFrame."""
        return create_test_ohlc(n=self.test_data_size, seed=self.test_seed)

    def test_no_current_bar_dependency(self):
        """Features dürfen nicht vom aktuellen Bar abhängen."""
        indicator = self.get_indicator()
        df = self.get_test_df()
        errors = check_no_current_bar_dependency(indicator, df)
        assert not errors, f"Lookahead bias detected:\n" + "\n".join(errors)

    def test_no_future_leakage(self):
        """Zukünftige Daten dürfen vergangene Features nicht beeinflussen."""
        indicator = self.get_indicator()
        df = self.get_test_df()
        errors = check_no_future_leakage(indicator, df)
        assert not errors, f"Future leakage detected:\n" + "\n".join(errors[:10])

    def test_first_bar_nan(self):
        """Der erste Bar sollte für die meisten Features NaN sein."""
        indicator = self.get_indicator()
        df = self.get_test_df()
        errors = check_first_bar_nan(indicator, df)
        assert not errors, errors[0] if errors else ""
