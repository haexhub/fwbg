"""
Lookahead-Bias Tests für alle Indikatoren.

Beweist, dass kein Indikator zukünftige Daten verwendet.

Methode (Future-Scramble-Test):
    1. Berechne Indikator auf vollständigen Daten.
    2. Permutiere alle Bars NACH einem Trennpunkt zufällig (zerstört Zukunft).
    3. Berechne Indikator erneut auf den manipulierten Daten.
    4. Alle Werte VOR dem Trennpunkt müssen identisch sein.

    Wenn ein Indikator diesen Test NICHT besteht, hat er Lookahead-Bias:
    er verwendet zukünftige Preise bei der Berechnung vergangener Werte.
"""
import numpy as np
import pandas as pd
import pytest

# Trigger plugin discovery BEFORE parametrize decorators are evaluated
from fwbg.pipeline import get_registry as _get_registry

_get_registry().auto_discover()

from fwbg_sdk import INDICATOR_REGISTRY, assert_features_shifted  # noqa: E402


# Indikatoren die ML-Training oder externe Infrastruktur benötigen
_SKIP = {
    "adversarial_validation",  # Benötigt ML-Modell (Classifier)
    "autoencoder_features",    # Benötigt neuronales Netz
    "topological_features",    # TDA: sehr rechenintensiv
}

# Indikator-Liste für parametrize (einmal evaluiert bei Modul-Import)
_ALL_INDICATORS = [
    (name, cls)
    for name, cls in sorted(INDICATOR_REGISTRY.items())
    if name not in _SKIP
]


def _make_ohlcv(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Erstellt realistische OHLCV-Daten für Tests."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.001, n)
    close = 1.1000 + np.cumsum(returns)
    close = np.clip(close, 0.5, None)

    high_offset = np.abs(rng.normal(0, 0.0005, n))
    low_offset = np.abs(rng.normal(0, 0.0005, n))
    open_offset = rng.normal(0, 0.0003, n)

    opens = close + open_offset
    highs = np.maximum(close, opens) + high_offset
    lows = np.minimum(close, opens) - low_offset
    volume = rng.integers(100, 10000, n).astype(float)

    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {"O": opens, "H": highs, "L": lows, "C": close, "V": volume},
        index=idx,
    )


@pytest.fixture(scope="module")
def sample_ohlc() -> pd.DataFrame:
    return _make_ohlcv(n=600)


def _get_new_cols(result: pd.DataFrame, base: pd.DataFrame) -> list[str]:
    """Gibt alle Spalten zurück die der Indikator neu hinzugefügt hat."""
    return [c for c in result.columns if c not in base.columns]


class TestFirstRowIsNaN:
    """
    Trivialer Nachweis: shift_features() wurde aufgerufen.

    Wenn shift_features() korrekt verwendet wurde, muss die erste Zeile
    jedes Feature-Columns NaN sein (der Shift lässt Bar 0 leer).
    """

    @pytest.mark.parametrize("name,cls", _ALL_INDICATORS, ids=[n for n, _ in _ALL_INDICATORS])
    def test_first_row_nan(self, name: str, cls, sample_ohlc: pd.DataFrame):
        indicator = cls()
        result = indicator.compute(sample_ohlc.copy())
        new_cols = _get_new_cols(result, sample_ohlc)

        if not new_cols:
            pytest.skip(
                f"Indikator '{name}' erzeugt ohne Vorspalten keine Features "
                f"(Meta-Indikator, braucht Input-Spalten von anderen Indikatoren)."
            )
        assert_features_shifted(result, new_cols)


class TestFutureScramble:
    """
    Kernbeweis: Zukünftige Daten dürfen vergangene Berechnungen nicht beeinflussen.

    Wenn ein Indikator Lookahead-Bias hat, ändert das Permutieren zukünftiger
    Bars die bereits berechneten vergangenen Werte. Korrekte Indikatoren
    bleiben davon unberührt.
    """

    # Warmup: viele Indikatoren brauchen 200+ Bars (z.B. EMA-200).
    # Wir prüfen nur Bars NACH dem Warmup (>= _WARMUP_BARS).
    _WARMUP_BARS = 300
    # Trennpunkt: Bars 300-449 = Vergangenheit (geprüft), Bars 450-599 = Zukunft (permutiert)
    _SPLIT = 450

    @pytest.mark.parametrize("name,cls", _ALL_INDICATORS, ids=[n for n, _ in _ALL_INDICATORS])
    def test_scrambled_future_leaves_past_unchanged(
        self, name: str, cls, sample_ohlc: pd.DataFrame
    ):
        """
        Beweist: Indikator ist kausal - vergangene Werte sind deterministisch
        aus vergangenen Preisen berechenbar, ohne Kenntnis der Zukunft.
        """
        split = self._SPLIT
        warmup = self._WARMUP_BARS

        indicator = cls()

        # Berechnung auf Originaldaten
        result_original = indicator.compute(sample_ohlc.copy())
        new_cols = _get_new_cols(result_original, sample_ohlc)
        if not new_cols:
            pytest.skip(
                f"Indikator '{name}' erzeugt ohne Vorspalten keine Features "
                f"(Meta-Indikator, braucht Input-Spalten von anderen Indikatoren)."
            )

        # Zukunft permutieren: Bars ab `split` in zufällige Reihenfolge bringen
        df_scrambled = sample_ohlc.copy()
        future_idx = df_scrambled.index[split:]
        scrambled_future = df_scrambled.loc[future_idx].sample(
            frac=1, random_state=7
        )
        scrambled_future.index = future_idx  # Index erhalten, nur Werte tauschen
        df_scrambled.loc[future_idx] = scrambled_future.values

        # Berechnung auf manipulierten Daten
        result_scrambled = indicator.compute(df_scrambled)

        # Alle Werte VOR dem Split müssen identisch sein
        for col in new_cols:
            orig = result_original[col].iloc[warmup:split]
            scram = result_scrambled[col].iloc[warmup:split]

            # Nur Positionen prüfen, wo mindestens einer der Werte nicht NaN ist
            both_nan = orig.isna() & scram.isna()
            check_mask = ~both_nan

            if check_mask.sum() == 0:
                # Alle NaN → Warmup deckt den gesamten geprüften Bereich ab, OK
                continue

            orig_check = orig[check_mask]
            scram_check = scram[check_mask]

            # NaN-Muster muss übereinstimmen
            nan_mismatch = orig_check.isna() != scram_check.isna()
            if nan_mismatch.any():
                first_mismatch = nan_mismatch.idxmax()
                pytest.fail(
                    f"LOOKAHEAD-BIAS in '{name}' Spalte '{col}':\n"
                    f"  NaN-Muster unterschiedlich bei Bar {first_mismatch}.\n"
                    f"  Original: {orig_check[first_mismatch]}, "
                    f"  Nach Scramble: {scram_check[first_mismatch]}\n"
                    f"  Das Permutieren zukünftiger Bars hat vergangene NaN-Werte verändert."
                )

            # Werte-Übereinstimmung prüfen (nur non-NaN Positionen)
            valid = check_mask & orig.notna() & scram.notna()
            if valid.sum() == 0:
                continue

            orig_valid = orig[valid]
            scram_valid = scram[valid]

            mismatches = ~np.isclose(
                orig_valid.values, scram_valid.values, rtol=1e-9, atol=1e-12, equal_nan=True
            )
            if mismatches.any():
                first_idx = orig_valid.index[mismatches][0]
                bar_num = sample_ohlc.index.get_loc(first_idx)
                pytest.fail(
                    f"LOOKAHEAD-BIAS in '{name}' Spalte '{col}':\n"
                    f"  Bar {bar_num} ({first_idx}) unterschiedlich nach Scramble der Zukunft.\n"
                    f"  Original:     {orig_valid[first_idx]:.10f}\n"
                    f"  Nach Scramble: {scram_valid[first_idx]:.10f}\n"
                    f"  Differenz:    {abs(orig_valid[first_idx] - scram_valid[first_idx]):.2e}\n"
                    f"  Ursache: Indikator verwendet Daten jenseits von Bar {split}."
                )
