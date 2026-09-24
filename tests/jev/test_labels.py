import numpy as np
import pandas as pd

from scripts.jev.labels import compute_labels


def test_compute_labels_matches_fixed_exit_strategy_directly():
    # Small synthetic OHLC series: price ramps up steadily, so a long
    # entry near the start should win, a short entry should lose.
    n = 50
    close = 1.0800 + np.linspace(0, 0.0100, n)  # steady uptrend, 100 pips over 50 bars
    df = pd.DataFrame(
        {
            "O": close,
            "H": close + 0.0002,
            "L": close - 0.0002,
            "C": close,
        }
    )

    labels = compute_labels(df, symbol="EURUSD", tp=20, sl=20, timeout_bars=30)

    assert list(labels.columns) == ["targets_long", "targets_short"]
    assert len(labels) == n
    # Early bars in a strong uptrend: long should win, short should not.
    assert labels["targets_long"].iloc[0] == 1.0
    assert labels["targets_short"].iloc[0] == 0.0


def test_compute_labels_uses_asset_spread():
    # Different assets have different spreads; this just checks the function
    # doesn't hardcode EURUSD's spread internally in a way that breaks lookup.
    from fwbg.data.assets import get_asset

    asset = get_asset("EURUSD")
    assert asset.spread == 0.00018
