import pandas as pd

from fwbg.core.registry import get_data_loader
from fwbg.plugins.custom.data_loading.jev_signal import JevSignalLoader


class _Ctx:
    def __init__(self, df):
        self.df = df


def test_plugin_registered():
    """jev_signal should be discoverable via the real registry (manifest.json
    + @register_data_loader wiring), not just importable directly."""
    cls = get_data_loader("jev_signal")
    assert cls is not None
    assert cls.name == "jev_signal"


def test_maps_raw_probability_columns_with_shift():
    df = pd.DataFrame(
        {
            "jev_official_is_long_win": [0.9, 0.2, 0.7],
            "jev_official_is_short_win": [0.1, 0.8, 0.3],
        }
    )
    ctx = _Ctx(df)

    JevSignalLoader().execute(ctx, provider="jev_official")

    # Mandatory 1-bar shift: row 0 is NaN, row 1 sees row 0's value, etc.
    assert pd.isna(ctx.df["_composed_signal_long"].iloc[0])
    assert ctx.df["_composed_signal_long"].iloc[1] == 0.9
    assert ctx.df["_composed_signal_short"].iloc[1] == 0.1


def test_missing_columns_produce_no_signal():
    df = pd.DataFrame({"C": [1.0, 1.1]})
    ctx = _Ctx(df)

    JevSignalLoader().execute(ctx, provider="jev_official")

    assert (ctx.df["_composed_signal_long"].fillna(0) == 0).all()
