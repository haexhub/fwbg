"""Tests für die zentrale Timeframe-Normalisierung (Single Source of Truth)."""
import pytest

from fwbg_sdk.enums import Timeframe


class TestTimeframeNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("DAY_1", Timeframe.D1),
            ("DAY", Timeframe.D1),
            ("1D", Timeframe.D1),
            ("D1", Timeframe.D1),
            ("1d", Timeframe.D1),
            ("HOUR_1", Timeframe.H1),
            ("HOUR", Timeframe.H1),
            ("HOUR_SWING", Timeframe.H1),
            ("1H", Timeframe.H1),
            ("1h", Timeframe.H1),
            ("HOUR_4", Timeframe.H4),
            ("4h", Timeframe.H4),
            ("MINUTE_15", Timeframe.M15),
            ("15M", Timeframe.M15),
            ("  minute_30  ", Timeframe.M30),
        ],
    )
    def test_from_str_aliases(self, raw, expected):
        assert Timeframe.from_str(raw) is expected

    def test_from_str_idempotent(self):
        assert Timeframe.from_str(Timeframe.H4) is Timeframe.H4

    def test_from_str_rejects_unknown(self):
        with pytest.raises(ValueError):
            Timeframe.from_str("FORTNIGHT")

    def test_canonical_matches_on_disk_naming(self):
        assert Timeframe.H1.canonical == "HOUR_1"
        assert Timeframe.D1.canonical == "DAY_1"
        assert Timeframe.M15.canonical == "MINUTE_15"

    def test_granularity(self):
        assert Timeframe.M5.granularity == "MINUTE"
        assert Timeframe.H1.granularity == "HOUR"
        assert Timeframe.D1.granularity == "DAY"
        assert Timeframe.W1.granularity == "WEEK"

    def test_minutes_ordering(self):
        ordered = sorted(Timeframe, key=lambda t: t.minutes)
        assert ordered[0] is Timeframe.M1
        assert ordered[-1] is Timeframe.W1
        assert Timeframe.H1.minutes == 60
        assert Timeframe.D1.minutes == 1440

    def test_resample_rule(self):
        assert Timeframe.H1.resample_rule == "1h"
        assert Timeframe.D1.resample_rule == "1D"
