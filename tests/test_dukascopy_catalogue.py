"""Tests for the Dukascopy instrument catalogue powering the dashboard UI."""

import pytest

pytest.importorskip("dukascopy_python")

from fwbg.data.dukascopy import instrument_catalogue, resolve_instrument


def test_catalogue_non_empty_and_well_formed():
    cat = instrument_catalogue()
    assert len(cat) > 1000  # ~1288 resolvable instruments with history metadata

    for entry in cat:
        assert set(entry) == {"symbol", "id", "description", "group", "historyStart"}
        assert entry["symbol"] and entry["id"] and entry["description"]
        assert set(entry["historyStart"]) == {"minute", "hourly", "daily"}


def test_catalogue_contains_eurusd_with_expected_history():
    by_symbol = {e["symbol"]: e for e in instrument_catalogue()}
    eurusd = by_symbol["EURUSD"]
    assert eurusd["id"] == "EUR/USD"
    assert eurusd["group"] == "Forex"
    # Minute candles start far later than daily candles for EUR/USD.
    assert eurusd["historyStart"]["minute"].startswith("2003")
    assert eurusd["historyStart"]["daily"].startswith("1973")


def test_every_catalogue_symbol_is_downloadable():
    # The catalogue must never offer a symbol the downloader can't resolve.
    for entry in instrument_catalogue():
        resolve_instrument(entry["symbol"])  # raises DukascopyError if unknown


def test_groups_cover_known_asset_classes():
    groups = {e["group"] for e in instrument_catalogue()}
    assert {"Forex", "Krypto", "Rohstoffe", "Indizes"} <= groups


def test_requests_get_gets_default_timeout():
    """dukascopy_python omits timeouts on its requests.get, so a stalled
    connection hangs forever. The module installs a proxy that injects a default
    timeout (turning a stall into a retryable error) while delegating everything
    else and never overriding an explicit timeout."""
    import dukascopy_python as dk

    from fwbg.data.dukascopy import _HTTP_TIMEOUT, _TimeoutRequests

    # The proxy is installed at import time onto the library's requests handle.
    assert isinstance(dk.requests, _TimeoutRequests)

    calls: list[dict] = []

    class _FakeReal:
        SENTINEL = object()

        def get(self, *args, **kwargs):
            calls.append(kwargs)

    proxy = _TimeoutRequests(_FakeReal(), _HTTP_TIMEOUT)
    proxy.get("http://x")
    assert calls[-1]["timeout"] == _HTTP_TIMEOUT  # injected when absent
    proxy.get("http://x", timeout=5)
    assert calls[-1]["timeout"] == 5  # explicit timeout preserved
    assert proxy.SENTINEL is _FakeReal.SENTINEL  # other attributes delegate


def test_measured_spread_overrides_configured_spread(tmp_path):
    # A spread measured during download must take precedence over the hand-tuned
    # DEFAULT_ASSETS value (and the 0.0002 fallback for unknown symbols).
    import fwbg.data.assets as assets
    from fwbg.core.data_sources import get_data_root, set_data_root

    previous_root = get_data_root()
    try:
        set_data_root(tmp_path)
        assets._OVERRIDES_CACHE = None  # drop mtime cache for the fresh data root

        # Known asset: EURUSD default is 0.00018 -> measured override wins.
        assets.save_asset_spread("EURUSD", 0.00031)
        assert abs(assets.get_asset("EURUSD").spread - 0.00031) < 1e-12

        # Unknown asset: 0.0002 fallback -> measured override wins.
        assets.save_asset_spread("SOMEEXOTIC", 0.0042)
        assert abs(assets.get_asset("SOMEEXOTIC").spread - 0.0042) < 1e-12

        # Manual override wins over the measured value...
        assets.save_asset_spread("EURUSD", 0.00099, manual=True)
        assert abs(assets.get_asset("EURUSD").spread - 0.00099) < 1e-12
        # ...and a later re-measure must not clobber the manual override.
        assets.save_asset_spread("EURUSD", 0.00025, manual=False)
        assert abs(assets.get_asset("EURUSD").spread - 0.00099) < 1e-12
    finally:
        set_data_root(previous_root)
        assets._OVERRIDES_CACHE = None


def test_per_asset_spread_set_list_and_clear(tmp_path):
    import fwbg.data.assets as assets
    from fwbg.core.data_sources import get_data_root, set_data_root

    previous_root = get_data_root()
    try:
        set_data_root(tmp_path)
        assets._OVERRIDES_CACHE = None

        assets.save_asset_spread("EURUSD", 0.00004, manual=False)  # measured p90
        assets.set_manual_spread("EURUSD", 0.0007)                 # user override

        listing = {e["symbol"]: e for e in assets.list_asset_spreads()}
        assert listing["EURUSD"]["measured"] == 0.00004
        assert listing["EURUSD"]["manual"] == 0.0007
        assert listing["EURUSD"]["effective"] == 0.0007

        # Clearing the override falls back to the measured value.
        assets.set_manual_spread("EURUSD", None)
        listing = {e["symbol"]: e for e in assets.list_asset_spreads()}
        assert listing["EURUSD"]["manual"] is None
        assert listing["EURUSD"]["effective"] == 0.00004
        assert abs(assets.get_asset("EURUSD").spread - 0.00004) < 1e-12
    finally:
        set_data_root(previous_root)
        assets._OVERRIDES_CACHE = None
